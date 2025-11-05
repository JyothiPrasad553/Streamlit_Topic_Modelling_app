# app.py
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from datetime import date, datetime
import re
import time
import io
import zipfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import matplotlib.pyplot as plt
from email.mime.base import MIMEBase
from email import encoders

# NLP & ML
import multiprocessing as mp
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_recall_fscore_support

# Optional LLM (OpenAI)
import openai

# Database + Auth (local modules)
from database import ensure_users_table, get_user, update_user, insert_user, get_connection
from auth import create_user, authenticate_user, update_password
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# --------------------
# Utilities & helpers
# --------------------
def valid_password(pw: str) -> bool:
    return (
        len(pw) >= 6
        and re.search(r"[A-Z]", pw)
        and re.search(r"[a-z]", pw)
        and re.search(r"\d", pw)
        and re.search(r"[!@#$%^&*()_+=\\-]", pw)
    )

def set_bg(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{encoded}");
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

# --------------------
# Text processing
# --------------------
analyzer = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def unsupervised_sentiment(text: str) -> str:
    score = analyzer.polarity_scores(text)
    c = score["compound"]
    if c > 0.05:
        return "Positive"
    elif c < -0.05:
        return "Negative"
    else:
        return "Neutral"

def llm_sentiment(text: str, model="gpt-3.5-turbo"):
    prompt = (
        "Classify this text as Positive, Negative or Neutral. "
        "Answer with exactly one word: Positive, Negative or Neutral.\n\n"
        f"Text: \"{text}\""
    )
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=10,
        temperature=0.0
    )
    label = resp["choices"][0]["message"]["content"].strip().split()[0]
    if label.lower().startswith("pos"):
        return "Positive"
    if label.lower().startswith("neg"):
        return "Negative"
    return "Neutral"

def process_item(args):
    text, method = args
    text_clean = clean_text(text)
    try:
        if method == "llm":
            if not OPENAI_API_KEY:
                raise RuntimeError("OpenAI API key not configured.")
            lbl = llm_sentiment(text_clean)
            return {"text": text, "text_clean": text_clean, "sentiment": lbl}
        else:
            lbl = unsupervised_sentiment(text_clean)
            return {"text": text, "text_clean": text_clean, "sentiment": lbl}
    except Exception as e:
        return {"text": text, "text_clean": text_clean, "sentiment": "Error: " + str(e)}

def parallel_process_texts(texts, method="unsupervised", processes=None):
    start = time.time()
    if processes is None:
        processes = max(1, mp.cpu_count() - 1)
    args = [(t, method) for t in texts]
    with mp.Pool(processes) as pool:
        results = list(pool.imap(process_item, args, chunksize=8))
    df_out = pd.DataFrame(results)
    elapsed = time.time() - start
    return df_out, elapsed

def cluster_topics(df_text_clean, n_clusters=5):
    tv = TfidfVectorizer(min_df=2, stop_words="english", max_features=2000)
    X = tv.fit_transform(df_text_clean)
    if X.shape[0] < 2 or X.shape[1] < 2:
        return None, None, None
    k = min(n_clusters, X.shape[0])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else None
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tv.get_feature_names_out()
    top_terms = {}
    for i in range(k):
        top_terms[i] = [terms[ind] for ind in order_centroids[i, :10]]
    return labels, sil, top_terms

def save_results_csv(df, base_name="text_processing_results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{base_name}_{ts}.csv"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, fname)
    df.to_csv(csv_path, index=False)
    zip_path = csv_path.replace(".csv", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=os.path.basename(csv_path))
    return csv_path, zip_path

def email_file(receiver_email, subject, body_text, attachment_paths):
    """
    Send an email with one or multiple attachments.
    attachment_paths: list of file paths to attach.
    """
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

    if not EMAIL_SENDER or not EMAIL_APP_PASSWORD:
        raise RuntimeError("Email credentials not configured in environment (.env).")

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain"))

    for attachment_path in attachment_paths:
        if not os.path.exists(attachment_path):
            print(f"⚠️ Skipping missing file: {attachment_path}")
            continue
        with open(attachment_path, "rb") as f:
            mime_type, _ = mimetypes.guess_type(attachment_path)
            if mime_type is None:
                mime_type = "application/octet-stream"
            main, sub = mime_type.split("/")
            mimepart = MIMEBase(main, sub)
            mimepart.set_payload(f.read())
            encoders.encode_base64(mimepart)
            mimepart.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(attachment_path)}"'
            )
            msg.attach(mimepart)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_SENDER, receiver_email, msg.as_string())

    print(f"✅ Email sent successfully to {receiver_email} with {len(attachment_paths)} attachment(s).")
def generate_comparison_graph(times_dict, acc_dict, output_path="outputs/performance_comparison.png"):
    os.makedirs("outputs", exist_ok=True)
    methods = list(times_dict.keys())
    plt.figure(figsize=(6, 4))
    for m in methods:
        if acc_dict[m] is not None:
            plt.scatter(times_dict[m], acc_dict[m], label=m, s=100)
            plt.text(times_dict[m], acc_dict[m]+0.5, m, fontsize=10)
    plt.xlabel("Prediction Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.title("Prediction Time vs Accuracy Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# --------------------
# Streamlit UI
# --------------------
ensure_users_table()
st.set_page_config(page_title="Toonify", layout="centered")

if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["email"] = None
if "forgot_pw" not in st.session_state:
    st.session_state["forgot_pw"] = False
if "editing_profile" not in st.session_state:
    st.session_state["editing_profile"] = False

min_date = date(1900, 1, 1)
max_date = date.today()

page = st.session_state["page"]
st.title("🎨 Toonify - (Text Processing Edition)")

# -------------------- HOME --------------------
if page == "Home":
    set_bg("images/home.jpg") if os.path.exists("images/home.jpg") else None
    st.markdown("## 🥳 Welcome to Toonify! (Now for Text Processing too)")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📝 Signup"):
            st.session_state["page"] = "Signup"
            st.rerun()
    with c2:
        if st.button("🔑 Login"):
            st.session_state["page"] = "Login"
            st.rerun()
    with c3:
        if st.button("👤 My Account"):
            st.session_state["page"] = "My Account"
            st.rerun()

# -------------------- SIGNUP --------------------
elif page == "Signup":
    set_bg("images/signup.jpg") if os.path.exists("images/signup.jpg") else None
    st.subheader("📝 Create Account")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    full_name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth", min_value=min_date, max_value=max_date)
    address = st.text_area("Address")
    pincode = st.text_input("Pincode (6 digits)", max_chars=6)
    gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
    email = st.text_input("Gmail")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Signup"):
            if len(pincode) != 6 or not pincode.isdigit():
                st.error("❌ Pincode must be exactly 6 digits.")
            elif not email.endswith("@gmail.com"):
                st.error("❌ Enter a valid Gmail address.")
            elif not valid_password(password):
                st.error("❌ Weak password. Use upper, lower, digit & special char.")
            elif password != confirm:
                st.error("❌ Passwords do not match.")
            else:
                ok, msg = create_user(first_name, last_name, full_name, dob, address, pincode, gender, email, password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    with col2:
        if st.button("➡ Go to Login"):
            st.session_state["page"] = "Login"
            st.rerun()
    if st.button("⬅ Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()

# -------------------- LOGIN --------------------
elif page == "Login":
    set_bg("images/login.jpg") if os.path.exists("images/login.jpg") else None
    st.subheader("🔑 Login")
    email = st.text_input("Gmail", placeholder="Enter your Gmail")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔓 Login"):
            ok, msg = authenticate_user(email, password)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["email"] = msg
                # redirect to Dashboard after login
                st.session_state["page"] = "Dashboard"
                st.success(f"✅ Logged in as {msg}")
                st.rerun()
            else:
                st.error(f"❌ Login failed: {msg}")
    with col2:
        if st.button("❓ Forgot Password?"):
            st.session_state["forgot_pw"] = True
            st.rerun()

    if st.session_state.get("forgot_pw"):
        st.subheader("🔐 Reset Password")
        reset_email = st.text_input("Enter your Gmail to reset password")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm Password", type="password")
        if st.button("🔄 Reset Password"):
            if not reset_email.endswith("@gmail.com"):
                st.error("❌ Enter a valid Gmail address.")
            elif not valid_password(new_pw):
                st.error("❌ Password must include upper, lower, digit & special char.")
            elif new_pw != confirm_pw:
                st.error("❌ Passwords do not match.")
            else:
                if update_password(reset_email, new_pw):
                    st.success("✅ Password updated successfully! Please log in.")
                    st.session_state["forgot_pw"] = False
                    st.rerun()
                else:
                    st.error("❌ Failed to reset password.")
    if st.button("⬅ Back to Home"):
        st.session_state["page"] = "Home"
        st.rerun()
# -------------------- DASHBOARD --------------------
elif st.session_state.get("logged_in") and (st.session_state.get("page") == "Dashboard" or st.session_state.get("page") == "Upload"):
    set_bg("images/upload.jpg") if os.path.exists("images/upload.jpg") else None
    st.subheader("📊 Parallel Text Processing Dashboard")
    uploaded_file = st.file_uploader("Choose dataset (CSV or TXT)", type=["csv", "txt"])
    mobile_text_input = st.text_area("Or paste text from mobile (one entry per line).", height=160)
    n_processes = st.slider("Parallel processes (pool size)", min_value=1, max_value=max(1, mp.cpu_count()), value=max(1, mp.cpu_count() - 1))

    # Load texts
    texts = []
    df_input = None
    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
                if "text" in df_input.columns:
                    texts = df_input["text"].fillna("").astype(str).tolist()
                elif "tweet" in df_input.columns:
                    texts = df_input["tweet"].fillna("").astype(str).tolist()
                else:
                    for col in df_input.columns:
                        if df_input[col].dtype == object:
                            texts = df_input[col].fillna("").astype(str).tolist()
                            break
            else:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
                texts = [l.strip() for l in content.splitlines() if l.strip()]
        except Exception as e:
            st.error("Failed to read upload: " + str(e))
    if mobile_text_input:
        mobile_items = [l.strip() for l in mobile_text_input.splitlines() if l.strip()]
        texts.extend(mobile_items)

    st.write(f"Items to process: {len(texts)}")

    if st.button("🚀 Start Processing"):
        if len(texts) == 0:
            st.error("No texts to process.")
        else:
            st.info("Processing started. This runs in parallel using multiprocessing.")

            # -------------------- Classical vs LLM Processing --------------------
            methods_to_run = ["unsupervised", "llm"]
            results = {}
            times = {}
            accuracies = {}

            for m in methods_to_run:
                t0 = time.time()
                df_res, elapsed = parallel_process_texts(texts, method=m, processes=n_processes)
                t1 = time.time()
                total_time = t1 - t0
                times[m] = total_time
                results[m] = df_res

                # Compute accuracy if label exists
                if df_input is not None and "label" in df_input.columns:
                    golds = df_input["label"].astype(str).fillna("").tolist()[:len(df_res)]
                    preds = df_res["sentiment"].astype(str).tolist()[:len(golds)]
                    acc = accuracy_score(golds, preds)
                    accuracies[m] = acc * 100  # percent
                    st.write(f"{m.upper()} - Accuracy: {accuracies[m]:.2f}%, Time: {total_time:.2f}s")
                else:
                    accuracies[m] = None
                    st.write(f"{m.upper()} - Time: {total_time:.2f}s, No label column to compute accuracy.")

            # Save results for chosen method
            chosen_method = "unsupervised"
            df_to_save = results[chosen_method]
            csv_path, zip_path = save_results_csv(df_to_save, base_name=f"to_processing_results_{chosen_method}")
            st.write("Results saved:", csv_path)
            st.write("Compressed:", zip_path)
            with open(zip_path, "rb") as f:
                st.download_button("⬇ Download Results ZIP", data=f, file_name=os.path.basename(zip_path), mime="application/zip")

            # -------------------- Generate Comparison Bar Graph --------------------
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            os.makedirs("outputs", exist_ok=True)

            methods = ["Classical", "LLM"]
            time_values = [times.get("unsupervised", 0), times.get("llm", 0)]
            accuracy_values = [
                accuracies.get("unsupervised", 0) if accuracies.get("unsupervised") else 0,
                accuracies.get("llm", 0) if accuracies.get("llm") else 0,
            ]

            x = np.arange(len(methods))
            width = 0.35
            fig, ax1 = plt.subplots(figsize=(6, 4))
            bar1 = ax1.bar(x - width / 2, time_values, width, label="Time (s)", color="skyblue")
            ax2 = ax1.twinx()
            bar2 = ax2.bar(x + width / 2, accuracy_values, width, label="Accuracy (%)", color="lightgreen")

            ax1.set_xlabel("Methods")
            ax1.set_ylabel("Time (seconds)", color="blue")
            ax2.set_ylabel("Accuracy (%)", color="green")
            ax1.set_title("Classical vs LLM Sentiment Performance")
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods)
            bars = bar1 + bar2
            labels = [b.get_label() for b in bars]
            ax1.legend(bars, labels, loc="upper left")
            plt.tight_layout()
            graph_path = "outputs/performance_bar_graph.png"
            plt.savefig(graph_path)
            st.image(graph_path, caption="Performance Comparison: Classical vs LLM", use_column_width=True)

            # -------------------- Summary Table --------------------
            st.write("### 📊 Summary Results")
            summary_data = {
                "Method": ["Classical (Unsupervised)", "LLM"],
                "Accuracy (%)": [f"{accuracy_values[0]:.2f}", f"{accuracy_values[1]:.2f}"],
                "Time (s)": [f"{time_values[0]:.2f}", f"{time_values[1]:.2f}"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            summary_csv_path = "outputs/summary_results.csv"
            summary_df.to_csv(summary_csv_path, index=False)

            # -------------------- Send Email --------------------
           # -------------------- Send Email --------------------
            receiver = st.session_state.get("email", None)

# Store paths in session to persist across reruns
            st.session_state["email_attachments"] = [csv_path, zip_path, graph_path, summary_csv_path]
            st.session_state["summary_body"] = (
                f"Hello,\n\n"
                f"Your text processing task completed successfully at {datetime.now().isoformat()}.\n"
                f"Processed items: {len(texts)}\n\n"
                f"{'Method':<28}{'Accuracy (%)':<15}{'Time (s)':<10}\n"
                f"{'-'*55}\n"
                f"{'Classical (Unsupervised)':<28}{accuracy_values[0]:<15.2f}{time_values[0]:<10.2f}\n"
                f"{'LLM':<28}{accuracy_values[1]:<15.2f}{time_values[1]:<10.2f}\n\n"
                f"📈 Insights:\n"
                f"- Classical methods are faster but less context-aware.\n"
                f"- LLM methods provide deeper understanding and context accuracy.\n\n"
                f"📎 Attachments:\n"
                f"1️⃣ Processed Results CSV\n"
                f"2️⃣ ZIP Archive of All Results\n"
                f"3️⃣ Performance Graph\n"
                f"4️⃣ Summary Report CSV\n\n"
                f"Best Regards,\nTeam Sentiment Analyzer"
            )

# Display the send button
            if receiver:
                send_now = st.button("📧 Send Results by Email")
                if send_now:
                    try:
                      subject = "📊 Sentiment Analysis: Classical vs LLM Summary Report"
                      attachments = st.session_state["email_attachments"]
                      body = st.session_state["summary_body"]

                      st.info("📨 Sending email... please wait")
                      print("DEBUG: Calling email_file()")

                      email_file(receiver, subject, body, attachments)

                      st.success(f"✅ Email sent successfully to {receiver}!")
                    except Exception as e:
                      st.error(f"❌ Failed to send email: {e}")
                      import traceback
                      st.text(traceback.format_exc())
            else:
                st.info("ℹ️ No user email detected. Please log in or provide an email address.")


# -------------------- MY ACCOUNT --------------------
elif page == "My Account":
    set_bg("images/myaccount.jpg") if os.path.exists("images/myaccount.jpg") else None
    if not st.session_state["logged_in"]:
        st.warning("⚠ Please login first.")
    else:
        email = st.session_state["email"]
        user = get_user(email)
        if user:
            if "show_uploads" not in st.session_state:
                st.session_state["show_uploads"] = False
            if not st.session_state["editing_profile"] and not st.session_state["show_uploads"]:
                st.subheader("👤 My Profile")
                st.write(f"Name: {user['full_name']}")
                st.write(f"Email: {user['email']}")
                st.write(f"DOB: {user['dob']}")
                st.write(f"Address: {user['address']}")
                st.write(f"Pincode: {user['pincode']}")
                st.write(f"Gender: {user['gender']}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("✏ Edit Profile"):
                        st.session_state["editing_profile"] = True
                        st.rerun()
                with c2:
                    if st.button("📸 Previous Uploads"):
                        st.session_state["show_uploads"] = True
                        st.rerun()
                with c3:
                    if st.button("🚪 Logout"):
                        st.session_state["logged_in"] = False
                        st.session_state["email"] = None
                        st.session_state["page"] = "Home"
                        st.success("✅ Logged out successfully!")
                        st.rerun()
                with c4:
                    if st.button("⬅ Go Back to Home"):
                        st.session_state["page"] = "Home"
                        st.rerun()
            elif st.session_state["show_uploads"]:
                st.subheader("🖼 Your Previous Uploads")
                upload_dir = f"user_uploads/{email}"
                if os.path.exists(upload_dir):
                    images = [f for f in os.listdir(upload_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if images:
                        for img_file in images:
                            path = os.path.join(upload_dir, img_file)
                            st.image(path, caption=img_file, use_container_width=True)
                    else:
                        st.info("No uploaded images yet.")
                else:
                    st.info("No uploads found.")
                if st.button("⬅ Back to Profile"):
                    st.session_state["show_uploads"] = False
                    st.rerun()
            elif st.session_state["editing_profile"]:
                st.subheader("✏ Edit Your Profile")
                new_full_name = st.text_input("Full Name", value=user["full_name"])
                new_dob = st.date_input("Date of Birth", value=user["dob"])
                new_address = st.text_area("Address", value=user["address"])
                new_pincode = st.text_input("Pincode", value=user["pincode"], max_chars=6)
                new_gender = st.selectbox("Gender", ["Male","Female","Other"], index=["Male","Female","Other"].index(user["gender"]) if user["gender"] in ["Male","Female","Other"] else 0)
                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("💾 Save Changes"):
                        if len(new_pincode) != 6 or not new_pincode.isdigit():
                            st.error("❌ Pincode must be exactly 6 digits.")
                        else:
                            success = update_user(email, new_full_name, new_dob, new_address, new_pincode, new_gender)
                            if success:
                                st.success("✅ Profile updated successfully!")
                                st.session_state["editing_profile"] = False
                                st.rerun()
                            else:
                                st.error("❌ Failed to update profile.")
                with cancel_col:
                    if st.button("❌ Cancel"):
                        st.session_state["editing_profile"] = False
                        st.rerun()
        else:
            st.error("❌ User not found.")