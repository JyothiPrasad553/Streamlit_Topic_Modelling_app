# test_email.py
import os
import smtplib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# -------------------- LOAD ENVIRONMENT VARIABLES --------------------
load_dotenv()
EMAIL_SENDER = os.getenv("EMAIL_SENDER")  # e.g., your_email@gmail.com
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", EMAIL_SENDER)

# -------------------- MOCK DATA (for testing) --------------------
# You can replace this block with your actual data pipeline
texts = ["I love AI", "This is bad", "Not sure about this"]
times = {"unsupervised": 1.8, "llm": 4.6}
accuracies = {"unsupervised": 72.5, "llm": 89.3}

# -------------------- PERFORMANCE GRAPH --------------------
os.makedirs("outputs", exist_ok=True)
methods = ["Classical", "LLM"]
time_values = [times.get("unsupervised", 0), times.get("llm", 0)]
accuracy_values = [accuracies.get("unsupervised", 0), accuracies.get("llm", 0)]

x = np.arange(len(methods))
width = 0.35
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.bar(x - width/2, time_values, width, label="Time (s)", color="skyblue")
ax2 = ax1.twinx()
ax2.bar(x + width/2, accuracy_values, width, label="Accuracy (%)", color="lightgreen")

ax1.set_xlabel("Methods")
ax1.set_ylabel("Time (s)", color="blue")
ax2.set_ylabel("Accuracy (%)", color="green")
ax1.set_title("Classical vs LLM Sentiment Performance")
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
plt.tight_layout()

graph_path = "outputs/performance_graph.png"
plt.savefig(graph_path)
plt.close()

# -------------------- SUMMARY TABLE --------------------
summary_df = pd.DataFrame({
    "Method": ["Classical (Unsupervised)", "LLM"],
    "Accuracy (%)": [f"{accuracy_values[0]:.2f}", f"{accuracy_values[1]:.2f}"],
    "Time (s)": [f"{time_values[0]:.2f}", f"{time_values[1]:.2f}"]
})
summary_csv_path = "outputs/summary_results.csv"
summary_df.to_csv(summary_csv_path, index=False)

# -------------------- EMAIL CONTENT --------------------
subject = "📊 Sentiment Analysis: Classical vs LLM Summary Report"
body = (
    f"Hello,\n\n"
    f"Your text processing task completed successfully.\n"
    f"Processed items: {len(texts)}\n\n"
    f"{'Method':<28}{'Accuracy (%)':<15}{'Time (s)':<10}\n"
    f"{'-'*55}\n"
    f"{'Classical (Unsupervised)':<28}{accuracy_values[0]:<15.2f}{time_values[0]:<10.2f}\n"
    f"{'LLM':<28}{accuracy_values[1]:<15.2f}{time_values[1]:<10.2f}\n\n"
    f"📈 Insights:\n"
    f"- Classical models are faster but may be less context-aware.\n"
    f"- LLMs provide richer understanding of sentiment context.\n\n"
    f"📎 Attachments include:\n"
    f"1️⃣ Performance Graph\n"
    f"2️⃣ Summary Report CSV\n\n"
    f"Best Regards,\nTeam Sentiment Analyzer"
)

attachments = [graph_path, summary_csv_path]

# -------------------- EMAIL SENDING FUNCTION --------------------
def send_email(sender, password, receiver, subject, body, attachments):
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach files
    for file in attachments:
        with open(file, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(file)}"'
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email sent successfully to {receiver}!")
    except Exception as e:
        print("❌ Email failed:", e)

# -------------------- RUN EMAIL TEST --------------------
if __name__ == "__main__":
    send_email(EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_RECEIVER, subject, body, attachments)
