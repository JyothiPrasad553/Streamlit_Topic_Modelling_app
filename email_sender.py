# =======================================
# 📧 EMAIL SENDER MODULE (email_sender.py)
# =======================================

import os
import smtplib
import ssl
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def email_file(receiver_email, subject, body, attachments=None, summary_df=None):
    """
    Send an email with attachments and optional HTML summary table.

    Required environment variables:
      EMAIL_SENDER  -> Your Gmail address
      EMAIL_APP_PASS -> Your Gmail app password
    """

    sender_email = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_APP_PASS")

    if not sender_email or not password:
        raise ValueError("❌ Missing EMAIL_SENDER or EMAIL_APP_PASS environment variables.")

    # Create base message
    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # Add text body
    plain_body = body

    # Convert summary DataFrame to HTML table (if provided)
    html_summary = ""
    if summary_df is not None and not summary_df.empty:
        html_summary = summary_df.to_html(index=False, border=1, justify="center")

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <p>{body.replace('\n', '<br>')}</p>
        <h3>📊 Summary Results</h3>
        {html_summary}
        <br>
        <p>📎 Attachments included:</p>
        <ul>
            <li>Processed Results CSV</li>
            <li>ZIP Archive of All Results</li>
            <li>Performance Graph</li>
            <li>Summary Report CSV</li>
        </ul>
        <p style="color:gray;font-size:12px;">Sent automatically by Sentiment Analyzer Dashboard.</p>
    </body>
    </html>
    """

    # Attach both text and HTML bodies
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    # Attach files
    if attachments:
        for file_path in attachments:
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
            msg.attach(part)

    # Send email securely via Gmail
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.send_message(msg)

    print(f"✅ Email sent successfully to {receiver_email}")
