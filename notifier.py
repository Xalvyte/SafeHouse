# notifier.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

from database.db import db
from database.models import User

# --- CONFIG ---
SENDER_EMAIL = "thesis.raspberrypi@gmail.com"
APP_PASSWORD = "ckzu fwhu aviv jsqc"  # Gmail App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465


def send_email(subject: str, body: str, recipients: list[str]):
    """
    Send an email to a list of recipients.
    """
    if not recipients:
        print("⚠️ No recipients specified. Skipping email.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

        print(f"✅ Email sent to: {', '.join(recipients)}")

    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def send_to_all(subject: str, body: str, app=None):
    """
    Query all users with an email and send them the message.
    Requires Flask app context.
    """
    if app is None:
        raise RuntimeError("❌ Flask app must be passed when calling send_to_all inside app.py")

    try:
        with app.app_context():
            users = User.query.filter(User.email.isnot(None)).all()
            recipients = [u.email for u in users]

            if not recipients:
                print("⚠️ No users with email found in DB.")
                return

            send_email(subject, body, recipients)

    except Exception as e:
        print(f"❌ Failed to fetch users from DB: {e}")


# --- TEST MODE ---
if __name__ == "__main__":
    print("🚀 Running notifier test...")

    # ✅ Import app only here to avoid circular import
    from app import app

    try:
        send_to_all("Safehouse Test", "This is a test email from Safehouse Notifier.", app)
    except Exception as e:
        print(f"❌ Test failed: {e}")
