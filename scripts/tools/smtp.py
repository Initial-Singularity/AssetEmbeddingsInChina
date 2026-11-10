#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Send an e-mail notification, with optional attachments, once a batch of jobs has finished.

The runners call it as ``python -m scripts.tools.smtp``; SMTP server, port, account and authorization code come
from the command line or, failing that, from a ``.env`` file.
"""

import os
import smtplib
import argparse
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv


class EmailNotifier:
    """SMTP-over-SSL sender for plain-text mails with file attachments.

    Args:
        smtp_server: SMTP server host.
        smtp_port: SMTP server port.
        username: Sender mailbox account.
        password: The mailbox's SMTP authorization code, not its login password.
    """

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_email(self, to_email: str, subject: str, body: str, attachments=None):
        """Send one mail; a missing attachment is reported and skipped, a delivery error is printed, not raised.

        Args:
            to_email: Recipient address.
            subject: Mail subject.
            body: Plain-text body.
            attachments: Paths of files to attach, or None.
        """
        if attachments is None:
            attachments = []

        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, _charset="utf-8"))

        for file_path in attachments:
            if not os.path.exists(file_path):
                print(f"[warning] attachment not found, skipped: {file_path}")
                continue
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
            part["Content-Disposition"] = f'attachment; filename="{os.path.basename(file_path)}"'
            msg.attach(part)

        try:
            server = smtplib.SMTP_SSL(self.smtp_server)
            server.connect(self.smtp_server, self.smtp_port)
            server.ehlo()
            server.login(self.username, self.password)
            server.send_message(msg)
            print(f"[ok] mail sent to {to_email}")
            server.quit()
        except Exception as e:
            print(f"[error] mail delivery failed: {e}")


def main():
    """Parse the command line, resolve the SMTP settings (flags first, then ``.env``) and send to every recipient."""
    # .env fills os.environ without overriding it; the explicit flags still win below
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Send an e-mail notification after a batch of jobs (attachments supported; SMTP settings from flags or .env)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-t", "--to", required=True, nargs="+", help="recipient addresses, space separated. Required.")

    parser.add_argument("-s", "--subject", default="Batch job finished", help="mail subject")
    parser.add_argument(
        "-b",
        "--body",
        default=f"Batch job finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        help="mail body",
    )

    parser.add_argument("-a", "--attachments", nargs="*", help="attachment paths (several allowed)")

    parser.add_argument("--smtp-server", help="SMTP server host (e.g. smtp.qq.com); falls back to SMTP_SERVER in .env")
    parser.add_argument("--smtp-port", type=int, help="SMTP port (e.g. 465); falls back to SMTP_PORT in .env")
    parser.add_argument("--email-username", help="sender mailbox account; falls back to EMAIL_USERNAME in .env")
    parser.add_argument(
        "--email-password",
        help="mailbox SMTP authorization code (not the login password); falls back to EMAIL_PASSWORD in .env",
    )

    args = parser.parse_args()

    smtp_server = args.smtp_server or os.getenv("SMTP_SERVER")
    smtp_port = args.smtp_port or int(os.getenv("SMTP_PORT", 0))
    username = args.email_username or os.getenv("EMAIL_USERNAME")
    password = args.email_password or os.getenv("EMAIL_PASSWORD")

    missing = []
    if not smtp_server:
        missing.append("SMTP_SERVER (--smtp-server or .env)")
    if not smtp_port:
        missing.append("SMTP_PORT (--smtp-port or .env)")
    if not username:
        missing.append("EMAIL_USERNAME (--email-username or .env)")
    if not password:
        missing.append("EMAIL_PASSWORD (--email-password or .env)")

    if missing:
        print("[error] missing settings; pass them on the command line or in .env:")
        for item in missing:
            print(f"  - {item}")
        return

    notifier = EmailNotifier(smtp_server, smtp_port, username, password)
    for recipient in args.to:
        notifier.send_email(recipient, args.subject, args.body, args.attachments)


if __name__ == "__main__":
    main()
