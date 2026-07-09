#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量任务完成后发送邮件通知脚本

功能特性：
1. 面向对象设计（EmailNotifier 类）
2. argparse 支持命令行参数（收件人、主题、正文、附件）
3. 支持发送附件（单个或多个）
4. 邮箱账号、密码等敏感信息从 .env 文件加载
5. 提供详细注释与命令行帮助文档
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
    """邮件通知类，支持纯文本和附件发送"""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        """
        初始化邮件通知器
        :param smtp_server: SMTP服务器地址
        :param smtp_port: SMTP服务器端口
        :param username: 邮箱用户名
        :param password: 邮箱授权码（非登录密码）
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_email(self, to_email: str, subject: str, body: str, attachments=None):
        """
        发送邮件
        :param to_email: 收件人邮箱
        :param subject: 邮件主题
        :param body: 邮件正文
        :param attachments: 附件路径列表
        """
        if attachments is None:
            attachments = []

        # 创建邮件对象（支持附件）
        msg = MIMEMultipart()
        msg["From"] = self.username
        msg["To"] = to_email
        msg["Subject"] = subject

        # 添加正文
        msg.attach(MIMEText(body, _charset="utf-8"))

        # 添加附件
        for file_path in attachments:
            if not os.path.exists(file_path):
                print(f"[警告] 附件不存在：{file_path}，已跳过。")
                continue
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
            part["Content-Disposition"] = f'attachment; filename="{os.path.basename(file_path)}"'
            msg.attach(part)

        # 发送邮件
        try:
            server = smtplib.SMTP_SSL(self.smtp_server)
            server.connect(self.smtp_server, self.smtp_port)
            server.ehlo()
            server.login(self.username, self.password)
            server.send_message(msg)
            print(f"[成功] 邮件已发送至 {to_email}")
            server.quit()
        except Exception as e:
            print(f"[错误] 邮件发送失败: {e}")


def main():
    """命令行入口"""
    # 加载 .env 文件（但仍允许命令行覆盖）
    load_dotenv()

    # 构建参数解析器
    parser = argparse.ArgumentParser(
        description="批量任务完成后发送邮件通知（支持附件，SMTP配置可从命令行或.env读取）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必填：收件人
    parser.add_argument("-t", "--to", required=True, nargs="+", help="收件人邮箱地址, 可多个，用空格分隔. Required.")

    # 可选：主题与正文
    parser.add_argument("-s", "--subject", default="批量任务完成通知", help="邮件主题")
    parser.add_argument(
        "-b", "--body", default=f"批量任务已完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", help="邮件正文"
    )

    # 附件（可选）
    parser.add_argument("-a", "--attachments", nargs="*", help="附件路径（可多个）")

    # SMTP 配置参数（优先命令行，其次 .env）
    parser.add_argument(
        "--smtp-server", help="SMTP服务器地址（如：smtp.qq.com），若不提供将从.env读取`SMTP_SERVER`字段"
    )
    parser.add_argument("--smtp-port", type=int, help="SMTP端口（如：465），若不提供将从.env读取`SMTP_PORT`字段")
    parser.add_argument("--email-username", help="发件人邮箱账号，若不提供将从.env读取`EMAIL_USERNAME`字段")
    parser.add_argument("--email-password", help="邮箱授权码（非登录密码），若不提供将从.env读取`EMAIL_PASSWORD`字段")

    args = parser.parse_args()

    # 获取 SMTP 配置：优先使用命令行参数，否则从 .env 读取
    smtp_server = args.smtp_server or os.getenv("SMTP_SERVER")
    smtp_port = args.smtp_port or int(os.getenv("SMTP_PORT", 0))
    username = args.email_username or os.getenv("EMAIL_USERNAME")
    password = args.email_password or os.getenv("EMAIL_PASSWORD")

    # 验证必要配置
    missing = []
    if not smtp_server:
        missing.append("SMTP_SERVER (--smtp-server 或 .env)")
    if not smtp_port:
        missing.append("SMTP_PORT (--smtp-port 或 .env)")
    if not username:
        missing.append("EMAIL_USERNAME (--email-username 或 .env)")
    if not password:
        missing.append("EMAIL_PASSWORD (--email-password 或 .env)")

    if missing:
        print("[错误] 以下配置缺失，请通过命令行参数或 .env 文件提供：")
        for item in missing:
            print(f"  - {item}")
        return

    # 发送邮件
    notifier = EmailNotifier(smtp_server, smtp_port, username, password)
    for recipient in args.to:
        notifier.send_email(recipient, args.subject, args.body, args.attachments)


if __name__ == "__main__":
    main()
