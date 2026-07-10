"""
Tools the agent can call:
  - Gmail: fetch unread emails, create drafts, archive/label messages
  - SQLite: store tasks, store decision logs, track processed email IDs

Kept dependency-free of any agent framework so it's easy to test in isolation:
    python -c "from tools import fetch_unread_emails; print(fetch_unread_emails(3))"
"""

import base64
import os
import sqlite3
from datetime import datetime
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]

DB_PATH = os.path.join(os.path.dirname(__file__), "agent.db")


# ---------------------------------------------------------------------------
# Gmail auth
# ---------------------------------------------------------------------------

def _get_gmail_service():
    creds = None
    token_path = os.path.join(os.path.dirname(__file__), "token.json")
    creds_path = os.path.join(os.path.dirname(__file__), "credentials.json")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    "credentials.json not found. Follow the Gmail API setup steps "
                    "in README.md before running the agent."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Gmail actions
# ---------------------------------------------------------------------------

def fetch_unread_emails(max_results: int = 10) -> list[dict]:
    """Returns unread emails as [{id, sender, subject, body}, ...]."""
    service = _get_gmail_service()
    results = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["UNREAD", "INBOX"], maxResults=max_results)
        .execute()
    )
    messages = results.get("messages", [])

    emails = []
    for m in messages:
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        body = _extract_body(msg["payload"])
        emails.append(
            {
                "id": m["id"],
                "sender": headers.get("From", ""),
                "subject": headers.get("Subject", "(no subject)"),
                "body": body,
            }
        )
    return emails


def _extract_body(payload: dict) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        # fallback: recurse into nested parts
        for part in payload["parts"]:
            result = _extract_body(part)
            if result:
                return result
        return ""
    data = payload.get("body", {}).get("data", "")
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore") if data else ""


def create_draft_reply(to_email: str, subject: str, body_text: str, thread_id: str) -> str:
    """Creates a Gmail draft. Never sends automatically."""
    service = _get_gmail_service()
    message = MIMEText(body_text)
    message["to"] = to_email
    message["subject"] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    draft = (
        service.users()
        .drafts()
        .create(userId="me", body={"message": {"raw": raw, "threadId": thread_id}})
        .execute()
    )
    return draft["id"]


def archive_email(email_id: str) -> None:
    service = _get_gmail_service()
    service.users().messages().modify(
        userId="me", id=email_id, body={"removeLabelIds": ["INBOX", "UNREAD"]}
    ).execute()


def mark_read(email_id: str) -> None:
    service = _get_gmail_service()
    service.users().messages().modify(
        userId="me", id=email_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT,
            title TEXT,
            created_at TEXT,
            done INTEGER DEFAULT 0
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT,
            sender TEXT,
            subject TEXT,
            intent TEXT,
            confidence REAL,
            reasoning TEXT,
            suggested_action TEXT,
            final_action TEXT,
            human_overrode INTEGER DEFAULT 0,
            created_at TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS processed_ids (
            email_id TEXT PRIMARY KEY
        )"""
    )
    conn.commit()
    conn.close()


def is_processed(email_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT 1 FROM processed_ids WHERE email_id = ?", (email_id,)).fetchone()
    conn.close()
    return row is not None


def mark_processed(email_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR IGNORE INTO processed_ids (email_id) VALUES (?)", (email_id,))
    conn.commit()
    conn.close()


def create_task(email_id: str, title: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO tasks (email_id, title, created_at) VALUES (?, ?, ?)",
        (email_id, title, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def log_decision(
    email_id: str,
    sender: str,
    subject: str,
    intent: str,
    confidence: float,
    reasoning: str,
    suggested_action: str,
    final_action: str,
    human_overrode: bool,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO logs
        (email_id, sender, subject, intent, confidence, reasoning,
         suggested_action, final_action, human_overrode, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            email_id,
            sender,
            subject,
            intent,
            confidence,
            reasoning,
            suggested_action,
            final_action,
            int(human_overrode),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
