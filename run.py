"""
Entrypoint: fetches unread emails, runs each through the agent graph, skips
anything already processed. Run manually or on a schedule (cron / Task Scheduler).
"""

import os

from dotenv import load_dotenv

load_dotenv()

import tools
from agent_graph import process_email


def main():
    tools.init_db()

    max_emails = int(os.getenv("MAX_EMAILS_PER_RUN", "10"))
    print(f"Fetching up to {max_emails} unread emails...")
    emails = tools.fetch_unread_emails(max_results=max_emails)

    if not emails:
        print("No unread emails found.")
        return

    new_emails = [e for e in emails if not tools.is_processed(e["id"])]
    print(f"{len(emails)} unread, {len(new_emails)} not yet processed.\n")

    for email in new_emails:
        print(f"Processing: '{email['subject']}' from {email['sender']}")
        result = process_email(email)
        print(f"  -> action taken: {result['final_action']}"
              f" (human overrode: {result['human_overrode']})\n")

    print("Done. Run `python eval.py` to see agent performance metrics.")


if __name__ == "__main__":
    main()
