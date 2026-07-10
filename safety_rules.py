"""
Rule-based safety net that runs BEFORE we trust the LLM's classification.

Why this exists: LLM confidence scores are not calibrated probabilities. An
email can get 0.85 confidence and still be something you actually needed to
see (security alerts, account verification, real recruiters/companies).
Rules here are cheap, predictable, and catch the cases where being wrong is
expensive -- they don't need to be smart, just conservative.

Add to this list based on what YOU see get misclassified. This file is meant
to be edited by hand as you learn what your inbox looks like.
"""

# Senders/domains that should ALWAYS escalate to human, regardless of LLM confidence.
# Add domains for companies/recruiters/schools you care about.
PROTECTED_SENDER_KEYWORDS = [
    "accounts.google.com",
    "no-reply@google.com",
    "security@",
    "noreply@github.com",
    "@linkedin.com",
    # add internship/company domains you care about, e.g.:
    # "@yourinternshipcompany.com",
]

# Subject/body keywords that should ALWAYS escalate, regardless of LLM confidence.
PROTECTED_CONTENT_KEYWORDS = [
    "verify your",
    "security alert",
    "sign-in attempt",
    "new device",
    "password",
    "2-step verification",
    "verification code",
    "otp",
    "account suspended",
    "unusual activity",
    "offer letter",
    "interview",
    "internship",
    "invoice",
    "contract",
    "deadline",
    "urgent",
]

# Senders that should NEVER get a draft_reply, because nobody reads that inbox.
# This is checked separately from PROTECTED_* above -- it doesn't force human
# approval, it just blocks the "draft_reply" action specifically and swaps it
# for "archive" (or "create_task" if the LLM thought a task was relevant).
NOREPLY_SENDER_PATTERNS = [
    "noreply",
    "no-reply",
    "do-not-reply",
    "donotreply",
    "no.reply",
    "notifications@",
    "notification@",
    "updates@",
    "newsletter",
    "mailer-daemon",
    "postmaster@",
    "automated@",
    "auto-confirm@",
    "bounce@",
    "bounces@",
]

# Subject/body signals that this is a mass/automated/marketing email, not a
# real person who could receive a reply.
SYSTEM_GENERATED_KEYWORDS = [
    "unsubscribe",
    "view this email in your browser",
    "you are receiving this email because",
    "this is an automated message",
    "this is a system-generated",
    "% off",
    "limited time offer",
    "shop now",
]


def is_noreply_or_automated(sender: str, subject: str, body: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if this email should never receive a draft_reply,
    because it's a noreply address, newsletter, or automated/marketing email.
    """
    sender_lower = sender.lower()
    text_lower = f"{subject} {body}".lower()

    for pattern in NOREPLY_SENDER_PATTERNS:
        if pattern in sender_lower:
            return True, f"sender matched noreply pattern: '{pattern}'"

    for keyword in SYSTEM_GENERATED_KEYWORDS:
        if keyword in text_lower:
            return True, f"content matched automated/marketing pattern: '{keyword}'"

    return False, ""


def is_protected(sender: str, subject: str, body: str) -> tuple[bool, str]:
    """
    Returns (True, reason) if this email must be escalated regardless of what
    the LLM says. Checked BEFORE the LLM classification result is trusted.
    """
    sender_lower = sender.lower()
    text_lower = f"{subject} {body}".lower()

    for keyword in PROTECTED_SENDER_KEYWORDS:
        if keyword.lower() in sender_lower:
            return True, f"sender matched protected keyword: '{keyword}'"

    for keyword in PROTECTED_CONTENT_KEYWORDS:
        if keyword.lower() in text_lower:
            return True, f"content matched protected keyword: '{keyword}'"

    return False, ""
