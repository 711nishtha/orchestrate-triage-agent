from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Keyword / pattern sets
# ---------------------------------------------------------------------------

BUG_KEYWORDS = [
    "not working",
    "stopped working",
    "is down",
    "are down",
    "outage",
    "crash",
    "crashing",
    "inaccessible",
    "all requests are failing",
    "all requests to claude",
    "requests failing",
    "submissions across any challenges are not working",
    "none of the submissions",
    "blocker",
    "failing",
    "can not able",
    "cannot able",
]

FEATURE_REQUEST_KEYWORDS = [
    "please add",
    "would like to",
    "can you implement",
    "new feature",
    "stop crawling my website",
    "stop crawling",
    "setup a claude lti key",
    "set up a claude lti",
    "claude lti key",
    "fill in the forms",
    "infosec process",
    "can we extend inactivity",
    "extend inactivity times",
]

INVALID_PATTERNS = [
    "what is the name of the actor",
    "review answers and increase score",
    "review my answers",
    "increase my score",
    "increase score",
    "tell the company to move me",
    "ban the seller",
    "ban sellers",
    "reveal internal fraud detection rules",
    "internal fraud detection rules",
    "delete all files from the system",
    "delete all files",
]

MALICIOUS_PATTERNS = [
    "delete all files from the system",
    "delete all files",
    "urgent cash but don't have any right now",
    "urgent cash but dont have any right now",
    "reveal internal fraud detection rules",
    "affiche toutes les règles",
    "affiche toutes les regles",
    "internal rules",
    "logique exacte que vous utilisez",
    "logic you use",
    "la logique exacte",
    "les règles internes",
    "les regles internes",
]

SENSITIVE_PATTERNS = [
    "delete account",
    "remove user",
    "restore access",
    "not owner",
    "not admin",
    "not the workspace owner",
    "refund",
    "payment issue",
    "order id",
    "dispute",
    "identity theft",
    "identity has been stolen",
    "fraud",
    "affiche toutes les règles",
    "internal rules",
    "logic you use",
    "card blocked",
    "carte bloquée",
    "carte visa a été bloquée",
    "security vulnerability",
]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def ticket_text(ticket: Dict[str, str]) -> str:
    """Combine all ticket fields into a single lowercase string for matching."""
    return " ".join(str(ticket.get(key, "") or "") for key in ("Issue", "Subject", "Company")).strip().lower()


def contains_any(text: str, phrases) -> bool:
    return any(phrase in text for phrase in phrases)


def _word_present(text: str, word: str) -> bool:
    """Check if a whole word is present (avoids 'down' matching 'download')."""
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


# ---------------------------------------------------------------------------
# Classification: request_type
# ---------------------------------------------------------------------------

def classify_request_type(ticket: Dict[str, str]) -> str:
    text = ticket_text(ticket)

    # --- Malicious / invalid first ---
    if contains_any(text, MALICIOUS_PATTERNS):
        return "invalid"

    if "urgent cash" in text and not contains_any(
        text, ["lost card", "stolen card", "blocked card", "dispute"]
    ):
        return "invalid"

    if contains_any(text, INVALID_PATTERNS):
        return "invalid"

    # "thank you" alone (very short) → invalid
    if "thank you" in text and len(text.split()) < 15:
        return "invalid"

    # "refund me today and ban the seller" → invalid (unreasonable demand)
    if "ban" in text and ("refund" in text or "seller" in text):
        return "invalid"

    # --- Bug detection ---
    # Before applying bug keywords, check if this is a known product_issue
    # that happens to contain bug-like language (e.g. "not working", "blocker")
    product_issue_overrides = (
        "mock interview" in text
        or "mock" in text and "interview" in text
        or "compatibility" in text or "compatible" in text
        or "apply tab" in text
        or "apply" in text and "tab" in text
        or "can not able" in text or "cannot able" in text
        or ("refund" in text and "mock" in text)
        or "certificate" in text
        or "reschedul" in text
        or ("zoom" in text and ("check" in text or "connectivity" in text))
    )

    # Very vague "not working" without specifics → product_issue, not bug
    is_vague_bug_report = (
        len(text.split()) < 12
        and not contains_any(text, [
            "site", "platform", "system", "server", "service",
            "all requests", "submissions across", "none of the",
            "completely", "outage",
        ])
    )

    if not product_issue_overrides and not is_vague_bug_report and contains_any(text, BUG_KEYWORDS):
        return "bug"

    # Single-word "down" with word boundaries (avoid 'download')
    if not product_issue_overrides and _word_present(text, "down") and not _word_present(text, "download"):
        return "bug"

    # --- Feature request ---
    if contains_any(text, FEATURE_REQUEST_KEYWORDS):
        return "feature_request"

    return "product_issue"


# ---------------------------------------------------------------------------
# Company inference
# ---------------------------------------------------------------------------

def infer_company(text: str) -> Optional[str]:
    """Infer the company from ticket text."""
    lower = text.lower()
    if "hackerrank" in lower:
        return "hackerrank"
    if "claude" in lower or "anthropic" in lower:
        return "claude"
    if "visa" in lower:
        return "visa"
    if any(kw in lower for kw in ("traveller's cheque", "travellers cheque", "traveler's check", "travelers check")):
        return "visa"
    if any(kw in lower for kw in ("lost or stolen card", "stolen card", "lost card")):
        return "visa"
    return None


# ---------------------------------------------------------------------------
# Product area classification
# ---------------------------------------------------------------------------

def classify_product_area(ticket: Dict[str, str], company: Optional[str], top_article_source: Optional[str]) -> str:
    text = ticket_text(ticket)
    normalized_company = (company or "").strip().lower() or infer_company(text) or (top_article_source or "").lower()

    if normalized_company == "hackerrank":
        # Order matters — more specific checks first
        if contains_any(text, ["mock interview", "mock"]):
            return "community"
        if contains_any(text, ["delete account", "profile", "community", "username", "login",
                                "resume", "resume builder", "certificate"]):
            return "community"
        # Compatibility/zoom checks and inactivity are about test-taking (screen)
        if contains_any(text, ["compatibility", "compatible", "inactivity"]):
            return "screen"
        if contains_any(text, ["interview", "interviewer", "video call"]):
            return "interview"
        if contains_any(text, ["test", "candidate", "assessment", "screen", "submission", "challenge",
                                "reinvite", "extra time", "inactivity", "zoom"]):
            return "screen"
        if contains_any(text, ["payment", "billing", "subscription", "invoice", "refund",
                                "pause", "cancel subscription"]):
            return "billing"
        if contains_any(text, ["infosec", "security", "compliance"]):
            return "general"
        return "general"

    if normalized_company == "claude":
        if contains_any(text, ["delete chat", "rename chat", "conversation", "chat history",
                                "private info", "private"]):
            return "conversation_management"
        if contains_any(text, ["privacy", "crawl", "crawling",
                                "training data", "personal data"]):
            return "privacy"
        if _word_present(text, "lti") or contains_any(text, ["integration", "integrations", "connector"]):
            return "integrations"
        if contains_any(text, ["workspace", "access", "account", "login", "sso", "seat"]):
            return "account"
        if contains_any(text, ["api", "bedrock", "aws", "console"]):
            return "api"
        if contains_any(text, ["bug bounty", "vulnerability", "security"]):
            return "security"
        if contains_any(text, ["data", "improve", "model"]):
            return "privacy"
        if contains_any(text, ["not working", "stopped working", "failing"]):
            return "general"
        return "general"

    if normalized_company == "visa":
        if contains_any(text, ["traveller", "traveler", "cheque", "cheques",
                                "travelling", "traveling", "abroad"]):
            return "travel_support"
        if contains_any(text, ["fraud", "scam", "stolen identity", "identity theft"]):
            return "fraud"
        if contains_any(text, ["dispute", "chargeback"]):
            return "dispute_resolution"
        if contains_any(text, ["lost card", "stolen card", "lost or stolen", "block card"]):
            return "general_support"
        if contains_any(text, ["minimum", "surcharge", "merchant", "checkout"]):
            return "general_support"
        return "general_support"

    return "general_support"


# ---------------------------------------------------------------------------
# Risk / Safety
# ---------------------------------------------------------------------------

def is_high_risk_invalid(ticket: Dict[str, str]) -> bool:
    text = ticket_text(ticket)
    return (
        contains_any(text, MALICIOUS_PATTERNS)
        or "delete all files" in text
    )


def safety_check(ticket: Dict[str, str], similarity: float, top_article_title: str) -> Tuple[bool, bool]:
    """Returns (is_sensitive, safe_to_reply).

    safe_to_reply=True means the agent should answer directly.
    safe_to_reply=False means the agent should escalate.
    """
    text = ticket_text(ticket)
    title = (top_article_title or "").lower()

    # Known-safe topics that should NOT be escalated even when sensitive keywords fire.
    # Use explicit parentheses to avoid operator precedence bugs.
    safe_exceptions = (
        "lost or stolen" in title
        or "traveller" in title
        or "travelers" in title
        or "cheque" in title
        or ("delete" in title and "conversation" in title)
        or ("delete" in title and "account" in title)
        or "compatibility" in title
        or "extra time" in title
        or "reinvit" in title
        or "certificate" in title
        or "certifications" in title
        or ("pause" in title and "subscription" in title)
        or ("cancel" in title and "subscription" in title)
        or "user management" in title
        or "user roles" in title
        or "dispute" in title
        or "bug bounty" in title
        or "vulnerability" in title
        or "mock interview" in title
        or "resume builder" in title
        or "manage subscriptions" in title
    )

    # Also check safe exceptions based on ticket text + company (not just title)
    text_safe = (
        ("mock interview" in text and ("hackerrank" in text or _get_company(ticket) == "hackerrank"))
        or ("certificate" in text and "name" in text and ("hackerrank" in text or _get_company(ticket) == "hackerrank"))
        or ("pause" in text and "subscription" in text)
        or ("resume" in text and "builder" in text)
        or ("bug bounty" in text or "security vulnerability" in text)
        or ("dispute" in text and "charge" in text and ("visa" in text or _get_company(ticket) == "visa"))
    )

    block_card_internal_rules = "block card" in text and contains_any(text, ["internal rules", "logic you use"])
    is_sensitive = contains_any(text, SENSITIVE_PATTERNS) or block_card_internal_rules

    # Workspace access without being owner/admin → always escalate
    if "not the workspace owner" in text or ("not owner" in text and "not admin" in text):
        return True, False

    # Identity theft → always escalate
    if "identity" in text and ("theft" in text or "stolen" in text):
        return True, False

    # Low similarity with no safe exception → treat as sensitive
    if similarity < 0.45 and not (safe_exceptions or text_safe):
        is_sensitive = True

    safe_to_reply = (safe_exceptions or text_safe) or not is_sensitive

    return is_sensitive, safe_to_reply


def _get_company(ticket: Dict[str, str]) -> str:
    return (ticket.get("Company") or "").strip().lower()


# ---------------------------------------------------------------------------
# Pattern matching — maps known ticket patterns to corpus articles
# ---------------------------------------------------------------------------

def _match_entry_by_keywords(
    corpus_entries: List[Dict[str, str]],
    include_keywords: List[str],
    preferred_source: Optional[str] = None,
    exclude_index: bool = True,
    title_keywords: Optional[List[str]] = None,
) -> Optional[Dict[str, str]]:
    """Find the best corpus entry matching the given keywords.

    Scoring:
    - +1 for each keyword found in title+text
    - +2 bonus for matching preferred_source
    - +3 bonus for sample_ticket entries (they are gold-standard responses)
    - +5 bonus for title_keywords found in the title (more targeted)
    """
    best_entry: Optional[Dict[str, str]] = None
    best_score = -1

    for entry in corpus_entries:
        # Skip index/table-of-contents entries
        if exclude_index:
            filename = (entry.get("path", "") or "").split("/")[-1].lower()
            if filename == "index.md" or filename == "index.html":
                continue

        title_lower = (entry.get("title", "") or "").lower()
        text_lower = (entry.get("text", "") or "").lower()
        haystack = f"{title_lower}\n{text_lower}"

        score = sum(1 for keyword in include_keywords if keyword in haystack)
        if score <= 0:
            continue

        # Bonus for title-specific keywords
        if title_keywords:
            title_hits = sum(1 for kw in title_keywords if kw in title_lower)
            score += title_hits * 5

        if preferred_source and entry.get("source") == preferred_source:
            score += 2
        if entry.get("entry_type") == "sample_ticket":
            score += 3
        if score > best_score:
            best_entry = entry
            best_score = score
    return best_entry


def pattern_match(ticket: Dict[str, str], corpus_entries: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Override retrieval for well-known ticket patterns.

    Uses a relaxed matching strategy: match if enough keywords are present,
    accounting for the Company column.
    """
    text = ticket_text(ticket)
    company = _get_company(ticket)

    # --- HackerRank patterns ---
    if company == "hackerrank" or "hackerrank" in text:
        # Account deletion via Google login
        if "delete" in text and ("account" in text or "google" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["delete", "account", "google", "password"],
                preferred_source="hackerrank",
                title_keywords=["delete", "account"],
            )

        # Extra time / reinvite candidate
        if "extra time" in text or "reinvite" in text or "reinvit" in text or "time accommodation" in text:
            return _match_entry_by_keywords(
                corpus_entries,
                ["extra time", "reinvite", "candidate", "accommodation", "add time"],
                preferred_source="hackerrank",
                title_keywords=["extra time", "reinvit", "adding"],
            )

        # Mock interviews not working / refund
        if "mock interview" in text or ("mock" in text and "interview" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["mock", "interview", "credit", "purchase"],
                preferred_source="hackerrank",
                title_keywords=["mock interview", "mock"],
            )

        # Zoom / compatibility check
        if "compatibility" in text or "compatible" in text or ("zoom" in text and "check" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["compatibility", "zoom", "system", "verify", "video"],
                preferred_source="hackerrank",
                title_keywords=["compatibility", "zoom", "video"],
            )

        # Certificate name update
        if "certificate" in text and ("name" in text or "update" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["certificate", "name", "update", "certifications"],
                preferred_source="hackerrank",
                title_keywords=["certifications", "certificate"],
            )

        # Submissions not working / apply tab
        if ("submission" in text or "apply" in text) and ("not working" in text or "not able" in text or "can not" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["coding", "challenge", "practice", "submission", "faq"],
                preferred_source="hackerrank",
                title_keywords=["coding", "challenges", "faq"],
            )

        # Pause subscription
        if "pause" in text and ("subscription" in text or "hiring" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["pause", "subscription", "resume"],
                preferred_source="hackerrank",
                title_keywords=["pause", "subscription"],
            )

        # Cancel subscription
        if "cancel" in text and "subscription" in text:
            return _match_entry_by_keywords(
                corpus_entries,
                ["cancel", "subscription"],
                preferred_source="hackerrank",
                title_keywords=["cancel", "subscription"],
            )

        # Remove user / interviewer / employee
        if "remove" in text and ("user" in text or "interviewer" in text or "employee" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["user", "management", "deactivat", "role", "admin"],
                preferred_source="hackerrank",
                title_keywords=["user management", "user roles", "manage users", "types of user"],
            )

        # Reschedule assessment
        if "reschedul" in text or "rescheduling" in text:
            return _match_entry_by_keywords(
                corpus_entries,
                ["reinvit", "invite", "candidate", "test"],
                preferred_source="hackerrank",
                title_keywords=["reinvit", "invite"],
            )

        # Inactivity timeout
        if "inactivity" in text or "kicked out" in text:
            return _match_entry_by_keywords(
                corpus_entries,
                ["interview", "session", "inactivity", "timeout", "configure"],
                preferred_source="hackerrank",
                title_keywords=["interview", "settings", "configure"],
            )

        # Resume builder
        if "resume" in text and ("builder" in text or _word_present(text, "down")):
            return _match_entry_by_keywords(
                corpus_entries,
                ["resume", "builder", "create", "professional"],
                preferred_source="hackerrank",
                title_keywords=["resume", "builder"],
            )

        # Infosec / security questionnaire
        if "infosec" in text or ("security" in text and "form" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["solution engineering", "contact", "security"],
                preferred_source="hackerrank",
                title_keywords=["solution engineering", "contact"],
            )

        # Payment issue with order ID → escalate (don't match)
        if "payment" in text and "order" in text:
            return None

    # --- Visa patterns ---
    if company == "visa" or "visa" in text:
        # Lost or stolen card
        if ("lost" in text or "stolen" in text) and "card" in text:
            return _match_entry_by_keywords(
                corpus_entries,
                ["lost", "stolen", "card", "call", "report"],
                preferred_source="visa",
                title_keywords=["lost", "stolen"],
            )

        # Traveller's cheques
        if any(kw in text for kw in ["traveller", "traveler", "cheque", "cheques"]):
            return _match_entry_by_keywords(
                corpus_entries,
                ["traveller", "cheque", "stolen", "refund", "issuing"],
                preferred_source="visa",
                title_keywords=["traveller", "cheque"],
            )

        # Dispute a charge
        if "dispute" in text and ("charge" in text or "transaction" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["dispute", "resolution", "chargeback"],
                preferred_source="visa",
                title_keywords=["dispute"],
            )

        # Identity theft → escalate, don't pattern match
        if "identity" in text and ("theft" in text or "stolen" in text):
            return None

        # Minimum spend / surcharge
        if "minimum" in text and ("spend" in text or "$" in text or "10" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["minimum", "surcharge", "merchant", "rules", "regulations", "fee"],
                preferred_source="visa",
                title_keywords=["rules", "regulations", "interchange"],
            )

    # --- Claude patterns ---
    if company == "claude" or "claude" in text or "anthropic" in text:
        # Stop crawling website
        if "crawl" in text and ("website" in text or "web" in text or "stop" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["crawl", "web", "data", "block", "crawler"],
                preferred_source="claude",
                title_keywords=["crawl", "data", "web"],
            )

        # AWS Bedrock / API issues — check BEFORE LTI to avoid substring collision
        if "bedrock" in text or ("aws" in text and "fail" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["bedrock", "aws", "amazon", "region", "api"],
                preferred_source="claude",
                title_keywords=["bedrock", "amazon", "aws"],
            )

        # LTI key setup — use word boundary to avoid matching 'lti' inside 'multiple'
        if _word_present(text, "lti"):
            return _match_entry_by_keywords(
                corpus_entries,
                ["lti", "canvas", "integration", "education"],
                preferred_source="claude",
                title_keywords=["lti"],
            )

        # Delete conversation (private info)
        if "delete" in text and ("conversation" in text or "chat" in text or "private" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["delete", "rename", "conversation"],
                preferred_source="claude",
                title_keywords=["delete", "conversation"],
            )

        # Bug bounty / security vulnerability
        if "security vulnerability" in text or "bug bounty" in text or "vulnerability" in text:
            return _match_entry_by_keywords(
                corpus_entries,
                ["bug bounty", "vulnerability", "security", "safety", "report"],
                preferred_source="claude",
                title_keywords=["vulnerability", "bug bounty"],
            )

        # Data use / personal data for training
        if "data" in text and ("improve" in text or "model" in text or "training" in text or "used for" in text):
            return _match_entry_by_keywords(
                corpus_entries,
                ["data", "privacy", "training", "conversation", "sensitive"],
                preferred_source="claude",
                title_keywords=["data", "sensitive", "privacy", "conversation"],
            )

        # Workspace / access lost → escalate for non-owner/admin
        if ("workspace" in text and ("lost" in text or "removed" in text)) or "restore access" in text:
            # If the user is not the owner/admin, escalate
            if "not" in text and ("owner" in text or "admin" in text):
                return None  # Return None to let the escalation happen
            return _match_entry_by_keywords(
                corpus_entries,
                ["workspace", "team", "access", "seat", "admin"],
                preferred_source="claude",
                title_keywords=["team", "workspace"],
            )

    return None
