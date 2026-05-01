from __future__ import annotations

from typing import Dict, List, Optional

from classifier import (
    classify_product_area,
    classify_request_type,
    contains_any,
    is_high_risk_invalid,
    pattern_match,
    safety_check,
    ticket_text,
)


LOW_CONFIDENCE_MESSAGE = "Your request has been escalated to our support team for review. We will get back to you shortly."
OUT_OF_SCOPE_MESSAGE = "I am sorry, this is out of scope from my capabilities."
HIGH_RISK_MESSAGE = "This request has been escalated for review."
THANK_YOU_RESPONSE = "Happy to help!"


def format_article_response(article: Dict[str, str], max_chars: int = 2000) -> str:
    """Format a corpus article into a user-facing response.

    - For sample tickets, return the gold-standard response verbatim.
    - For corpus articles, return the body text (up to max_chars, trimmed at
      a sentence boundary) with a reference URL.
    """
    if article.get("entry_type") == "sample_ticket":
        return article.get("text", "").strip()

    body = (article.get("text", "") or "").strip()
    if len(body) > max_chars:
        # Trim at the last sentence-ending punctuation before the limit
        trimmed = body[:max_chars]
        for sep in ("\n\n", ".\n", ". ", "\n"):
            last_sep = trimmed.rfind(sep)
            if last_sep > max_chars // 3:
                trimmed = trimmed[: last_sep + len(sep)].strip()
                break
        body = trimmed

    url = article.get("url", "") or article.get("path", "")
    if url:
        return f"{body}\n\nReference: {url}"
    return body


def _select_best_retrieval(
    retrieval_results: List[Dict[str, object]],
    ticket: Dict[str, str],
) -> Dict[str, object]:
    """Re-rank retrieval results to prefer articles from the right source.

    If the ticket's Company field matches a source, boost those results.
    """
    if not retrieval_results:
        return {"index": -1, "score": 0.0, "article": {}}

    company = (ticket.get("Company") or "").strip().lower()
    source_map = {
        "hackerrank": "hackerrank",
        "claude": "claude",
        "visa": "visa",
    }
    preferred = source_map.get(company)

    if not preferred:
        return retrieval_results[0]

    # Re-rank: boost score for articles from the preferred source
    best = retrieval_results[0]
    best_adjusted_score = float(best["score"])
    if best["article"].get("source") == preferred:
        best_adjusted_score += 0.05

    for result in retrieval_results[1:]:
        adjusted = float(result["score"])
        if result["article"].get("source") == preferred:
            adjusted += 0.05
        if adjusted > best_adjusted_score:
            best = result
            best_adjusted_score = adjusted

    return best


def decide_ticket(ticket: Dict[str, str], retriever, corpus_entries: List[Dict[str, str]]) -> Dict[str, str]:
    """Main decision logic for a single support ticket.

    Pipeline:
    1. Classify request_type (invalid / bug / feature_request / product_issue)
    2. Retrieve top-k corpus articles
    3. Re-rank retrieval results by company affinity
    4. Classify product_area
    5. Run safety check
    6. Try pattern match override
    7. Decide status (replied / escalated) and generate response
    """
    text = ticket_text(ticket)
    request_type = classify_request_type(ticket)

    # Retrieve and re-rank
    retrieval_results = retriever.retrieve(ticket, top_k=5)
    top_result = _select_best_retrieval(retrieval_results, ticket)
    top_article = top_result["article"] if top_result else {}
    top_score = float(top_result["score"]) if top_result else 0.0
    top_title = top_article.get("title", "") if top_article else ""
    top_source = top_article.get("source", "") if top_article else ""

    product_area = classify_product_area(ticket, ticket.get("Company"), top_source)
    is_sensitive, safe_to_reply = safety_check(ticket, top_score, top_title)
    matched_article = pattern_match(ticket, corpus_entries)

    # --- Decision tree ---

    # 1. Invalid requests
    if request_type == "invalid":
        if is_high_risk_invalid(ticket):
            status = "escalated"
            response = HIGH_RISK_MESSAGE
            justification = "Escalated because the request is invalid and high-risk (potential malicious intent)."
        elif "thank you" in text:
            status = "replied"
            response = THANK_YOU_RESPONSE
            justification = "Replied with acknowledgment to a thank-you message."
        else:
            status = "replied"
            response = OUT_OF_SCOPE_MESSAGE
            justification = "Replied because the request is invalid/out-of-scope but not operationally risky."

    # 2. Pattern match override — highest-confidence path for known ticket types
    elif matched_article:
        # Even with a pattern match, check if the topic is too sensitive to auto-reply
        matched_is_sample = matched_article.get("entry_type") == "sample_ticket"
        if is_sensitive and not safe_to_reply and not matched_is_sample:
            status = "escalated"
            response = LOW_CONFIDENCE_MESSAGE
            justification = (
                f"Pattern match found: '{matched_article.get('title', 'Unknown')}', "
                f"but escalated due to sensitive topic (request_type={request_type})."
            )
        else:
            status = "replied"
            response = format_article_response(matched_article)
            justification = (
                f"Pattern match override used: '{matched_article.get('title', 'Unknown')}'. "
                f"request_type={request_type}; safe_to_reply={safe_to_reply}."
            )
        if matched_article.get("product_area"):
            product_area = matched_article["product_area"]
        if matched_article.get("request_type") and matched_is_sample:
            request_type = matched_article["request_type"]

    # 3. High-confidence retrieval (similarity >= 0.50 and safe)
    elif top_score >= 0.50 and safe_to_reply:
        status = "replied"
        response = format_article_response(top_article)
        justification = (
            f"Top match: '{top_title}' (similarity {top_score:.2f}); "
            f"request_type={request_type}; safe_to_reply={safe_to_reply}. Replied because confident and safe."
        )

    # 4. Moderate confidence (0.40-0.50) — reply only if clearly not sensitive
    elif top_score >= 0.40 and not is_sensitive:
        status = "replied"
        response = format_article_response(top_article)
        justification = (
            f"Top match: '{top_title}' (similarity {top_score:.2f}); "
            f"request_type={request_type}. Replied at moderate confidence, not sensitive."
        )

    # 5. Everything else → escalate
    else:
        status = "escalated"
        response = LOW_CONFIDENCE_MESSAGE
        reason = "sensitive topic" if is_sensitive and not safe_to_reply else "low retrieval confidence"
        justification = (
            f"Top match: '{top_title}' (similarity {top_score:.2f}); "
            f"request_type={request_type}; safe_to_reply={safe_to_reply}. Escalated due to {reason}."
        )

    return {
        "issue": (ticket.get("Issue") or "").strip(),
        "subject": (ticket.get("Subject") or "").strip(),
        "company": (ticket.get("Company") or "").strip(),
        "response": response,
        "product_area": product_area,
        "status": status,
        "request_type": request_type,
        "justification": justification,
    }
