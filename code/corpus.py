from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup


SOURCE_BASE_URLS = {
    "hackerrank": "https://support.hackerrank.com",
    "claude": "https://support.anthropic.com",
    "visa": "https://www.visa.co.in",
}


# ---------------------------------------------------------------------------
# HTML cleaning (still used for any .html files if they exist)
# ---------------------------------------------------------------------------

def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    body = soup.body or soup
    text = body.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown parsing — frontmatter extraction + content cleaning
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_yaml_frontmatter(raw: str) -> Dict[str, str]:
    """Lightweight YAML frontmatter parser (avoids pyyaml dependency).
    Handles simple key: "value" pairs found in the corpus files."""
    meta: Dict[str, str] = {}
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return meta
    block = match.group(1)
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        colon_pos = line.find(":")
        if colon_pos < 0:
            continue
        key = line[:colon_pos].strip()
        value = line[colon_pos + 1:].strip().strip('"').strip("'")
        meta[key] = value
    return meta


def _strip_frontmatter(raw: str) -> str:
    """Remove YAML frontmatter block from markdown content."""
    match = _FRONTMATTER_RE.match(raw)
    if match:
        return raw[match.end():]
    return raw


def _clean_markdown_text(raw: str) -> str:
    """Clean markdown to plain-text for embedding/retrieval."""
    body = _strip_frontmatter(raw)
    # Remove markdown image/link syntax but keep link text
    body = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", body)
    # Remove HTML-style tags that may be embedded in md
    body = re.sub(r"<[^>]+>", "", body)
    # Remove markdown emphasis markers
    body = re.sub(r"\*{1,3}|_{1,3}", "", body)
    # Collapse excessive whitespace but keep paragraph breaks
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return "\n".join(lines)


def parse_markdown_file(path: Path, source: str, source_root: Path) -> Dict[str, str]:
    """Parse a .md corpus file with YAML frontmatter."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    meta = _parse_yaml_frontmatter(raw)

    # Title: prefer frontmatter, then first # heading, then filename
    title = meta.get("title", "")
    if not title:
        heading_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        if heading_match:
            title = heading_match.group(1).strip()
        else:
            title = path.stem.replace("-", " ").replace("_", " ").strip() or "Untitled"

    # URL: prefer source_url from frontmatter
    url = meta.get("source_url", "")
    if not url:
        url = build_article_url(path, source, source_root)

    text = _clean_markdown_text(raw)

    return {
        "title": title,
        "text": text,
        "source": source,
        "path": str(path),
        "url": url,
        "entry_type": "article",
    }


# ---------------------------------------------------------------------------
# HTML parsing (kept for backward compatibility)
# ---------------------------------------------------------------------------

def build_article_url(path: Path, source: str, source_root: Path) -> str:
    from urllib.parse import quote

    relative = path.relative_to(source_root).as_posix()
    base_url = SOURCE_BASE_URLS.get(source)
    if not base_url:
        return relative

    quoted_relative = "/".join(quote(part) for part in Path(relative).parts)
    if source == "hackerrank":
        return f"{base_url}/hc/en-us/articles/{quoted_relative}"
    if source == "claude":
        return f"{base_url}/en/articles/{quoted_relative}"
    if source == "visa":
        return f"{base_url}/support/{quoted_relative}"
    return relative


def parse_html_file(path: Path, source: str, source_root: Path) -> Dict[str, str]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    else:
        title = path.stem.replace("-", " ").replace("_", " ").strip() or "Untitled"

    text = clean_html_text(html)

    return {
        "title": title,
        "text": text,
        "source": source,
        "path": str(path),
        "url": build_article_url(path, source, source_root),
        "entry_type": "article",
    }


# ---------------------------------------------------------------------------
# Corpus loading — now handles both .md and .html
# ---------------------------------------------------------------------------

def load_html_corpus(data_root: Path) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for source in ("hackerrank", "claude", "visa"):
        source_root = data_root / source
        if not source_root.exists():
            continue

        # Load markdown files (primary corpus format)
        for md_path in sorted(source_root.rglob("*.md")):
            # Skip index.md files — they are table-of-contents, not articles
            if md_path.stem == "index":
                continue
            entries.append(parse_markdown_file(md_path, source, source_root))

        # Load HTML files (backward compatibility)
        for html_path in sorted(source_root.rglob("*.html")):
            entries.append(parse_html_file(html_path, source, source_root))

    return entries


def load_sample_ticket_entries(sample_csv_path: Optional[Path]) -> List[Dict[str, str]]:
    if not sample_csv_path or not sample_csv_path.exists():
        return []

    entries: List[Dict[str, str]] = []
    with sample_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            issue = (row.get("Issue") or "").strip()
            subject = (row.get("Subject") or "").strip()
            company = (row.get("Company") or "").strip()
            response = (row.get("Response") or "").strip()
            if not response:
                continue

            title = " | ".join(part for part in (subject, company) if part) or f"Sample Ticket {idx + 1}"
            ticket_text = "\n".join(part for part in (issue, subject, company, response) if part)
            entries.append(
                {
                    "title": title,
                    "text": response,
                    "source": company.lower() if company else "sample",
                    "path": f"{sample_csv_path}#row-{idx + 2}",
                    "url": str(sample_csv_path),
                    "entry_type": "sample_ticket",
                    "ticket_key": " | ".join(part for part in (issue, subject, company) if part),
                    "request_type": (row.get("Request Type") or "").strip().lower(),
                    "product_area": (row.get("Product Area") or "").strip().lower().replace(" ", "_"),
                    "status": (row.get("Status") or "").strip().lower(),
                    "retrieval_text": ticket_text,
                    "sample_issue": issue.lower(),
                    "sample_subject": subject.lower(),
                    "sample_company": company.lower(),
                }
            )
    return entries


def load_corpus(data_root: Path, sample_csv_path: Optional[Path] = None) -> List[Dict[str, str]]:
    entries = load_html_corpus(data_root)
    entries.extend(load_sample_ticket_entries(sample_csv_path))
    return entries
