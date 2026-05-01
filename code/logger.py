from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable


LOG_DIR = Path.home() / "hackerrank_orchestrate"
LOG_PATH = LOG_DIR / "log.txt"
CHALLENGE_END = datetime.fromisoformat("2026-05-02T11:00:00+05:30")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_PATH


def redact_secrets(text: str) -> str:
    redacted = text
    patterns = [
        r"(sk-[A-Za-z0-9_\-]{10,})",
        r"(api[_-]?key\s*[:=]\s*)([^\s,]+)",
        r"(token\s*[:=]\s*)([^\s,]+)",
        r"(authorization\s*[:=]\s*)([^\s,]+)",
    ]
    for pattern in patterns:
        redacted = re.sub(pattern, lambda match: f"{match.group(1)}[REDACTED]" if match.lastindex and match.lastindex > 1 else "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted


def git_branch(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        branch = completed.stdout.strip()
        return branch or "unknown"
    except Exception:
        return "unknown"


def format_time_remaining() -> str:
    now = datetime.now().astimezone()
    delta = CHALLENGE_END - now.astimezone(CHALLENGE_END.tzinfo)
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes <= 0:
        return "0d 0h 0m"
    days, rem_minutes = divmod(total_minutes, 60 * 24)
    hours, minutes = divmod(rem_minutes, 60)
    return f"{days}d {hours}h {minutes}m"


def _stringify_actions(actions: Iterable[str]) -> str:
    action_lines = [f"* {action}" for action in actions]
    return "\n".join(action_lines) if action_lines else "* none"


def log_turn(
    entry_type: str,
    title: str,
    user_prompt: str,
    agent_response_summary: str,
    actions,
    context,
) -> Path:
    log_path = ensure_log_dir()
    repo_root = Path(context.get("repo_root") or Path.cwd()).resolve()
    branch = context.get("branch") or git_branch(repo_root)
    worktree = context.get("worktree") or "main"
    parent_agent = context.get("parent_agent") or "none"
    agent_name = context.get("tool") or "python_support_agent"
    language = context.get("language") or "py"
    timestamp = iso_now()

    if entry_type == "session_start":
        block = (
            f"## [{timestamp}] SESSION START\n\n"
            f"Agent: {agent_name}\n"
            f"Repo Root: {repo_root}\n"
            f"Branch: {branch}\n"
            f"Worktree: {worktree}\n"
            f"Parent Agent: {parent_agent}\n"
            f"Language: {language}\n"
            f"Time Remaining: {format_time_remaining()}\n\n"
        )
    else:
        safe_prompt = redact_secrets(user_prompt)
        block = (
            f"## [{timestamp}] {title[:80]}\n\n"
            f"User Prompt (verbatim, secrets redacted):\n"
            f"{safe_prompt}\n\n"
            f"Agent Response Summary:\n"
            f"{agent_response_summary}\n\n"
            f"Actions:\n"
            f"{_stringify_actions(actions)}\n\n"
            f"Context:\n"
            f"tool={agent_name}\n"
            f"branch={branch}\n"
            f"repo_root={repo_root}\n"
            f"worktree={worktree}\n"
            f"parent_agent={parent_agent}\n\n"
        )

    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block)
    return log_path


def log_to_file(entry_type: str, title: str, user_prompt: str, agent_response_summary: str, actions, context) -> Path:
    return log_turn(entry_type, title, user_prompt, agent_response_summary, actions, context)
