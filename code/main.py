from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from corpus import load_corpus
from decision_engine import decide_ticket
from logger import log_turn
from retriever import Retriever


def set_deterministic_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_data_root(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data",
        repo_root / "hackerrank-orchestrate-may26" / "data",
    ]
    for candidate in candidates:
        if (candidate / "hackerrank").exists() or (candidate / "claude").exists() or (candidate / "visa").exists():
            return candidate
    raise FileNotFoundError("Could not find a valid data directory.")


def find_support_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "support_tickets",
        repo_root / "support_tickets" / "support_tickets",
        repo_root / "hackerrank-orchestrate-may26" / "support_tickets",
    ]
    for candidate in candidates:
        if (candidate / "support_tickets.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find support_tickets.csv in the expected locations.")


def load_ticket_dataframe(support_dir: Path) -> pd.DataFrame:
    ticket_path = support_dir / "support_tickets.csv"
    dataframe = pd.read_csv(ticket_path).fillna("")
    expected_columns = {"Issue", "Subject", "Company"}
    missing = expected_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns in {ticket_path}: {sorted(missing)}")
    return dataframe


def build_context(repo_root: Path) -> dict:
    return {
        "tool": "python_support_agent",
        "repo_root": str(repo_root.resolve()),
        "worktree": "main",
        "parent_agent": "none",
        "language": "py",
    }


def run_pipeline(repo_root: Path) -> Tuple[Path, Path]:
    set_deterministic_seed()
    data_root = find_data_root(repo_root)
    support_dir = find_support_dir(repo_root)
    sample_csv_path = support_dir / "sample_support_tickets.csv"
    output_path = support_dir / "output.csv"
    cache_dir = repo_root / "code"

    context = build_context(repo_root)
    log_turn(
        entry_type="session_start",
        title="SESSION START",
        user_prompt="",
        agent_response_summary="Support triage run started.",
        actions=["initialized orchestrator", f"data_root={data_root}", f"support_dir={support_dir}"],
        context=context,
    )

    corpus_entries = load_corpus(data_root, sample_csv_path=sample_csv_path)
    retriever = Retriever(corpus_entries, cache_dir=cache_dir)
    tickets_df = load_ticket_dataframe(support_dir)

    results = []
    for row_index, row in tickets_df.iterrows():
        ticket = {
            "Issue": str(row.get("Issue", "") or ""),
            "Subject": str(row.get("Subject", "") or ""),
            "Company": str(row.get("Company", "") or ""),
        }
        decision = decide_ticket(ticket, retriever=retriever, corpus_entries=corpus_entries)
        results.append(decision)

        summary = (
            f"Processed ticket with status={decision['status']}, request_type={decision['request_type']}, "
            f"product_area={decision['product_area']}."
        )
        actions = [
            f"retrieved top articles for row {row_index + 2}",
            f"decided status={decision['status']}",
            f"prepared output row for company={decision['company'] or 'None'}",
        ]
        log_turn(
            entry_type="per_turn",
            title=f"Ticket {row_index + 1}: {(ticket['Subject'] or ticket['Issue'] or 'No subject')[:60]}",
            user_prompt="\n".join(f"{key}: {value}" for key, value in ticket.items()),
            agent_response_summary=summary,
            actions=actions,
            context=context,
        )

    output_columns = [
        "issue",
        "subject",
        "company",
        "response",
        "product_area",
        "status",
        "request_type",
        "justification",
    ]
    output_df = pd.DataFrame(results, columns=output_columns)
    output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    log_turn(
        entry_type="per_turn",
        title="Output CSV Written",
        user_prompt="Write final support_tickets/output.csv",
        agent_response_summary=f"Wrote {len(output_df)} triaged ticket rows to {output_path}.",
        actions=[f"wrote {output_path}"],
        context=context,
    )

    return output_path, support_dir


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path, _ = run_pipeline(repo_root)
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
