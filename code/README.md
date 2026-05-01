# Support Triage Agent

A deterministic, terminal-based support triage pipeline for HackerRank Orchestrate. It parses 774 Markdown articles from the local support corpus, embeds them with `sentence-transformers`, classifies each ticket via pattern matching + safety heuristics, and writes `support_tickets/output.csv` plus append-only logs to `$HOME/hackerrank_orchestrate/log.txt`.

## Install

From the repository root:

```bash
cd code
pip install -r requirements.txt
```

No API keys are required. The only network dependency is the first-time download of the `all-MiniLM-L6-v2` sentence-transformer model (~80 MB).

## Run

From the repository root:

```bash
python code/main.py
```

Or use the helper script:

```bash
bash test_run.sh
```

The script will:

1. Discover the local `data/` and `support_tickets/` directories.
2. Parse **Markdown** articles (with YAML frontmatter extraction) and ingest `sample_support_tickets.csv` as extra retrieval context.
3. Build or reuse cached embeddings in `code/embeddings.npy` and `code/corpus_meta.pkl`.
4. Process `support_tickets/support_tickets.csv`.
5. Write `support_tickets/output.csv` with these exact columns:

```text
issue, subject, company, response, product_area, status, request_type, justification
```

## Architecture

```
main.py              → Pipeline orchestrator & CSV I/O
corpus.py            → Markdown/HTML corpus loader with frontmatter parser
retriever.py         → SentenceTransformer embedding + cosine similarity retrieval
classifier.py        → Request type, product area, safety, & pattern matching
decision_engine.py   → Integrates classifier + retriever → status & response
logger.py            → Append-only logging to $HOME/hackerrank_orchestrate/log.txt
```

## Key Design Decisions

- **Corpus-grounded responses**: Every reply is backed by an actual article from the local corpus. No hallucinated policies.
- **Pattern matching overrides**: 30+ high-confidence patterns for known ticket types (account deletion, mock interviews, Visa lost card, etc.) bypass low-confidence retrieval.
- **Safety-first escalation**: Identity theft, non-owner workspace access, and malicious requests (social engineering, system deletion) are always escalated.
- **Source-aware re-ranking**: Retrieval results are boosted when they match the ticket's Company column.
- **Deterministic**: No randomness, no temperature sampling, no external API calls.
