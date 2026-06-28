
# Orchestrate Triage Agent

A multi-domain **support ticket triage system** that classifies, retrieves relevant documentation, and determines the correct response or escalation path across **HackerRank**, **Claude (Anthropic)**, and **Visa** support scenarios.

Built entirely with **local Retrieval-Augmented Generation (RAG)** using open-source models—**no API keys or paid LLMs required**.

## Live Demo

**Hugging Face Space:** https://huggingface.co/spaces/nishtha711/orchestrate-triage-agent

---

## Features

- Multi-domain support ticket triage
- Offline Retrieval-Augmented Generation (RAG)
- No OpenAI or Anthropic API required
- SentenceTransformer semantic search
- Deterministic classification pipeline
- Safety-aware escalation logic
- Pattern-based decision overrides
- Interactive Gradio interface
- Session logging
- Batch CSV processing support

---

# Pipeline

```
Support Ticket
      │
      ▼
Classifier
(keyword + regex heuristics)
      │
      ├── Request Type
      │     • bug
      │     • feature_request
      │     • product_issue
      │     • invalid
      │
      └── Safety Detection
            • malicious requests
            • sensitive information
            • escalation triggers
      │
      ▼
Retriever
SentenceTransformers
(all-MiniLM-L6-v2)
Cosine Similarity Search
      │
      ▼
Top Relevant Support Articles
(HackerRank • Claude • Visa)
      │
      ▼
Decision Engine
      │
      ├── Pattern Matching (30+ rules)
      ├── Source-aware ranking
      └── Escalate / Respond
      │
      ▼
Structured Output

• Status
• Product Area
• Response
• Justification
```

---

# Design Decisions

### Local RAG

Uses the lightweight **sentence-transformers/all-MiniLM-L6-v2** embedding model (~80 MB) for semantic retrieval.

### Corpus Grounding

Every generated response is grounded in an actual support article to minimize hallucinations.

### Safety First

Automatically escalates:

- Identity theft
- Malicious requests
- Unauthorized workspace access
- Sensitive account issues

### Deterministic Pipeline

No temperature sampling or stochastic generation, ensuring reproducible outputs.

---

# Project Structure

```
.
├── app.py                      # Gradio UI
├── requirements.txt
├── LICENSE
├── README.md
├── problem_statement.md
├── test_run.sh
│
├── code/
│   ├── README.md
│   ├── classifier.py
│   ├── corpus.py
│   ├── decision_engine.py
│   ├── logger.py
│   ├── main.py
│   ├── requirements.txt
│   └── retriever.py
│
└── support_tickets/
    └── support_tickets/
        ├── output.csv
        ├── sample_support_tickets.csv
        └── support_tickets.csv
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/711nishtha/orchestrate-triage-agent.git
cd orchestrate-triage-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

The first launch downloads the embedding model (`all-MiniLM-L6-v2`) and caches embeddings locally for faster subsequent runs.

---

# Batch CSV Processing

To process support tickets from a CSV file:

```bash
python code/main.py
```

Predictions are saved to:

```
support_tickets/output.csv
```

---

# Dataset & Support Corpus

To keep the GitHub repository lightweight, the support corpus and dataset are **not included** in this repository.

They are available in the Hugging Face Space used for the live demo:

https://huggingface.co/spaces/nishtha711/orchestrate-triage-agent

This includes:

- Support knowledge base
- Demo corpus
- Sample support tickets
- Batch processing data

---

# Tech Stack

- Python
- Gradio
- Sentence Transformers
- scikit-learn
- NumPy
- Regex
- YAML
- Cosine Similarity Search

---

# Supported Domains

- HackerRank Support
- Claude (Anthropic) Support
- Visa Support

---

# Future Improvements

- Hybrid BM25 + Dense Retrieval
- LLM-powered response generation
- Confidence scoring
- Additional enterprise support domains
- Feedback-driven continual learning

---

# License

MIT License
