from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
# Increased from 500 to capture more actionable content (phone numbers, steps, etc.)
MAX_SNIPPET_CHARS = 1500


def build_corpus_fingerprint(corpus_entries: List[Dict[str, str]]) -> str:
    hasher = hashlib.sha256()
    for entry in corpus_entries:
        for key in ("title", "text", "source", "path", "url", "entry_type"):
            hasher.update((entry.get(key, "") or "").encode("utf-8", errors="ignore"))
            hasher.update(b"\0")
    return hasher.hexdigest()


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


class Retriever:
    def __init__(self, corpus_entries: List[Dict[str, str]], cache_dir: Path):
        self.corpus_entries = corpus_entries
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.meta_path = self.cache_dir / "corpus_meta.pkl"
        self.model = SentenceTransformer(MODEL_NAME)
        self.fingerprint = build_corpus_fingerprint(corpus_entries)
        self.embeddings = self._load_or_create_embeddings()

    def _build_embedding_inputs(self) -> List[str]:
        """Build text inputs for embedding.

        Uses title + a generous snippet of the article body for better
        retrieval quality.  For sample tickets, uses the combined
        retrieval_text field.
        """
        texts = []
        for entry in self.corpus_entries:
            retrieval_text = entry.get("retrieval_text") or entry.get("text", "")
            snippet = retrieval_text[:MAX_SNIPPET_CHARS]
            texts.append(f"{entry.get('title', '')}\n{snippet}".strip())
        return texts

    def _load_or_create_embeddings(self) -> np.ndarray:
        if self.embeddings_path.exists() and self.meta_path.exists():
            try:
                with self.meta_path.open("rb") as handle:
                    meta = pickle.load(handle)
                if meta.get("fingerprint") == self.fingerprint and meta.get("model_name") == MODEL_NAME:
                    embeddings = np.load(self.embeddings_path)
                    return normalize_rows(embeddings.astype(np.float32))
            except Exception:
                pass

        embeddings = self.model.encode(
            self._build_embedding_inputs(),
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(self.embeddings_path, embeddings)
        with self.meta_path.open("wb") as handle:
            pickle.dump(
                {
                    "fingerprint": self.fingerprint,
                    "model_name": MODEL_NAME,
                    "count": len(self.corpus_entries),
                },
                handle,
            )
        return normalize_rows(embeddings)

    def retrieve(self, ticket: Dict[str, str], top_k: int = 5) -> List[Dict[str, object]]:
        """Retrieve the top-k most similar corpus entries for a ticket.

        Increased default top_k from 3 → 5 so the decision engine has more
        candidates to consider for source-preference re-ranking.
        """
        if not self.corpus_entries:
            return []

        query = "\n".join(
            part for part in (ticket.get("Issue", ""), ticket.get("Subject", ""), ticket.get("Company", "")) if part
        ).strip()
        query_embedding = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        query_embedding = normalize_rows(query_embedding)[0]
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Dict[str, object]] = []
        for idx in top_indices:
            results.append(
                {
                    "index": int(idx),
                    "score": float(similarities[idx]),
                    "article": self.corpus_entries[int(idx)],
                }
            )
        return results
