"""
src/retriever.py

Retrieval implementations used across all tests.
Self-contained — no external vector DB needed.

Implementations:
    BM25Retriever     — Keyword-based retrieval (sparse)
    CosineRetriever   — Embedding-based retrieval (dense)
    HybridRetriever   — RRF fusion of BM25 + Cosine
"""

import math
import re
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int
    retriever: str


class BM25Retriever:
    """
    BM25 retrieval. Fast, no external dependencies.
    Good for exact keyword matches. Fails on semantic queries.

    Args:
        k1: Term saturation parameter (default 1.5)
        b:  Length normalisation parameter (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[dict] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0

    def index(self, documents: list[dict]):
        """
        Index a list of documents.

        Args:
            documents: List of dicts with 'id' and 'text' keys.
        """
        self._docs = documents
        self._tf = []
        self._df = {}

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for term in set(tokens):
                self._df[term] = self._df.get(term, 0) + 1

        total_len = sum(len(self._tokenize(d["text"])) for d in documents)
        self._avgdl = total_len / len(documents) if documents else 1.0

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k documents for query."""
        if not self._docs:
            raise RuntimeError("Index is empty. Call .index(documents) first.")

        query_terms = self._tokenize(query)
        n = len(self._docs)
        scores = []

        for i, doc in enumerate(self._docs):
            dl = sum(self._tf[i].values())
            score = 0.0
            for term in query_terms:
                if term not in self._tf[i]:
                    continue
                tf_val = self._tf[i][term]
                df_val = self._df.get(term, 0)
                idf = math.log((n - df_val + 0.5) / (df_val + 0.5) + 1)
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * numerator / denominator
            scores.append((score, i))

        scores.sort(reverse=True)
        return [
            RetrievalResult(
                chunk_id=self._docs[idx]["id"],
                text=self._docs[idx]["text"],
                score=round(score, 4),
                rank=rank + 1,
                retriever="bm25",
            )
            for rank, (score, idx) in enumerate(scores[:top_k])
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        stopwords = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "with", "by", "from", "be", "are",
        }
        tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
        return [t for t in tokens if t not in stopwords]


class CosineRetriever:
    """
    Cosine similarity retrieval using TF-IDF vectors.
    No embedding API needed — suitable for tests.
    In production: replace _vectorize() with a real embedding model.

    Demonstrates semantic matching but also semantic mismatch failures.
    """

    def __init__(self):
        self._docs: list[dict] = []
        self._vectors: list[dict[str, float]] = []

    def index(self, documents: list[dict]):
        """Index documents by computing TF-IDF vectors."""
        self._docs = documents
        tf_lists = [self._tf(d["text"]) for d in documents]
        df: dict[str, int] = {}
        n = len(documents)

        for tf in tf_lists:
            for term in tf:
                df[term] = df.get(term, 0) + 1

        self._vectors = []
        for tf in tf_lists:
            vec = {}
            for term, freq in tf.items():
                idf = math.log((n + 1) / (df.get(term, 0) + 1))
                vec[term] = freq * idf
            self._vectors.append(self._normalize(vec))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k documents by cosine similarity to query."""
        if not self._docs:
            raise RuntimeError("Index is empty. Call .index(documents) first.")

        q_vec = self._normalize(self._tf(query))
        scores = []
        for i, doc_vec in enumerate(self._vectors):
            sim = self._cosine(q_vec, doc_vec)
            scores.append((sim, i))
        scores.sort(reverse=True)

        return [
            RetrievalResult(
                chunk_id=self._docs[idx]["id"],
                text=self._docs[idx]["text"],
                score=round(score, 4),
                rank=rank + 1,
                retriever="cosine",
            )
            for rank, (score, idx) in enumerate(scores[:top_k])
        ]

    @staticmethod
    def _tf(text: str) -> dict[str, float]:
        stopwords = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "with", "by", "from", "be", "are",
        }
        tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
        counts: dict[str, float] = {}
        for t in tokens:
            if t not in stopwords:
                counts[t] = counts.get(t, 0) + 1
        total = sum(counts.values()) or 1
        return {w: c / total for w, c in counts.items()}

    @staticmethod
    def _normalize(vec: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1e-9
        return {k: v / norm for k, v in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(a.get(k, 0) * v for k, v in b.items())


class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) of BM25 + Cosine.
    Combines keyword precision with semantic recall.

    Args:
        alpha:  Weight for dense (cosine) scores in fusion (0=all BM25, 1=all cosine)
        rrf_k:  RRF constant (default 60 from the original paper)
    """

    def __init__(self, alpha: float = 0.5, rrf_k: int = 60):
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self.cosine = CosineRetriever()

    def index(self, documents: list[dict]):
        """Index documents in both retrievers."""
        self.bm25.index(documents)
        self.cosine.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve using RRF fusion of both retrievers."""
        # Get more candidates from each
        fetch_k = min(top_k * 3, 20)
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)
        cosine_results = self.cosine.retrieve(query, top_k=fetch_k)

        # RRF scoring
        rrf_scores: dict[str, float] = {}
        doc_text: dict[str, str] = {}

        for r in bm25_results:
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1 / (self.rrf_k + r.rank)
            doc_text[r.chunk_id] = r.text
        for r in cosine_results:
            rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0) + 1 / (self.rrf_k + r.rank)
            doc_text[r.chunk_id] = r.text

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                text=doc_text[chunk_id],
                score=round(score, 6),
                rank=rank + 1,
                retriever="hybrid_rrf",
            )
            for rank, (chunk_id, score) in enumerate(ranked)
        ]
