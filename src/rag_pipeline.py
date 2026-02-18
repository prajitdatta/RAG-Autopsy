"""
src/rag_pipeline.py

The full RAG pipeline used across all tests.
Configurable at each stage so tests can inject failures.

Usage:
    from src.rag_pipeline import RAGPipeline

    # Default (correct) pipeline
    pipeline = RAGPipeline()
    pipeline.index(documents)
    result = pipeline.query("What is the refund period?")

    # Broken pipeline (for failure tests)
    broken = RAGPipeline(chunker_strategy="fixed", top_k=1, generator_mode="hallucinating")
    broken.index(documents)
    result = broken.query("What is the refund period?")
"""

from dataclasses import dataclass, field
from src.chunker import Chunker, Chunk
from src.retriever import HybridRetriever, BM25Retriever, CosineRetriever, RetrievalResult
from src.generator import MockGenerator, GenerationResult
from src.evaluator import Evaluator, EvaluationResult


@dataclass
class PipelineResult:
    query: str
    chunks_retrieved: list[RetrievalResult]
    generation: GenerationResult
    evaluation: EvaluationResult | None = None

    def passed_eval(self) -> bool:
        return self.evaluation.passed if self.evaluation else False

    def __repr__(self):
        eval_status = self.evaluation.summary() if self.evaluation else "not evaluated"
        return (
            f"PipelineResult(\n"
            f"  query='{self.query[:60]}',\n"
            f"  chunks_retrieved={len(self.chunks_retrieved)},\n"
            f"  response='{self.generation.response[:80]}',\n"
            f"  eval={eval_status}\n"
            f")"
        )


class RAGPipeline:
    """
    Configurable RAG pipeline. Swap components to inject or fix failure modes.

    Args:
        chunker_strategy:   "fixed" | "sentence" | "recursive" | "semantic"
        chunk_size:         Target chunk size in characters
        overlap:            Overlap between fixed chunks
        retriever_type:     "hybrid" | "bm25" | "cosine"
        top_k:              Number of chunks to retrieve
        generator_mode:     "faithful" | "hallucinating" | "ignoring" | "injected"
        evaluate:           Whether to auto-evaluate each query result
    """

    def __init__(
        self,
        chunker_strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
        retriever_type: str = "hybrid",
        top_k: int = 5,
        generator_mode: str = "faithful",
        evaluate: bool = False,
    ):
        self.top_k = top_k
        self.evaluate_results = evaluate

        self.chunker = Chunker(
            strategy=chunker_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if retriever_type == "bm25":
            self.retriever = BM25Retriever()
        elif retriever_type == "cosine":
            self.retriever = CosineRetriever()
        else:
            self.retriever = HybridRetriever()

        self.generator = MockGenerator(mode=generator_mode)
        self.evaluator = Evaluator()

        self._chunks: list[Chunk] = []
        self._indexed = False

    def index(self, documents: list[dict]):
        """
        Chunk and index documents.

        Args:
            documents: List of dicts with 'id' (str) and 'text' (str).
        """
        self._chunks = []
        indexed_docs = []

        for doc in documents:
            chunks = self.chunker.chunk(doc["text"])
            for chunk in chunks:
                chunk_id = f"{doc['id']}_chunk_{chunk.index}"
                self._chunks.append(chunk)
                indexed_docs.append({"id": chunk_id, "text": chunk.text})

        self.retriever.index(indexed_docs)
        self._indexed = True

    def query(
        self,
        query: str,
        key_facts: list[str] | None = None,
    ) -> PipelineResult:
        """
        Run a query through the full pipeline.

        Args:
            query:     User's question
            key_facts: Optional list of facts that should appear in the answer (for eval)

        Returns:
            PipelineResult with retrieval, generation, and optional evaluation
        """
        if not self._indexed:
            raise RuntimeError("Pipeline not indexed. Call .index(documents) first.")

        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        context_texts = [r.text for r in retrieved]

        generation = self.generator.generate(query, context_texts)

        evaluation = None
        if self.evaluate_results:
            evaluation = self.evaluator.evaluate(
                query=query,
                response=generation.response,
                context_chunks=context_texts,
                key_facts=key_facts,
                citations=generation.citations,
            )

        return PipelineResult(
            query=query,
            chunks_retrieved=retrieved,
            generation=generation,
            evaluation=evaluation,
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunk_sizes(self) -> list[int]:
        return [len(c.text) for c in self._chunks]
