"""Core RAG pipeline: retrieve grounded context, then generate a cited answer."""

from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_RELATIVE_MARGIN,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from src.llm import LLMClient, StatusCallback, get_llm_client
from src.store import Chunk, cosine_similarity_search, get_connection, load_all

NOT_FOUND_MESSAGE = (
    "I don't have that information in the MLSC knowledge base. "
    "Please check with an MLSC coordinator or domain lead."
)

SYSTEM_PROMPT = """You are the MLSC Knowledge Assistant. Answer the user's question using ONLY \
the context excerpts provided below, which come from official MLSC documents.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge and do not guess.
- If the context does not contain enough information to answer the question, respond with exactly: \
"NOT_FOUND" (nothing else).
- If the question is only partially answered by the context, answer the part you can and note what \
is missing.
- When the context includes information from multiple documents, synthesize across all of them.
- For broad or open-ended questions (e.g. "tell me about X"), cover all relevant points from the \
context, not just the first one or two -- do not stop at a partial summary when more relevant \
information is available.
- Answer directly and naturally, as if you already know this. Never mention "provided context", "the context", "the \
provided documents", "based on the information given", or similar phrases -- just state the facts. \
The only exception is when explicitly noting that something is missing or unavailable.
- Be concise and factual: no filler, no repetition, no fabricated names, numbers, or policies not \
present in the context. Concise means "no wasted words," not "leave out relevant information.\""""


@dataclass
class RagResult:
    answer: str
    sources: list[str]
    retrieved_chunks: list[tuple[Chunk, float]] = field(default_factory=list)
    is_answerable: bool = True


class RagPipeline:
    def __init__(self, llm_client: LLMClient | None = None):
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self._llm = llm_client or get_llm_client()
        conn = get_connection()
        self._chunks, self._matrix = load_all(conn)
        conn.close()
        if not self._chunks:
            raise RuntimeError("Knowledge base is empty. Run `python -m src.ingest` first.")

    def _retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        query_embedding = self._embedder.encode(question, convert_to_numpy=True)
        results = cosine_similarity_search(query_embedding, self._matrix, RETRIEVAL_TOP_K)
        if not results:
            return []
        top_score = results[0][1]
        results = [r for r in results if r[1] >= top_score - RETRIEVAL_RELATIVE_MARGIN]
        return [(self._chunks[idx], score) for idx, score in results]

    def _build_user_prompt(self, question: str, retrieved: list[tuple[Chunk, float]]) -> str:
        context_blocks = []
        for chunk, _ in retrieved:
            context_blocks.append(f"[Source: {chunk.source_file}]\n{chunk.text}")
        context = "\n\n".join(context_blocks)
        return f"Context:\n{context}\n\nQuestion: {question}"

    def answer(self, question: str, on_retry: StatusCallback | None = None) -> RagResult:
        retrieved = self._retrieve(question)

        if not retrieved or retrieved[0][1] < RETRIEVAL_SCORE_THRESHOLD:
            return RagResult(
                answer=NOT_FOUND_MESSAGE,
                sources=[],
                retrieved_chunks=retrieved,
                is_answerable=False,
            )

        user_prompt = self._build_user_prompt(question, retrieved)
        raw_answer = self._llm.generate(SYSTEM_PROMPT, user_prompt, on_retry=on_retry)

        if raw_answer.strip().upper().startswith("NOT_FOUND"):
            return RagResult(
                answer=NOT_FOUND_MESSAGE,
                sources=[],
                retrieved_chunks=retrieved,
                is_answerable=False,
            )

        unique_sources = list(dict.fromkeys(chunk.source_file for chunk, _ in retrieved))
        return RagResult(
            answer=raw_answer,
            sources=unique_sources,
            retrieved_chunks=retrieved,
            is_answerable=True,
        )


if __name__ == "__main__":
    import os

    pipeline = RagPipeline()
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'gemini')}\n")

    questions = [
        "What technical domains exist in MLSC?",
        "Who leads the AI/ML domain and how do I become a member?",
        "What is the capital of France?",
    ]
    for q in questions:
        result = pipeline.answer(q)
        print(f"Q: {q}")
        print(f"A: {result.answer}")
        print(f"Sources: {result.sources}")
        print()
