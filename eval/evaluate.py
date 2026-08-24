"""Evaluate the RAG pipeline against eval/eval_set.json.

Defaults LLM_PROVIDER to Ollama (local, no rate limit) so a full eval run of
many back-to-back LLM calls doesn't get throttled by Gemini's free-tier
quota. The pipeline code under test is identical either way -- only the
backend swaps -- so this still evaluates the real system. Override with
LLM_PROVIDER=gemini before running if you want to spot-check the cloud
backend instead.

Run from the project root with the venv active:
    python -m eval.evaluate
"""

import json
import os
import re
from pathlib import Path

os.environ.setdefault("LLM_PROVIDER", "ollama")

from src.llm import get_llm_client  # noqa: E402
from src.rag import RagPipeline  # noqa: E402

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_JSON_PATH = Path(__file__).parent / "results.json"
RESULTS_MD_PATH = Path(__file__).parent / "results.md"

RELEVANCY_SYSTEM_PROMPT = """You are an evaluation judge. Rate how directly and completely the \
ANSWER addresses the QUESTION, on a scale of 1 to 5:
5 = fully and directly addresses the question
3 = partially addresses it
1 = irrelevant, or evades a question that should be answerable
If the answer correctly states the information isn't available AND the question genuinely \
cannot be answered from a general MLSC knowledge base, that counts as fully relevant (5).
Respond with ONLY a single integer from 1 to 5, nothing else."""

FAITHFULNESS_SYSTEM_PROMPT = """You are an evaluation judge checking groundedness. Given the \
CONTEXT and an ANSWER derived from it, rate whether every claim in the ANSWER is supported by \
the CONTEXT, on a scale of 1 to 5:
5 = fully supported by the context, no fabricated or unsupported claims
3 = mostly supported, minor unsupported details
1 = contains claims that contradict or are not present in the context
Respond with ONLY a single integer from 1 to 5, nothing else."""


def parse_score(text: str) -> float:
    match = re.search(r"[1-5]", text)
    return float(match.group()) if match else 3.0  # neutral fallback if judge misbehaves


def precision_recall(retrieved_sources: list[str], expected_sources: list[str]) -> tuple[float, float]:
    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)
    overlap = len(retrieved_set & expected_set)
    precision = overlap / len(retrieved_set) if retrieved_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 0.0
    return precision, recall


def evaluate() -> None:
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    provider = os.environ["LLM_PROVIDER"]
    print(f"Evaluating with LLM_PROVIDER={provider} ({len(eval_set)} questions)\n")

    pipeline = RagPipeline()
    judge = get_llm_client()

    results = []
    for item in eval_set:
        question = item["question"]
        expected_sources = item["expected_sources"]
        is_unanswerable_case = item["category"] == "unanswerable"

        print(f"[{item['id']}/{len(eval_set)}] ({item['category']}) {question}")
        rag_result = pipeline.answer(question)
        retrieved_sources = list(dict.fromkeys(c.source_file for c, _ in rag_result.retrieved_chunks))

        answerable_correct = rag_result.is_answerable != is_unanswerable_case

        if is_unanswerable_case:
            precision, recall = None, None
        else:
            precision, recall = precision_recall(retrieved_sources, expected_sources)

        relevancy_raw = judge.generate(
            RELEVANCY_SYSTEM_PROMPT, f"Question: {question}\nAnswer: {rag_result.answer}"
        )
        relevancy = parse_score(relevancy_raw) / 5.0

        if rag_result.is_answerable:
            context = "\n\n".join(c.text for c, _ in rag_result.retrieved_chunks)
            faithfulness_raw = judge.generate(
                FAITHFULNESS_SYSTEM_PROMPT, f"Context:\n{context}\n\nAnswer: {rag_result.answer}"
            )
            faithfulness = parse_score(faithfulness_raw) / 5.0
        else:
            # A fixed, non-fabricating refusal message can't contain unsupported claims.
            faithfulness = 1.0

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": question,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "answer": rag_result.answer,
                "is_answerable": rag_result.is_answerable,
                "answerable_correct": answerable_correct,
                "context_precision": precision,
                "context_recall": recall,
                "answer_relevancy": relevancy,
                "faithfulness": faithfulness,
            }
        )

    write_reports(results, provider)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_aggregates(results: list[dict]) -> dict:
    """Aggregate metrics used both in the written report and the UI's eval snapshot."""
    answerable_results = [r for r in results if r["context_precision"] is not None]
    return {
        "n_questions": len(results),
        "n_answerable": len(answerable_results),
        "context_precision": _avg([r["context_precision"] for r in answerable_results]),
        "context_recall": _avg([r["context_recall"] for r in answerable_results]),
        "answer_relevancy": _avg([r["answer_relevancy"] for r in results]),
        "faithfulness": _avg([r["faithfulness"] for r in results]),
        "answerable_accuracy": _avg([1.0 if r["answerable_correct"] else 0.0 for r in results]),
    }


def write_reports(results: list[dict], provider: str) -> None:
    RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    agg = compute_aggregates(results)
    answerable_results = [r for r in results if r["context_precision"] is not None]
    precisions = [r["context_precision"] for r in answerable_results]
    recalls = [r["context_recall"] for r in answerable_results]
    relevancies = [r["answer_relevancy"] for r in results]
    faithfulnesses = [r["faithfulness"] for r in results]
    answerable_accuracy = agg["answerable_accuracy"]

    categories = sorted(set(r["category"] for r in results))

    lines = [
        "# MLSC Knowledge Assistant -- Evaluation Results",
        "",
        f"LLM provider used for this run: `{provider}`",
        f"Total questions: {len(results)}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Context Precision (answerable questions only, n={len(answerable_results)}) | {_avg(precisions):.2f} |",
        f"| Context Recall (answerable questions only, n={len(answerable_results)}) | {_avg(recalls):.2f} |",
        f"| Answer Relevancy (all questions) | {_avg(relevancies):.2f} |",
        f"| Faithfulness / Groundedness (all questions) | {_avg(faithfulnesses):.2f} |",
        f"| Answerable/Unanswerable Detection Accuracy | {answerable_accuracy:.2f} |",
        "",
        "## Per-category breakdown",
        "",
        "| Category | n | Avg Precision | Avg Recall | Avg Relevancy | Avg Faithfulness | Detection Accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        cat_answerable = [r for r in cat_results if r["context_precision"] is not None]
        p = _avg([r["context_precision"] for r in cat_answerable])
        rcl = _avg([r["context_recall"] for r in cat_answerable])
        rel = _avg([r["answer_relevancy"] for r in cat_results])
        faith = _avg([r["faithfulness"] for r in cat_results])
        acc = _avg([1.0 if r["answerable_correct"] else 0.0 for r in cat_results])
        p_str = f"{p:.2f}" if cat_answerable else "n/a"
        rcl_str = f"{rcl:.2f}" if cat_answerable else "n/a"
        lines.append(f"| {category} | {len(cat_results)} | {p_str} | {rcl_str} | {rel:.2f} | {faith:.2f} | {acc:.2f} |")

    lines += ["", "## Per-question results", ""]
    lines.append("| ID | Category | Question | Expected Sources | Retrieved Sources | Precision | Recall | Relevancy | Faithfulness | Answerable OK |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        p_str = f"{r['context_precision']:.2f}" if r["context_precision"] is not None else "n/a"
        rcl_str = f"{r['context_recall']:.2f}" if r["context_recall"] is not None else "n/a"
        lines.append(
            f"| {r['id']} | {r['category']} | {r['question']} | "
            f"{', '.join(r['expected_sources']) or '-'} | {', '.join(r['retrieved_sources']) or '-'} | "
            f"{p_str} | {rcl_str} | {r['answer_relevancy']:.2f} | {r['faithfulness']:.2f} | "
            f"{'yes' if r['answerable_correct'] else 'NO'} |"
        )

    RESULTS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS_JSON_PATH} and {RESULTS_MD_PATH}")


if __name__ == "__main__":
    evaluate()
