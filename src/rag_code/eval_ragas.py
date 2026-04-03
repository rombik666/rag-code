import csv
import json
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from ragas.llms import llm_factory

try:
    from ragas.embeddings import HuggingFaceEmbeddings
except ImportError:
    from ragas.embeddings import HuggingfaceEmbeddings as HuggingFaceEmbeddings

try:
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
except ImportError:
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

from rag_code.config import BASE_DIR, INDEX_DIR, settings
from rag_code.generator import OpenAIChatGenerator
from rag_code.reranker import CrossEncoderReranker
from rag_code.retriever import FaissRetriever

INDEX_FILE_NAME = "faiss.index"
METADATA_FILE_NAME = "chunks_metadata.json"

EVAL_DIR = BASE_DIR / "data" / "eval"
EVAL_CASES_PATH = EVAL_DIR / "eval_cases.json"
RESULTS_CSV_PATH = EVAL_DIR / "ragas_results.csv"
SUMMARY_JSON_PATH = EVAL_DIR / "ragas_summary.json"


def load_eval_cases(file_path: Path) -> list[dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Evaluation file must contain a list of cases")

    return data


def extract_contexts(chunks: list[dict]) -> list[str]:
    return [chunk["text"] for chunk in chunks]


def build_rag_pipeline() -> tuple[FaissRetriever, CrossEncoderReranker, OpenAIChatGenerator]:
    retriever = FaissRetriever(
        embedding_model_name=settings.embedding_model_name,
        index_path=INDEX_DIR / INDEX_FILE_NAME,
        metadata_path=INDEX_DIR / METADATA_FILE_NAME,
        top_k=settings.top_k,
    )

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
    )

    generator = OpenAIChatGenerator(
        model_name=settings.llm_model_name,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    return retriever, reranker, generator

def build_ragas_metrics() -> tuple[Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall]:
    evaluator_client = AsyncOpenAI(
        api_key=settings.eval_llm_api_key,
        base_url=settings.eval_llm_base_url or None,
        http_client=DefaultAsyncHttpxClient(
            trust_env=False,
        ),
    )

    evaluator_llm = llm_factory(
        settings.eval_llm_model_name,
        client=evaluator_client,
        temperature=0,
    )

    evaluator_embeddings = HuggingFaceEmbeddings(
        model=settings.embedding_model_name,
    )

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    context_precision_metric = ContextPrecision(llm=evaluator_llm)
    context_recall_metric = ContextRecall(llm=evaluator_llm)

    return (
        faithfulness_metric,
        answer_relevancy_metric,
        context_precision_metric,
        context_recall_metric,
    )

def run_single_case(
    case: dict,
    retriever: FaissRetriever,
    reranker: CrossEncoderReranker,
    generator: OpenAIChatGenerator,
    faithfulness_metric: Faithfulness,
    answer_relevancy_metric: AnswerRelevancy,
    context_precision_metric: ContextPrecision,
    context_recall_metric: ContextRecall,
) -> dict:
    case_id = case["id"]
    query = case["query"]
    reference = case["reference"]

    faiss_results = retriever.retrieve(
        query=query,
        top_k=settings.top_k,
    )

    reranked_chunks = reranker.rerank(
        query=query,
        candidates=faiss_results,
        top_n=settings.rerank_top_n,
    )

    generation_result = generator.generate_answer(
        query=query,
        chunks=reranked_chunks,
    )

    answer = generation_result["answer"]
    retrieved_contexts = extract_contexts(reranked_chunks)

    faithfulness_result = faithfulness_metric.score(
        user_input=query,
        response=answer,
        retrieved_contexts=retrieved_contexts,
    )

    answer_relevancy_result = answer_relevancy_metric.score(
        user_input=query,
        response=answer,
    )

    context_precision_result = context_precision_metric.score(
        user_input=query,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )

    context_recall_result = context_recall_metric.score(
        user_input=query,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )

    return {
        "id": case_id,
        "query": query,
        "reference": reference,
        "answer": answer,
        "faithfulness": float(faithfulness_result.value),
        "answer_relevancy": float(answer_relevancy_result.value),
        "context_precision": float(context_precision_result.value),
        "context_recall": float(context_recall_result.value),
    }


def save_results_csv(rows: list[dict], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "query",
        "reference",
        "answer",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]) -> dict:
    return {
        "num_cases": len(rows),
        "avg_faithfulness": mean(row["faithfulness"] for row in rows),
        "avg_answer_relevancy": mean(row["answer_relevancy"] for row in rows),
        "avg_context_precision": mean(row["context_precision"] for row in rows),
        "avg_context_recall": mean(row["context_recall"] for row in rows),
    }


def save_summary_json(summary: dict, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    cases = load_eval_cases(EVAL_CASES_PATH)

    retriever, reranker, generator = build_rag_pipeline()

    (
        faithfulness_metric,
        answer_relevancy_metric,
        context_precision_metric,
        context_recall_metric,
    ) = build_ragas_metrics()

    rows: list[dict] = []

    for case in cases:
        row = run_single_case(
            case=case,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            faithfulness_metric=faithfulness_metric,
            answer_relevancy_metric=answer_relevancy_metric,
            context_precision_metric=context_precision_metric,
            context_recall_metric=context_recall_metric,
        )
        rows.append(row)

        print("-" * 80)
        print(f'Case: {row["id"]}')
        print(f'Query: {row["query"]}')
        print(f'Faithfulness: {row["faithfulness"]:.4f}')
        print(f'Answer Relevancy: {row["answer_relevancy"]:.4f}')
        print(f'Context Precision: {row["context_precision"]:.4f}')
        print(f'Context Recall: {row["context_recall"]:.4f}')

    summary = build_summary(rows)

    save_results_csv(rows, RESULTS_CSV_PATH)
    save_summary_json(summary, SUMMARY_JSON_PATH)

    print("=" * 80)
    print("RAGAS summary:")
    print(f'Cases: {summary["num_cases"]}')
    print(f'Average Faithfulness: {summary["avg_faithfulness"]:.4f}')
    print(f'Average Answer Relevancy: {summary["avg_answer_relevancy"]:.4f}')
    print(f'Average Context Precision: {summary["avg_context_precision"]:.4f}')
    print(f'Average Context Recall: {summary["avg_context_recall"]:.4f}')
    print(f"Detailed results saved to: {RESULTS_CSV_PATH}")
    print(f"Summary saved to: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()