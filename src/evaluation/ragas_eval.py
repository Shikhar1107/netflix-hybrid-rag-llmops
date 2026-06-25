import os
import mlflow
from datasets import Dataset
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from src.pipeline.rag_pipeline import run_rag_pipeline
from src.evaluation.test_dataset import TEST_QUERIES
from src.generation.llm_chain import RAG_PROMPT
from src.ingestion.embedder import get_embedding_model


load_dotenv()


def get_ragas_llm():
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Netflix Content Intelligence RAG Evaluation",
        },
    )

    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    embeddings = get_embedding_model()
    return LangchainEmbeddingsWrapper(embeddings)


def build_ragas_dataset(retrieval_mode: str):
    questions = []
    answers = []
    contexts = []
    references = []

    for item in TEST_QUERIES:
        question = item["question"]
        reference = item["ground_truth"]

        result = run_rag_pipeline(
            question=question,
            retrieval_mode=retrieval_mode,
        )

        if not result["success"]:
            print(f"Failed query: {question}")
            print(result["error"])
            continue

        context_list = []

        context_list = result.get("contexts", [])

        questions.append(question)
        answers.append(result["answer"])
        contexts.append(context_list)
        references.append(reference)

    dataset = Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": references,
        }
    )

    return dataset


def run_ragas_evaluation(retrieval_mode: str):
    dataset = build_ragas_dataset(retrieval_mode=retrieval_mode)

    evaluator_llm = get_ragas_llm()
    evaluator_embeddings = get_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            ),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ],
    )

    return result


def log_to_mlflow(results, retrieval_mode: str):
    mlflow.set_experiment("netflix-rag-evaluation")

    with mlflow.start_run(run_name=f"rag-{retrieval_mode}"):
        mlflow.log_param("retrieval_mode", retrieval_mode)
        mlflow.log_param("retrieval_stack", f"{retrieval_mode}+reranker")
        mlflow.log_param("embedding_model", "BAAI/bge-base-en-v1.5")
        mlflow.log_param("vector_db", "ChromaDB")
        mlflow.log_param("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        mlflow.log_param("llm_provider", "OpenRouter")
        mlflow.log_param("llm_model", "openai/gpt-oss-20b:free")
        mlflow.log_param("evaluator_llm", "openai/gpt-oss-20b:free")
        mlflow.log_param("cache", "in-memory")
        # mlflow.log_param("retrieval_mode", "vector+reranker+metadata_filters")
        mlflow.log_param("prompt_version", "v2_grounded_prompt")

        mlflow.log_text(
            RAG_PROMPT.template,
            artifact_file="prompts/rag_prompt_template.txt",
        )

        result_dict = results.to_pandas().mean(numeric_only=True).to_dict()
        results_df = results.to_pandas()

        report_path = f"reports/ragas_{retrieval_mode}_results.csv"
        os.makedirs("reports", exist_ok=True)
        results_df.to_csv(report_path, index=False)

        mlflow.log_artifact(report_path, artifact_path="ragas_reports")
        for metric_name, metric_value in result_dict.items():
            mlflow.log_metric(metric_name, float(metric_value))

        print("\nRAGAS Results:")
        print(results)

        print("\nLogged to MLflow experiment: netflix-rag-evaluation")


if __name__ == "__main__":
    retrieval_modes = ["vector", "graph", "hybrid"]

    for mode in retrieval_modes:
        print(f"\nRunning RAGAS evaluation for mode: {mode}")

        ragas_result = run_ragas_evaluation(retrieval_mode=mode)
        log_to_mlflow(ragas_result, retrieval_mode=mode)

        print(f"Completed MLflow logging for mode: {mode}")