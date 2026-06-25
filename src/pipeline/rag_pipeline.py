import time

from src.generation.llm_chain import generate_answer
from src.cache.redis_cache import get_from_redis_cache, save_to_redis_cache, check_redis_connection


def run_rag_pipeline(question: str, retrieval_mode: str = "auto"):
    """
    Main reusable RAG pipeline function.

    Used by:
    - CLI testing
    - FastAPI
    - Telegram bot
    - RAGAS evaluation
    - MLflow experiments
    """

    if not question or not question.strip():
        return {
            "success": False,
            "error": "Question cannot be empty.",
            "question": question,
            "answer": None,
            "sources": [],
            "cache_hit": False,
        }

    clean_question = question.strip()
    start_time = time.time()

    use_redis_cache = check_redis_connection()
    cached_result = get_from_redis_cache(clean_question, retrieval_mode) if use_redis_cache else None

    if cached_result:
        latency = round(time.time() - start_time, 3)

        cached_result = cached_result.copy()
        cached_result["cache_hit"] = True
        cached_result["cache_type"] = "redis"
        cached_result["latency_seconds"] = latency
        cached_result["cached_original_timings"] = cached_result.get("timings", {})
        cached_result["timings"] = {
            "cache_lookup_seconds": latency
        }

        return cached_result

    try:
        result = generate_answer(question=clean_question,
        retrieval_mode=retrieval_mode,)
        latency = round(time.time() - start_time, 3)

        response = {
            "success": True,
            "error": None,
            "question": result["question"],
            "answer": result["answer"],
            "reference_title": result.get("reference_title"),
            "filters": result.get("filters", {}),
            "sources": result.get("sources", []),
            "latency_seconds": latency,
            "timings": result.get("timings", {}),
            "contexts": result.get("contexts", []),
            "cache_hit": False,
            "cache_type": "redis" if use_redis_cache else "none",
            "retrieval_mode": result.get("retrieval_mode", retrieval_mode),
            "vector_count": result.get("vector_count", 0),
            "graph_count": result.get("graph_count", 0),
            "final_count": result.get("final_count", 0),
        }

        if use_redis_cache:
            save_to_redis_cache(clean_question, response, retrieval_mode)

        return response

    except Exception as e:
        latency = round(time.time() - start_time, 3)

        return {
            "success": False,
            "error": str(e),
            "question": clean_question,
            "answer": None,
            "sources": [],
            "latency_seconds": latency,
            "cache_hit": False,
            "cache_type": "redis" if use_redis_cache else "none",
        }