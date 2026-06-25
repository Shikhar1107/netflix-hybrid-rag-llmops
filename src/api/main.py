from typing import Any, Dict, List, Optional, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.pipeline.rag_pipeline import run_rag_pipeline
from src.cache.redis_cache import check_redis_connection

app = FastAPI(
    title = "Netflix Content Intelligence API",
    description = "Hybrid-ready RAG API for netlfix ontent recommendations using vector retrieval, reranking, OpenRouter LLM, Redis cache, RAGAS, and MLflow.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        description="Natural Language recommendation query.",
        examples = ["thriller shows like Dark"]
    )
    retrieval_mode: Optional[Literal["auto", "vector", "graph", "hybrid"]] = Field(
        default="auto",
        description="Retrieval strategy: auto, vector, graph, or hybrid.",
    )

class SourceItem(BaseModel):
    title: str
    type: str
    year: str
    rating: str
    country: str
    genre: str
    score: float

class QueryResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    question: str
    answer: Optional[str] = None
    reference_title: Optional[str] = None
    filters: Dict[str, Any] = {}
    sources: List[SourceItem] = []
    latency_seconds: Optional[float] = None
    timings: Dict[str, Any] = {}
    cache_hit: bool = False
    cache_type : str = "none"
    retrieval_mode: Optional[str] = None
    vector_count: int = 0
    graph_count: int = 0
    final_count: int = 0

@app.get("/")
def root():
    return {
        "message": "Netflix Content Intelligence API is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "redis_connected": check_redis_connection()
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = run_rag_pipeline(request.question, retrieval_mode=request.retrieval_mode,)

    # Avoid sending full context in normal API response
    result.pop("context", None)

    return result

@app.post("/query/debug")
def query_rag_debug(request: QueryRequest):
    result = run_rag_pipeline(
        question=request.question,
        retrieval_mode=request.retrieval_mode,
    )

    return result

@app.get("/graph/status")
def graph_status():
    from src.graph.neo4j_client import Neo4jClient

    client = Neo4jClient()

    try:
        title_count = client.execute_read(
            "MATCH (t:Title) RETURN count(t) AS count"
        )[0]["count"]

        genre_count = client.execute_read(
            "MATCH (g:Genre) RETURN count(g) AS count"
        )[0]["count"]

        actor_count = client.execute_read(
            "MATCH (a:Actor) RETURN count(a) AS count"
        )[0]["count"]

        relationship_count = client.execute_read(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        )[0]["count"]

        return {
            "neo4j_connected": True,
            "titles": title_count,
            "genres": genre_count,
            "actors": actor_count,
            "relationships": relationship_count,
        }

    except Exception as e:
        return {
            "neo4j_connected": False,
            "error": str(e),
        }

    finally:
        client.close()

@app.get("/cache/status")
def cache_status():
    return {
        "redis_connected": check_redis_connection(),
        "cache_type": "redis",
    }