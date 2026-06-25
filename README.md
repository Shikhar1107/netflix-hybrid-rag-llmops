# Netflix Hybrid RAG LLMOps

A production-style **Hybrid GraphRAG recommendation system** built on the Netflix Movies and TV Shows dataset. The system combines **semantic vector search**, **Neo4j graph-based retrieval**, **cross-encoder reranking**, **OpenRouter LLM generation**, **Redis caching**, **FastAPI serving**, and **LLMOps evaluation with RAGAS + MLflow**.

The goal of this project is to show how Retrieval-Augmented Generation can go beyond basic vector search by combining semantic similarity with structured graph relationships.

---

## Project Highlights

- Built a Hybrid GraphRAG API over **8,800+ Netflix titles**.
- Used **BAAI/bge-base-en-v1.5** embeddings with **ChromaDB** for semantic retrieval.
- Modeled Netflix metadata in **Neo4j** using title, genre, actor, director, country, and rating relationships.
- Implemented retrieval routing with support for `auto`, `vector`, `graph`, and `hybrid` modes.
- Added **cross-encoder reranking** using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Integrated **OpenRouter-compatible LLM generation** for grounded natural-language recommendations.
- Added **Redis caching with TTL** to reduce repeated-query latency to sub-second responses.
- Exposed the system through **FastAPI** with production-style health, cache, graph, and debug endpoints.
- Evaluated vector, graph, and hybrid retrieval strategies using **RAGAS** and tracked experiments in **MLflow**.
- Containerized the local stack with **Docker Compose** for FastAPI, Redis, and Neo4j.

---

## Project Overview

Most basic RAG systems rely only on vector similarity. This project implements a more advanced **Hybrid GraphRAG architecture** where user queries are handled using:

1. **Vector Retrieval**  
   Finds semantically similar titles using embedded title metadata, genres, cast, country, rating, duration, and descriptions.

2. **Graph Retrieval**  
   Uses Neo4j relationships between titles, actors, directors, genres, countries, and ratings.

3. **Hybrid Retrieval**  
   Combines ChromaDB vector candidates with Neo4j graph candidates, deduplicates them, and reranks final results.

4. **LLM Response Generation**  
   Produces concise recommendations grounded in retrieved Netflix context.

5. **LLMOps Layer**  
   Tracks retrieval mode, prompt version, RAGAS metrics, evaluation artifacts, and experiment runs in MLflow.

---

## Example Queries

```text
Suggest thriller shows similar to Dark
```

```text
Recommend Indian comedy movies
```

```text
Find kids movies about animals
```

```text
Suggest Korean horror shows
```

```text
Find shows similar to Stranger Things using both description similarity and graph relationships
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Dataset | Netflix Movies and TV Shows CSV |
| Language | Python |
| RAG Framework | LangChain |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Vector Database | ChromaDB |
| Graph Database | Neo4j |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM Provider | OpenRouter-compatible Chat API |
| API Backend | FastAPI |
| Caching | Redis |
| Evaluation | RAGAS |
| Experiment Tracking | MLflow |
| Containerization | Docker, Docker Compose |

### Planned Additions

| Area | Status |
|---|---|
| Telegram bot | Planned |
| Prometheus + Grafana monitoring | Planned |
| GitHub Actions CI/CD | Planned |
| Cloud deployment | Planned |

---

## Architecture

```text
User Query
   ↓
FastAPI Endpoint
   ↓
Retrieval Router
   ├── auto
   ├── vector
   ├── graph
   └── hybrid
   ↓
Hybrid Retrieval Layer
   ├── ChromaDB vector search
   └── Neo4j graph traversal
   ↓
Candidate Merge + Deduplication
   ↓
Cross-Encoder Reranking
   ↓
OpenRouter LLM Generation
   ↓
Redis Cache
   ↓
API Response
   ↓
RAGAS + MLflow Evaluation
```

---

## Core Features

### Vector RAG

The vector retrieval pipeline converts Netflix rows into LangChain documents and stores embeddings in ChromaDB.

Each Netflix title is represented using:

- Title
- Type
- Director
- Cast
- Genre
- Country
- Release year
- Rating
- Duration
- Description

This allows semantic search over both natural-language descriptions and structured metadata.

---

### Neo4j GraphRAG

The graph retrieval layer models Netflix content as a knowledge graph.

Implemented graph schema:

```text
(:Title)-[:HAS_GENRE]->(:Genre)
(:Title)-[:AVAILABLE_IN]->(:Country)
(:Title)-[:RATED_AS]->(:Rating)
(:Title)-[:DIRECTED_BY]->(:Director)
(:Actor)-[:ACTED_IN]->(:Title)
```

This enables relationship-aware retrieval, such as:

- Titles sharing genres with a reference title
- Titles featuring a specific actor
- Titles directed by a specific director
- Titles connected by country, rating, or genre

---

### Retrieval Modes

The API supports four retrieval modes:

| Mode | Behavior |
|---|---|
| `auto` | Automatically routes the query to vector, graph, or hybrid retrieval |
| `vector` | Uses only ChromaDB semantic retrieval |
| `graph` | Uses only Neo4j relationship retrieval |
| `hybrid` | Combines vector + graph candidates and reranks them |

Example request:

```json
{
  "question": "shows like Stranger Things",
  "retrieval_mode": "hybrid"
}
```

---

### Cross-Encoder Reranking

The system uses a cross-encoder reranker to improve final result quality.

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker compares the user query with each retrieved candidate and selects the most relevant titles before LLM generation.

---

### Redis Caching

Redis caches repeated queries using the query text and retrieval mode as part of the cache key.

This prevents collisions between:

```text
same question + vector mode
same question + graph mode
same question + hybrid mode
```

Cached responses return with:

```json
{
  "cache_hit": true,
  "cache_type": "redis"
}
```

---

### FastAPI Backend

The system is served through FastAPI.

Implemented endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Root API check |
| `/health` | GET | API and Redis health check |
| `/query` | POST | Main recommendation endpoint |
| `/query/debug` | POST | Debug endpoint returning retrieved contexts and retrieval metadata |
| `/cache/status` | GET | Redis cache status |
| `/graph/status` | GET | Neo4j graph statistics |

Swagger UI:

```text
http://localhost:8000/docs
```

---

### RAGAS Evaluation

RAGAS is used to evaluate the retrieval and generation pipeline using:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

The evaluation compares:

```text
vector-only retrieval
vs graph-only retrieval
vs hybrid GraphRAG retrieval
```

---

### MLflow Experiment Tracking

MLflow tracks:

- Retrieval mode
- Retrieval stack
- Prompt version
- Embedding model
- Vector database
- Graph database
- Reranker model
- LLM model
- RAGAS scores
- Prompt template artifact
- Query-level RAGAS report CSV artifacts

Run MLflow locally:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## RAGAS + MLflow Evaluation Results

The system was evaluated across three retrieval strategies: vector-only, graph-only, and hybrid GraphRAG.

| Retrieval Mode | Answer Relevancy | Faithfulness | Context Precision | Context Recall |
|---|---:|---:|---:|---:|
| Vector | 0.827 | 0.730 | 0.650 | 0.400 |
| Graph | 0.619 | 0.000 | 0.000 | 0.000 |
| Hybrid GraphRAG | 0.786 | 0.609 | 0.667 | 1.000 |

### Key Findings

- Vector retrieval produced the strongest answer relevancy and faithfulness for broad semantic recommendation queries.
- Hybrid GraphRAG improved context recall from `0.400` to `1.000`, showing stronger supporting-context coverage.
- Hybrid retrieval slightly improved context precision compared to vector-only retrieval.
- Graph-only retrieval underperformed on broad semantic queries, showing that Neo4j works best as a complementary relationship layer rather than a replacement for vector search.

---

## Project Structure

```text
netflix-rag-llmops/
├── data/
│   └── netflix_titles.csv
├── chroma_db/
│   └── ...
├── reports/
│   └── ragas_<retrieval_mode>_results.csv
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── router.py
│   │   └── reranker.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── neo4j_client.py
│   │   ├── graph_ingest.py
│   │   └── graph_retriever.py
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm_chain.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── memory_cache.py
│   │   └── redis_cache.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── test_dataset.py
│   │   └── ragas_eval.py
│   └── api/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   ├── run_pipeline.py
│   ├── test_api_request.py
│   └── test_graph_retriever.py
├── build_index.py
├── query_vector.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Main Files Explained

### `src/ingestion/loader.py`

Loads the Netflix CSV dataset into a Pandas DataFrame.

### `src/ingestion/chunker.py`

Converts each Netflix row into a LangChain `Document`. Each title becomes one retrievable document containing title, metadata, and description.

### `src/ingestion/embedder.py`

Loads the BGE embedding model using a cached function so the embedding model is initialized only once.

### `src/retrieval/vector_store.py`

Builds and loads the ChromaDB vector store using Netflix documents and BGE embeddings.

### `src/retrieval/retriever.py`

Implements vector retrieval logic including metadata-aware filtering, reference-title detection, manual filtering, and reranking.

### `src/retrieval/router.py`

Routes each query into `auto`, `vector`, `graph`, or `hybrid` retrieval mode.

### `src/retrieval/hybrid_retriever.py`

Combines vector results from ChromaDB with graph results from Neo4j, removes duplicate titles, filters noisy reference-title matches, and reranks merged candidates.

### `src/retrieval/reranker.py`

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank retrieved documents.

### `src/graph/neo4j_client.py`

Creates the Neo4j driver client and provides helper methods for read/write Cypher queries.

### `src/graph/graph_ingest.py`

Creates constraints and ingests Netflix titles, genres, countries, ratings, directors, and actors into Neo4j.

### `src/graph/graph_retriever.py`

Retrieves graph candidates from Neo4j using shared genres, actors, and directors.

### `src/generation/llm_chain.py`

Builds the final prompt, formats retrieved context, calls the OpenRouter-compatible chat model, and returns the final answer with sources and timings.

### `src/pipeline/rag_pipeline.py`

Main orchestration layer. Handles query validation, Redis cache lookup, RAG generation, latency tracking, and final response formatting.

### `src/cache/redis_cache.py`

Implements Redis-based cache get/set logic with TTL and retrieval-mode-aware cache keys.

### `src/cache/memory_cache.py`

Earlier in-memory cache implementation kept for local testing and reference.

### `src/evaluation/test_dataset.py`

Contains the test queries and ground-truth/reference expectations used for RAGAS evaluation.

### `src/evaluation/ragas_eval.py`

Runs RAGAS evaluation for vector, graph, and hybrid modes and logs metrics, prompt templates, and CSV artifacts to MLflow.

### `src/api/main.py`

FastAPI application exposing `/query`, `/query/debug`, `/health`, `/cache/status`, and `/graph/status` endpoints.

### `build_index.py`

Builds the ChromaDB vector index from the Netflix dataset.

### `query_vector.py`

Local debug script for testing vector retrieval from the command line.

### `tests/run_pipeline.py`

Runs end-to-end pipeline tests and verifies caching behavior.

### `tests/test_api_request.py`

Sends a test request to the FastAPI `/query` endpoint.

### `tests/test_graph_retriever.py`

Tests Neo4j graph retrieval independently before integrating it into Hybrid GraphRAG.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/netflix-rag-llmops.git
cd netflix-rag-llmops
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL_SECONDS=3600

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
```

---

## Dataset

Download the Netflix Movies and TV Shows dataset from Kaggle and place it at:

```text
data/netflix_titles.csv
```

Expected columns:

```text
show_id
type
title
director
cast
country
date_added
release_year
rating
duration
listed_in
description
```

---

## Running Locally

### 1. Build the Vector Index

```bash
python build_index.py
```

This will:

- Load the Netflix CSV
- Convert each row into a LangChain document
- Generate embeddings
- Store vectors in ChromaDB
- Run a sample retrieval query

### 2. Start Redis and Neo4j with Docker Compose

```bash
docker compose up --build
```

This starts:

```text
FastAPI API
Redis cache
Neo4j graph database
```

### 3. Ingest Graph Data into Neo4j

If Neo4j is empty, run:

```bash
python -m src.graph.graph_ingest
```

### 4. Run the API Manually

If not using Docker for the API, run:

```bash
uvicorn src.api.main:app --reload
```

API docs:

```text
http://localhost:8000/docs
```

---

## API Usage

### Main Query Endpoint

```http
POST /query
```

Request:

```json
{
  "question": "shows like Stranger Things",
  "retrieval_mode": "hybrid"
}
```

Response includes:

```json
{
  "success": true,
  "question": "shows like Stranger Things",
  "retrieval_mode": "hybrid",
  "answer": "...",
  "reference_title": "Stranger Things",
  "vector_count": 10,
  "graph_count": 10,
  "final_count": 5,
  "cache_hit": false,
  "cache_type": "redis",
  "sources": []
}
```

### Debug Query Endpoint

```http
POST /query/debug
```

Returns the full response including retrieved contexts. Useful for debugging, evaluation, and demos.

### Health Check

```http
GET /health
```

### Cache Status

```http
GET /cache/status
```

### Graph Status

```http
GET /graph/status
```

---

## Evaluation

Run RAGAS evaluation across retrieval modes:

```bash
python -m src.evaluation.ragas_eval
```

This evaluates:

```text
vector
graph
hybrid
```

and logs results to MLflow.

Run MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## Docker Compose

Start the stack:

```bash
docker compose up --build
```

Services included:

| Service | Purpose |
|---|---|
| `api` | FastAPI backend |
| `redis` | Query cache |
| `neo4j` | Graph database |

Useful URLs:

| Service | URL |
|---|---|
| FastAPI | `http://localhost:8000` |
| Swagger UI | `http://localhost:8000/docs` |
| Neo4j Browser | `http://localhost:7474` |

---

## Useful Neo4j Queries

Check graph counts:

```cypher
MATCH (n)
RETURN labels(n), count(n);
```

Visualize one title's relationships:

```cypher
MATCH (t:Title {title: "Stranger Things"})-[r]-(n)
RETURN t, r, n
LIMIT 50;
```

Find titles sharing genres with a reference title:

```cypher
MATCH path = (ref:Title {title: "Stranger Things"})-[:HAS_GENRE]->(:Genre)<-[:HAS_GENRE]-(other:Title)
WHERE other.title <> ref.title
RETURN path
LIMIT 50;
```

Case-insensitive title search:

```cypher
MATCH (t:Title)-[r]->(n)
WHERE toLower(t.title) CONTAINS toLower("stranger things")
RETURN t, r, n
LIMIT 50;
```

---

## Current Development Status

| Component | Status |
|---|---|
| Project setup | Completed |
| Netflix dataset loading | Completed |
| LangChain document creation | Completed |
| Vector embedding pipeline | Completed |
| ChromaDB vector store | Completed |
| Metadata-aware vector retrieval | Completed |
| Reference-title detection | Completed |
| Cross-encoder reranking | Completed |
| FastAPI backend | Completed |
| Redis caching | Completed |
| Neo4j graph schema and ingestion | Completed |
| Graph retriever | Completed |
| Hybrid GraphRAG retrieval | Completed |
| Auto / vector / graph / hybrid routing | Completed |
| RAGAS evaluation | Completed |
| MLflow experiment tracking | Completed |
| Docker Compose setup | Completed |
| Telegram bot | Planned |
| Prometheus + Grafana monitoring | Planned |
| GitHub Actions workflow | Planned |
| Cloud deployment | Planned |

---

## Future Improvements

- Add Telegram bot once Telegram access is available.
- Add Prometheus + Grafana for live API metrics.
- Add GitHub Actions for tests and Docker build validation.
- Add streaming responses from FastAPI.
- Add LangGraph-based agentic routing.
- Add user preference memory.
- Add multilingual recommendation support.
- Deploy API to cloud infrastructure.

---

## Resume Summary

Built a production-style **Hybrid GraphRAG recommendation API** over 8,800+ Netflix titles using LangChain, ChromaDB, Neo4j, BGE embeddings, cross-encoder reranking, FastAPI, Redis, RAGAS, MLflow, OpenRouter, and Docker Compose.

### Resume Bullet

```text
Developed a production-style Hybrid GraphRAG API for Netflix content intelligence using ChromaDB semantic search, Neo4j relationship traversal, cross-encoder reranking, Redis caching, and OpenRouter LLM generation; evaluated vector vs graph vs hybrid retrieval in MLflow using RAGAS, with hybrid retrieval improving context recall from 0.40 to 1.00.
```

---

## License

This project is for educational and portfolio use.
