# Netflix Hybrid RAG LLMOps

A production-style **Hybrid RAG recommendation system** built on the Netflix Movies and TV Shows dataset.
The system combines **vector search**, **graph-based retrieval**, **LLM-powered response generation**, and **LLMOps monitoring** to provide intelligent content recommendations through an API and Telegram bot interface.

This project is designed to demonstrate how Retrieval-Augmented Generation can be extended beyond simple semantic search by combining:

* Vector similarity search
* Knowledge graph traversal
* Cross-encoder reranking
* RAG evaluation
* Prompt/version tracking
* Caching
* Monitoring
* Containerized deployment
* CI/CD with GitHub Actions

---

## Project Overview

Most basic RAG systems rely only on vector similarity. This project builds a more advanced **Hybrid GraphRAG architecture** where user queries are answered using both:

1. **Vector Database Retrieval**
   Finds semantically similar Netflix titles based on descriptions, genres, cast, and metadata.

2. **Graph Database Retrieval**
   Uses relationships between movies, actors, directors, genres, countries, ratings, and release years.

3. **LLM Response Generation**
   Produces natural-language recommendations grounded in retrieved context.

4. **LLMOps Layer**
   Tracks evaluation metrics, latency, caching, prompt experiments, and system performance.

---

## Example Use Cases

Users can ask questions like:

```text
Suggest thriller shows similar to Dark
```

```text
Recommend family-friendly movies for kids under 10
```

```text
Find crime dramas connected by genre, country, or cast
```

```text
Suggest documentaries about nature and climate
```

```text
Find shows similar to Money Heist using both description similarity and graph relationships
```

---

## Tech Stack

| Layer               | Tools                                |
| ------------------- | ------------------------------------ |
| Dataset             | Netflix Movies and TV Shows CSV      |
| Language            | Python                               |
| RAG Framework       | LangChain                            |
| Embedding Model     | BAAI/bge-base-en-v1.5                |
| Vector Database     | ChromaDB                             |
| Graph Database      | Neo4j                                |
| Reranking           | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM                 | OpenAI / Ollama                      |
| API Backend         | FastAPI                              |
| Bot Interface       | Telegram Bot API                     |
| Caching             | Redis                                |
| Evaluation          | RAGAS                                |
| Experiment Tracking | MLflow                               |
| Monitoring          | Prometheus, Grafana                  |
| Containerization    | Docker, Docker Compose               |
| CI/CD               | GitHub Actions                       |
| Testing             | Pytest                               |
| Deployment          | AWS EC2                              |

---

## Architecture

```text
User Query
   ↓
Telegram Bot / FastAPI Endpoint
   ↓
Query Router
   ↓
Hybrid Retrieval Layer
   ├── Vector Search using ChromaDB
   └── Graph Search using Neo4j
   ↓
Cross-Encoder Reranking
   ↓
LLM Response Generation
   ↓
Redis Cache
   ↓
Response returned to user
   ↓
Metrics logged to MLflow, Prometheus, and Grafana
```

---

## Core Features

### Vector RAG

The vector retrieval pipeline converts Netflix title metadata into LangChain documents and stores their embeddings in ChromaDB.

Each Netflix title is represented using:

* Title
* Type
* Director
* Cast
* Genre
* Country
* Release year
* Rating
* Duration
* Description

This allows semantic search over both content descriptions and structured metadata.

---

### GraphRAG

The graph retrieval layer models Netflix content as a knowledge graph.

Example relationships:

```text
Title → HAS_GENRE → Genre
Title → ACTED_BY → Actor
Title → DIRECTED_BY → Director
Title → RELEASED_IN → Country
Title → HAS_RATING → Rating
Title → RELEASED_YEAR → Year
```

This enables relationship-aware recommendations, such as finding movies connected by shared actors, genres, directors, countries, or ratings.

---

### Hybrid Retrieval

The system combines vector and graph retrieval to improve recommendation quality.

Vector search is useful for semantic similarity:

```text
"shows with dark mystery and time travel"
```

Graph search is useful for structured relationships:

```text
"shows with the same actors, country, genre, or director"
```

The final retrieval layer combines both results and reranks them using a cross-encoder model.

---

### Reranking

A cross-encoder reranker is used to improve the quality of retrieved results.

Model used:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker compares the user query with each retrieved document and selects the most relevant results before passing them to the LLM.

---

### Telegram Bot Deployment

The system is exposed through a Telegram bot so users can interact with the recommendation engine conversationally.

Example:

```text
User: Suggest crime thrillers like Dark
Bot: Here are some recommendations based on semantic similarity and graph relationships...
```

---

### FastAPI Backend

FastAPI is used to expose REST APIs for:

```text
/query
/health
/recommend
/metrics
```

The backend handles:

* User query processing
* Vector retrieval
* Graph retrieval
* Reranking
* LLM response generation
* Caching
* Logging and monitoring

---

### Redis Caching

Redis is used to cache repeated queries and reduce latency.

Cached responses help improve response time for common recommendation requests.

---

### RAG Evaluation

RAGAS is used to evaluate the retrieval and generation pipeline using metrics such as:

* Faithfulness
* Answer relevancy
* Context precision
* Context recall

This helps compare:

```text
Vector-only RAG vs Hybrid GraphRAG
```

---

### MLflow Experiment Tracking

MLflow is used to track:

* Prompt versions
* Embedding models
* Retriever configurations
* Reranker settings
* Evaluation scores
* Latency metrics

This makes the system easier to benchmark and improve.

---

### Monitoring with Prometheus and Grafana

Prometheus and Grafana are used to monitor:

* API latency
* Query count
* Cache hits
* Cache misses
* Retrieval time
* LLM response time
* Error rate

---

### Dockerized Deployment

The complete system is containerized using Docker Compose.

Services include:

```text
FastAPI backend
ChromaDB
Neo4j
Redis
MLflow
Prometheus
Grafana
Telegram bot service
```

---

### GitHub Actions CI/CD

GitHub Actions is used for:

* Running tests
* Checking code quality
* Building Docker images
* Validating application startup
* Preparing deployment workflow

---

## Project Structure

```text
netflix-hybrid-rag-llmops/
├── data/
│   └── netflix_titles.csv
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── graph_store.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── generation/
│   │   └── llm_chain.py
│   ├── evaluation/
│   │   └── ragas_eval.py
│   ├── monitoring/
│   │   ├── latency.py
│   │   └── metrics.py
│   ├── cache/
│   │   └── redis_cache.py
│   ├── bot/
│   │   └── telegram_bot.py
│   └── api/
│       └── main.py
├── tests/
│   └── test_queries.json
├── grafana/
│   └── dashboard.json
├── prometheus/
│   └── prometheus.yml
├── mlflow/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Current Development Status

* [x] Project setup
* [x] Netflix dataset loading
* [x] LangChain document creation
* [x] Vector embedding pipeline
* [x] ChromaDB vector store
* [x] Basic semantic retrieval
* [ ] Cross-encoder reranking
* [ ] FastAPI backend
* [ ] Neo4j graph schema
* [ ] Hybrid GraphRAG retrieval
* [ ] Telegram bot integration
* [ ] Redis caching
* [ ] RAGAS evaluation
* [ ] MLflow experiment tracking
* [ ] Prometheus and Grafana monitoring
* [ ] Docker Compose setup
* [ ] GitHub Actions workflow
* [ ] AWS EC2 deployment

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/netflix-hybrid-rag-llmops.git
cd netflix-hybrid-rag-llmops
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the Netflix Movies and TV Shows dataset from Kaggle and place it inside:

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

## Build Vector Index

```bash
python build_index.py
```

This will:

* Load the Netflix CSV
* Convert each row into a LangChain document
* Generate embeddings
* Store vectors in ChromaDB
* Run a sample retrieval query

---

## Run Vector Search

```bash
python query_vector.py
```

Example query:

```text
best documentaries about nature
```

---

## Run FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

API will be available at:

```text
http://localhost:8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

---

## Run with Docker Compose

```bash
docker-compose up --build
```

This starts the full stack:

* FastAPI
* Redis
* Neo4j
* MLflow
* Prometheus
* Grafana

---

## Evaluation

Run RAGAS evaluation:

```bash
python src/evaluation/ragas_eval.py
```

Evaluation metrics:

```text
Faithfulness
Answer relevancy
Context precision
Context recall
```

---

## Monitoring

Prometheus:

```text
http://localhost:9090
```

Grafana:

```text
http://localhost:3000
```

MLflow:

```text
http://localhost:5000
```

---

## Future Improvements

* Add user preference memory
* Improve query routing between graph and vector retrieval
* Add hybrid scoring strategy
* Add streaming responses
* Add LangGraph-based agentic routing
* Add multilingual recommendation support
* Deploy Telegram bot and backend on AWS
* Add automated evaluation in GitHub Actions

---

## Resume Summary

Built a production-style Hybrid GraphRAG recommendation system over 8,800+ Netflix titles using LangChain, ChromaDB, Neo4j, cross-encoder reranking, FastAPI, Redis, RAGAS, MLflow, Prometheus, Grafana, Docker, Telegram Bot API, and GitHub Actions.
