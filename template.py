import os
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/ingestion/loader.py",
    "src/ingestion/chunker.py",
    "src/ingestion/embedder.py",
    "src/retrieval/vector_store.py",
    "src/retrieval/reranker.py",
    "src/generation/llm_chain.py",
    "src/evaluation/ragas_eval.py",
    "src/monitoring/latency.py",
    "src/monitoring/cost_tracker.py",
    "src/cache/redis_cache.py",
    "src/api/main.py",
    "mlflow/",
    "grafana/dashboard.json",
    "tests/test_queries.py",
    "docker-compose.yml",
    "Dockerfile",
    ]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating file directory {filedir} for file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")