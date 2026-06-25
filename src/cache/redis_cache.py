import os
import json
import hashlib
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST',"localhost")
REDIS_PORT = os.getenv("REDIS_PORT",6379)
REDIS_DB = os.getenv('REDIS_DB', 0)
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', 3600))

redis_client = redis.Redis(
    host = REDIS_HOST,
    port=REDIS_PORT,
    db= REDIS_DB,
    decode_responses=True
)

def generate_cache_key(question: str, retrieval_mode: str = "auto") -> str:
    normalized_question = question.strip().lower()
    normalized_mode = retrieval_mode.strip().lower()

    raw_key = f"{normalized_mode}:{normalized_question}"
    hashed_key = hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    return f"netflix_rag:{hashed_key}"

def get_from_redis_cache(question: str, retrieval_mode: str = "auto"):
    cache_key = generate_cache_key(question, retrieval_mode)

    cached_data = redis_client.get(cache_key)

    if not cached_data:
        return None
    
    return json.loads(cached_data)

def save_to_redis_cache(question: str, data: dict, retrieval_mode: str = "auto"):
    cache_key = generate_cache_key(question, retrieval_mode)

    redis_client.setex(
        cache_key,
        CACHE_TTL_SECONDS,
        json.dumps(data),
    )

def check_redis_connection():
    try:
        redis_client.ping()
        return True
    except Exception:
        return False