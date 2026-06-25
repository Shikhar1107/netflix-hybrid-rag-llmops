import time
import hashlib


_CACHE = {}


def generate_cache_key(question: str) -> str:
    normalized_question = question.strip().lower()
    return hashlib.md5(normalized_question.encode("utf-8")).hexdigest()


def get_from_cache(question: str):
    cache_key = generate_cache_key(question)

    cached_item = _CACHE.get(cache_key)

    if not cached_item:
        return None

    return cached_item["data"]


def save_to_cache(question: str, data: dict):
    cache_key = generate_cache_key(question)

    _CACHE[cache_key] = {
        "data": data,
        "created_at": time.time(),
    }


def clear_cache():
    _CACHE.clear()


def cache_size():
    return len(_CACHE)