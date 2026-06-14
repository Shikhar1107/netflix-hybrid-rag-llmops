from src.retrieval.vector_store import load_vector_store
from src.retrieval.reranker import Reranker


def detect_metadata_filters(query: str):
    query_lower = query.lower()
    filters = {}

    movie_words = ["movie", "movies", "film", "films"]
    show_words = ["show", "shows", "series", "tv show", "tv series"]

    if any(word in query_lower for word in movie_words):
        filters["type"] = "Movie"

    if any(word in query_lower for word in show_words):
        filters["type"] = "TV Show"

    country_map = {
        "india": "India",
        "indian": "India",
        "korea": "South Korea",
        "korean": "South Korea",
        "japan": "Japan",
        "japanese": "Japan",
        "spain": "Spain",
        "spanish": "Spain",
        "germany": "Germany",
        "german": "Germany",
        "uk": "United Kingdom",
        "british": "United Kingdom",
        "america": "United States",
        "american": "United States",
        "usa": "United States",
    }

    for keyword, country in country_map.items():
        if keyword in query_lower:
            filters["country"] = country
            break

    kids_words = ["kids", "children", "child", "family", "under 10", "for 10 year old"]
    mature_words = ["mature", "adult", "18+", "violent"]

    if any(word in query_lower for word in kids_words):
        filters["rating"] = ["TV-Y", "TV-Y7", "G", "PG"]

    if any(word in query_lower for word in mature_words):
        filters["rating"] = ["TV-MA", "R"]

    if any(word in query_lower for word in ["recent", "latest", "new", "newly released"]):
        filters["year_gte"] = 2018

    if any(word in query_lower for word in ["classic", "old", "older"]):
        filters["year_lte"] = 2010

    return filters


def extract_reference_title(query: str):
    query_lower = query.lower()

    patterns = [
        "like ",
        "similar to ",
        "similar shows to ",
        "similar movies to ",
        "recommend shows like ",
        "recommend movies like ",
        "shows like ",
        "movies like ",
    ]

    for pattern in patterns:
        if pattern in query_lower:
            title_part = query_lower.split(pattern, 1)[1].strip()
            return title_part

    return None

def find_reference_document(vector_store, reference_title: str):
    if not reference_title:
        return None

    results = vector_store.similarity_search(
        reference_title,
        k=10,
    )

    for doc in results:
        title = doc.metadata.get("title", "").lower()

        if title == reference_title.lower():
            return doc

    return None

def build_recommendation_query(reference_doc):
    if not reference_doc:
        return None

    metadata = reference_doc.metadata

    title = metadata.get("title", "")
    content_type = metadata.get("type", "")
    genre = metadata.get("genre", "")
    country = metadata.get("country", "")
    rating = metadata.get("rating", "")

    return f"""
    Find content similar to this Netflix title.

    Reference Title: {title}
    Type: {content_type}
    Genre: {genre}
    Country: {country}
    Rating: {rating}

    Use the genre, mood, theme, country, and description.
    Do not match only by title words.
    Description:
    {reference_doc.page_content}
        """.strip()

def remove_reference_title_matches(reference_title: str, documents):
    if not reference_title:
        return documents

    reference_title_lower = reference_title.lower()
    filtered_docs = []

    for doc in documents:
        title = doc.metadata.get("title", "").lower()

        # Remove exact title
        if title == reference_title_lower:
            continue

        # Remove titles that contain the reference title word
        if reference_title_lower in title:
            continue

        filtered_docs.append(doc)

    return filtered_docs

def apply_manual_filters(documents, filters):
    if not filters:
        return documents

    filtered_docs = []

    for doc in documents:
        metadata = doc.metadata

        doc_type = metadata.get("type", "")
        country = metadata.get("country", "")
        rating = metadata.get("rating", "")
        genre = metadata.get("genre", "")
        year = metadata.get("year", "")

        keep = True

        if "type" in filters and doc_type != filters["type"]:
            keep = False

        if "country" in filters:
            if filters["country"].lower() not in country.lower():
                keep = False

        if "rating" in filters:
            if rating not in filters["rating"]:
                keep = False

        if "genre" in filters:
            if filters["genre"].lower() not in genre.lower():
                keep = False

        if "year_gte" in filters:
            try:
                if int(year) < filters["year_gte"]:
                    keep = False
            except ValueError:
                keep = False

        if "year_lte" in filters:
            try:
                if int(year) > filters["year_lte"]:
                    keep = False
            except ValueError:
                keep = False

        if keep:
            filtered_docs.append(doc)

    return filtered_docs


def retrieve_documents(query: str, initial_k: int = 50, final_k: int = 5):
    vector_store = load_vector_store()

    filters = detect_metadata_filters(query)

    reference_title = extract_reference_title(query)
    reference_doc = find_reference_document(vector_store, reference_title)

    if reference_doc:
        print(f"Detected reference title: {reference_doc.metadata.get('title')}")
        search_query = build_recommendation_query(reference_doc)
    else:
        search_query = query

    initial_results = vector_store.similarity_search(
        search_query,
        k=initial_k,
    )

    filtered_results = apply_manual_filters(
        documents=initial_results,
        filters=filters,
    )

    if reference_doc:
        filtered_results = remove_reference_title_matches(
            reference_title=reference_doc.metadata.get("title", ""),
            documents=filtered_results,
        )

    if len(filtered_results) < final_k:
        print("Too few documents matched filters. Continuing with available filtered results.")


    reranker = Reranker()

    reranked_results = reranker.rerank(
        query=search_query,
        documents=filtered_results,
        top_k=final_k,
    )
    print( {"query": query,
        "filters": filters,
        "reference_title": reference_doc.metadata.get("title") if reference_doc else None,
        "documents": reranked_results,
    })
    return {
        "query": query,
        "filters": filters,
        "reference_title": reference_doc.metadata.get("title") if reference_doc else None,
        "documents": reranked_results,
    }
