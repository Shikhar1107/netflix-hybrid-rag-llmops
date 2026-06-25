from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.graph.graph_retriever import graph_retrieve, find_title_by_name
from src.retrieval.retriever import (
    retrieve_documents as vector_retrieve_documents,
    extract_reference_title,
)
from src.retrieval.reranker import Reranker
from src.retrieval.router import route_retrieval

def remove_reference_title_noise(items: List[Dict], reference_title: Optional[str]) -> List[Dict]:
    if not reference_title:
        return items

    reference_lower = reference_title.lower().strip()
    reference_tokens = [
        token
        for token in reference_lower.split()
        if len(token) > 3
    ]

    cleaned_items = []

    for item in items:
        doc = item["document"]
        title = doc.metadata.get("title", "").lower().strip()

        if not title:
            continue

        # Remove exact reference title
        if title == reference_lower:
            continue

        # Remove noisy partial title matches like:
        # reference = "Stranger Things"
        # title = "THE STRANGER"
        if any(token in title for token in reference_tokens):
            continue

        cleaned_items.append(item)

    return cleaned_items

def graph_record_to_document(record: Dict) -> Document:
    title = record.get("title", "Unknown")
    content_type = record.get("type", "Unknown")
    year = record.get("year", "Unknown")
    rating = record.get("rating", "Unknown")
    duration = record.get("duration", "Unknown")
    description = record.get("description", "Unknown")

    genres = record.get("genres") or record.get("shared_genres") or []
    countries = record.get("countries") or []
    shared_genres = record.get("shared_genres") or []

    genre_text = ", ".join(genres) if isinstance(genres, list) else str(genres)
    country_text = ", ".join(countries) if isinstance(countries, list) else str(countries)
    shared_genre_text = ", ".join(shared_genres) if isinstance(shared_genres, list) else str(shared_genres)

    page_content = f"""
Title: {title}
Type: {content_type}
Genre: {genre_text}
Shared Genres: {shared_genre_text}
Country: {country_text}
Release Year: {year}
Rating: {rating}
Duration: {duration}
Description: {description}
Graph Reason: {record.get("graph_reason", "graph_match")}
""".strip()

    metadata = {
        "title": title,
        "type": content_type,
        "year": str(year),
        "rating": str(rating),
        "duration": str(duration),
        "genre": genre_text,
        "country": country_text,
        "description": description,
        "source": "graph",
        "graph_score": float(record.get("graph_score", 0)),
        "graph_reason": record.get("graph_reason", "graph_match"),
    }

    return Document(page_content=page_content, metadata=metadata)


def graph_results_to_rerank_items(records: List[Dict]) -> List[Dict]:
    items = []

    for record in records:
        doc = graph_record_to_document(record)

        items.append(
            {
                "score": float(record.get("graph_score", 0)),
                "document": doc,
                "retrieval_source": "graph",
            }
        )

    return items



def merge_retrieval_items(vector_items: List[Dict], graph_items: List[Dict]) -> List[Dict]:
    merged = {}

    for item in vector_items:
        doc = item["document"]
        title = doc.metadata.get("title", "").strip().lower()

        if not title:
            continue

        merged[title] = {
            "score": float(item.get("score", 0)),
            "document": doc,
            "retrieval_source": "vector",
            "vector_score": float(item.get("score", 0)),
            "graph_score": 0.0,
        }

    for item in graph_items:
        doc = item["document"]
        title = doc.metadata.get("title", "").strip().lower()

        if not title:
            continue

        graph_score = float(doc.metadata.get("graph_score", 0))

        if title in merged:
            existing = merged[title]
            existing_doc = existing["document"]

            existing_doc.metadata["source"] = "hybrid"
            existing_doc.metadata["graph_score"] = graph_score
            existing_doc.metadata["graph_reason"] = doc.metadata.get("graph_reason")
            existing_doc.metadata["retrieval_bonus"] = "matched_by_vector_and_graph"

            existing["retrieval_source"] = "hybrid"
            existing["graph_score"] = graph_score

            # Small bonus because the title appeared in both vector and graph retrieval.
            existing["score"] = existing["score"] + 1.0 + graph_score

        else:
            doc.metadata["source"] = "graph"
            merged[title] = {
                "score": graph_score,
                "document": doc,
                "retrieval_source": "graph",
                "vector_score": 0.0,
                "graph_score": graph_score,
            }

    return list(merged.values())


def rerank_items(query: str, items: List[Dict], top_k: int = 5) -> List[Dict]:
    if not items:
        return []

    documents = [item["document"] for item in items]

    reranker = Reranker()
    reranked = reranker.rerank(query=query, documents=documents, top_k=top_k)

    final_items = []

    for reranked_item in reranked:
        doc = reranked_item["document"]

        final_items.append(
            {
                "score": float(reranked_item["score"]),
                "document": doc,
                "retrieval_source": doc.metadata.get("source", "unknown"),
            }
        )

    return final_items


def resolve_reference_title(query: str) -> Optional[str]:
    reference_title = extract_reference_title(query)

    if not reference_title:
        return None

    graph_title = find_title_by_name(reference_title)

    if graph_title:
        return graph_title["title"]

    return reference_title


def retrieve_documents(
    query: str,
    retrieval_mode: str = "auto",
    initial_k: int = 40,
    final_k: int = 5,
) -> Dict:
    resolved_mode = route_retrieval(query, retrieval_mode)
    reference_title = resolve_reference_title(query)

    vector_items = []
    graph_items = []

    if resolved_mode in ["vector", "hybrid"]:
        vector_output = vector_retrieve_documents(
            query=query,
            initial_k=initial_k,
            final_k=10 if resolved_mode == "hybrid" else final_k,
        )
        vector_items = vector_output.get("documents", [])

        if not reference_title:
            reference_title = vector_output.get("reference_title")

    if resolved_mode in ["graph", "hybrid"]:
        graph_records = graph_retrieve(
            query=query,
            reference_title=reference_title,
            limit=10,
        )
        graph_items = graph_results_to_rerank_items(graph_records)
    
    vector_items = remove_reference_title_noise(vector_items, reference_title)
    graph_items = remove_reference_title_noise(graph_items, reference_title)

    if resolved_mode == "vector":
        final_docs = vector_items[:final_k]

    elif resolved_mode == "graph":
        final_docs = rerank_items(query, graph_items, top_k=final_k)

    else:
        merged_items = merge_retrieval_items(vector_items, graph_items)
        final_docs = rerank_items(query, merged_items, top_k=final_k)
    
    return {
        "query": query,
        "retrieval_mode": resolved_mode,
        "reference_title": reference_title,
        "documents": final_docs,
        "vector_count": len(vector_items),
        "graph_count": len(graph_items),
        "final_count": len(final_docs),
    }