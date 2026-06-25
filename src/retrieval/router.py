def route_retrieval(query: str, requested_mode: str = "auto") -> str:
    if requested_mode != "auto":
        return requested_mode

    query_lower = query.lower()

    graph_keywords = [
        "same actor",
        "same cast",
        "same director",
        "directed by",
        "acted in",
        "starring",
        "with actor",
        "shared genre",
        "same genre",
        "same country",
        "from same country",
    ]

    hybrid_keywords = [
        "like",
        "similar to",
        "recommend",
        "suggest",
        "shows like",
        "movies like",
    ]

    if any(keyword in query_lower for keyword in graph_keywords):
        return "graph"

    if any(keyword in query_lower for keyword in hybrid_keywords):
        return "hybrid"

    return "vector"