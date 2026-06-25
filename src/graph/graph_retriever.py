from typing import Dict, List, Optional

from src.graph.neo4j_client import Neo4jClient


def find_title_by_name(title: str) -> Optional[Dict]:
    client = Neo4jClient()

    try:
        # 1. Exact match first
        exact_query = """
        MATCH (t:Title)
        WHERE toLower(t.title) = toLower($title)
        RETURN t.title AS title,
               t.type AS type,
               t.release_year AS year,
               t.rating AS rating,
               t.duration AS duration,
               t.description AS description
        LIMIT 1
        """

        records = client.execute_read(exact_query, {"title": title})

        if records:
            return dict(records[0])

        # 2. Fallback partial match only if exact match not found
        partial_query = """
        MATCH (t:Title)
        WHERE toLower(t.title) CONTAINS toLower($title)
        RETURN t.title AS title,
               t.type AS type,
               t.release_year AS year,
               t.rating AS rating,
               t.duration AS duration,
               t.description AS description
        ORDER BY size(t.title) ASC
        LIMIT 1
        """

        records = client.execute_read(partial_query, {"title": title})

        if not records:
            return None

        return dict(records[0])

    finally:
        client.close()

def retrieve_by_shared_genres(reference_title: str, limit: int = 10) -> List[Dict]:
    query = """
    MATCH (ref:Title)
    WHERE toLower(ref.title) = toLower($reference_title)

    MATCH (ref)-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(other:Title)
    WHERE other.title <> ref.title

    OPTIONAL MATCH (other)-[:AVAILABLE_IN]->(c:Country)
    OPTIONAL MATCH (other)-[:RATED_AS]->(r:Rating)
    OPTIONAL MATCH (other)-[:HAS_GENRE]->(og:Genre)

    RETURN other.title AS title,
           other.type AS type,
           other.release_year AS year,
           other.rating AS rating,
           other.duration AS duration,
           other.description AS description,
           collect(DISTINCT og.name) AS genres,
           collect(DISTINCT g.name) AS shared_genres,
           collect(DISTINCT c.name) AS countries,
           collect(DISTINCT r.name) AS ratings,
           count(DISTINCT g) AS graph_score,
           'shared_genres' AS graph_reason
    ORDER BY graph_score DESC
    LIMIT $limit
    """

    client = Neo4jClient()

    try:
        records = client.execute_read(
            query,
            {
                "reference_title": reference_title,
                "limit": limit,
            },
        )
        return [dict(record) for record in records]

    finally:
        client.close()


def retrieve_by_actor(actor_name: str, limit: int = 10) -> List[Dict]:
    query = """
    MATCH (a:Actor)-[:ACTED_IN]->(t:Title)
    WHERE toLower(a.name) CONTAINS toLower($actor_name)

    OPTIONAL MATCH (t)-[:HAS_GENRE]->(g:Genre)
    OPTIONAL MATCH (t)-[:AVAILABLE_IN]->(c:Country)
    OPTIONAL MATCH (t)-[:RATED_AS]->(r:Rating)

    RETURN t.title AS title,
           t.type AS type,
           t.release_year AS year,
           t.rating AS rating,
           t.duration AS duration,
           t.description AS description,
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT c.name) AS countries,
           collect(DISTINCT r.name) AS ratings,
           a.name AS matched_actor,
           1 AS graph_score,
           'actor_match' AS graph_reason
    LIMIT $limit
    """

    client = Neo4jClient()

    try:
        records = client.execute_read(
            query,
            {
                "actor_name": actor_name,
                "limit": limit,
            },
        )
        return [dict(record) for record in records]

    finally:
        client.close()


def retrieve_by_director(director_name: str, limit: int = 10) -> List[Dict]:
    query = """
    MATCH (t:Title)-[:DIRECTED_BY]->(d:Director)
    WHERE toLower(d.name) CONTAINS toLower($director_name)

    OPTIONAL MATCH (t)-[:HAS_GENRE]->(g:Genre)
    OPTIONAL MATCH (t)-[:AVAILABLE_IN]->(c:Country)
    OPTIONAL MATCH (t)-[:RATED_AS]->(r:Rating)

    RETURN t.title AS title,
           t.type AS type,
           t.release_year AS year,
           t.rating AS rating,
           t.duration AS duration,
           t.description AS description,
           collect(DISTINCT g.name) AS genres,
           collect(DISTINCT c.name) AS countries,
           collect(DISTINCT r.name) AS ratings,
           d.name AS matched_director,
           1 AS graph_score,
           'director_match' AS graph_reason
    LIMIT $limit
    """

    client = Neo4jClient()

    try:
        records = client.execute_read(
            query,
            {
                "director_name": director_name,
                "limit": limit,
            },
        )
        return [dict(record) for record in records]

    finally:
        client.close()


def graph_retrieve(
    query: str,
    reference_title: Optional[str] = None,
    limit: int = 10,
) -> List[Dict]:
    query_lower = query.lower()

    if reference_title:
        return retrieve_by_shared_genres(reference_title, limit=limit)

    if "directed by" in query_lower:
        possible_name = query_lower.split("directed by", 1)[1].strip()
        if possible_name:
            return retrieve_by_director(possible_name, limit=limit)

    if "with" in query_lower:
        possible_name = query_lower.split("with", 1)[1].strip()
        if possible_name:
            return retrieve_by_actor(possible_name, limit=limit)

    return []