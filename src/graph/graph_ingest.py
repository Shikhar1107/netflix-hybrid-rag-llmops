import pandas as pd

from src.graph.neo4j_client import Neo4jClient
from src.ingestion.loader import load_netflix_data

def safe_value(value):
    if pd.isna(value):
        return "Unknown"
    return str(value).strip()

def split_values(value):
    value = safe_value(value)
    if value == "Unknown":
        return []
    
    return [item.strip() for item in value.split(",") if item.strip()]

def create_constraints(client: Neo4jClient):
    queries = [
        "CREATE CONSTRAINT title_name IF NOT EXISTS FOR (t:Title) REQUIRE t.title IS UNIQUE",
        "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
        "CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT actor_name IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT director_name IF NOT EXISTS FOR (d:Director) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT rating_name IF NOT EXISTS FOR (r:Rating) REQUIRE r.name IS UNIQUE",
    ]

    for query in queries:
        client.execute_write(query)

def clear_graph(client:Neo4jClient):
    client.execute_write("MATCH (n) DETACH DELETE n")

def ingest_title(tx, row):
    title = safe_value(row["title"])
    content_type = safe_value(row["type"])
    release_year = safe_value(row["release_year"])
    rating = safe_value(row["rating"])
    duration = safe_value(row["duration"])
    description = safe_value(row["description"])

    tx.run(
        """
        MERGE (t:Title {title: $title})
        SET t.type = $type,
            t.release_year = $release_year,
            t.rating = $rating,
            t.duration = $duration,
            t.description = $description
        """,
        {
            "title": title,
            "type": content_type,
            "release_year": release_year,
            "rating": rating,
            "duration": duration,
            "description": description,
        },
    )

    genres = split_values(row['listed_in'])
    countries = split_values(row['country'])
    cast_members = split_values(row['cast'])
    directors = split_values(row['director'])

    for genre in genres:
        tx.run(
            """
            MATCH (t:Title {title: $title})
            MERGE (g:Genre {name: $genre})
            MERGE (t)-[:HAS_GENRE]->(g)
            """,
            {"title":title,"genre":genre}
        )
    for country in countries:
        tx.run(
            """
            MATCH (t:Title {title: $title})
            MERGE (c:Country {name: $country})
            MERGE (t)-[:AVAILABLE_IN]->(c)
            """,
            {"title":title,"country":country}
        )

    if rating != "Unknown":
        tx.run(
            """
            MATCH (t:Title {title: $title})
            MERGE (r:Rating {name: $rating})
            MERGE (t)-[:RATED_AS]->(r)
            """,
            {"title":title,"rating":rating}
        )

    for actor in cast_members:
        tx.run(
            """
            MATCH (t:Title {title: $title})
            MERGE (a:Actor {name: $actor})
            MERGE (a)-[:ACTED_IN]->(t)
            """,
            {"title":title,"actor":actor}
        )
    for director in directors:
        tx.run(
            """
            MATCH (t:Title {title: $title})
            MERGE (d:Director {name: $director})
            MERGE (t)-[:DIRECTED_BY]->(d)
            """,
            {"title":title,"director":director}
        )

def ingest_graph(limit=None, clear_existing=False):
    df = load_netflix_data("data/netflix_titles.csv")

    if limit:
        df = df.head(limit)
    
    client = Neo4jClient()

    try:
        if clear_existing:
            print("Clearing existing graph...")
            clear_graph(client)
        
        print("Creating constraints...")
        create_constraints(client)

        print(f"Ingesting {len(df)} title into Neo4j...")

        with client.driver.session() as session:
            for index, row in df.iterrows():
                session.execute_write(ingest_title, row)

                if (index+1) % 500 == 0:
                    print(f"Ingested {index + 1} titles")
        
        print("Graph Ingestion completed.")
    finally:
        client.close()

if __name__=='__main__':
    ingest_graph(clear_existing=True)