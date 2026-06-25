from src.graph.graph_retriever import graph_retrieve


def main():
    results = graph_retrieve(
        query="shows like Enola Holmes",
        reference_title="Enola Holmes",
        limit=10,
    )

    for item in results:
        print("\nTitle:", item.get("title"))
        print("Type:", item.get("type"))
        print("Year:", item.get("year"))
        print("Rating:", item.get("rating"))
        print("Shared genres:", item.get("shared_genres"))
        print("Graph score:", item.get("graph_score"))


if __name__ == "__main__":
    main()