from src.retrieval.vector_store import load_vector_store
from src.retrieval.reranker import Reranker

def main():
    query = "thriller shows like Dark"

    print("Loading Chroma DB vector stores...")
    vector_store = load_vector_store()
    print("Performing similarity search...")
    results = vector_store.similarity_search(query, k=10)
    print("Performing reranking...")
    reranker = Reranker()
    reranked_results = reranker.rerank(query, results, top_k=5)
    print("Reranked results:")
    print("\nFinal reranked results:")

    for index, item in enumerate(reranked_results, start=1):
        doc = item["document"]
        score = item["score"]

        print(f"\nResult {index}")
        print(f"Reranker Score: {score:.4f}")
        print(f"Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"Type: {doc.metadata.get('type', 'Unknown')}")
        print(f"Genre: {doc.metadata.get('genre', 'Unknown')}")
        print(f"Year: {doc.metadata.get('year', 'Unknown')}")
        print(f"Rating: {doc.metadata.get('rating', 'Unknown')}")
        # print(f"Country: {doc.metadata.get('country', 'Unknown')}")
        print(f"Preview: {doc.page_content[:300]}...")


if __name__ == "__main__":
    main()