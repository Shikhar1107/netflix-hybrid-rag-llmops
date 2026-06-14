from src.retrieval.retriever import retrieve_documents


def print_results(results):
    print(f"\nQuery: {results['query']}")
    print(f"Detected filters: {results['filters'] if results['filters'] else 'None'}")
    print(f"Reference title: {results['reference_title'] if results['reference_title'] else 'None'}")

    for index, item in enumerate(results["documents"], start=1):
        doc = item["document"]
        score = item["score"]

        print(f"\nResult {index}")
        print(f"Reranker Score: {score:.4f}")
        print(f"Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"Type: {doc.metadata.get('type', 'Unknown')}")
        print(f"Genre: {doc.metadata.get('genre', 'Unknown')}")
        print(f"Year: {doc.metadata.get('year', 'Unknown')}")
        print(f"Rating: {doc.metadata.get('rating', 'Unknown')}")
        print(f"Country: {doc.metadata.get('country', 'Unknown')}")
        print(f"Description: {doc.page_content.split('Description:')[-1].strip()}")


if __name__ == "__main__":
    results = retrieve_documents("thriller shows like Daredevil")
    print_results(results)