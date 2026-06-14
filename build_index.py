from src.ingestion.loader import load_netflix_data
from src.ingestion.chunker import create_documents
from src.retrieval.vector_store import build_vector_store

def main():

    print("Loading netflix dataset...")
    df = load_netflix_data("data/netflix_titles.csv")

    print("Creating Langchain documents...")
    documents = create_documents(df)
    print(f"Created {len(documents)} documents")

    print("Building ChromaDB vector store...")
    vector_store = build_vector_store(documents)
    print("Vector store built successfully.")

    print("\nTesting retrieval...")
    results = vector_store.similarity_search("thriller movies like Dark", k=5)

    for index,doc in enumerate(results,start = 1):
        print(f"Result {index}:")
        print(f"Title: {doc.metadata['title']}")
        print(f"Type: {doc.metadata['genre']}")
        print(f"Year: {doc.metadata['year']}")
        print(f"Rating: {doc.metadata['rating']}")
        print(f"Description: {doc.page_content[:200]}...")
        print("---")

if __name__ == "__main__":
    main()