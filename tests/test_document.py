from src.ingestion.loader import load_netflix_data
from src.ingestion.chunker import create_documents

df = load_netflix_data()
documents = create_documents(df)

print(f"Created {len(documents)} documents.")

print("Sample document: ")
print(documents[0])

print(f"Document Page contents: {documents[0].page_content}")
print(f"Document Metadata: {documents[0].metadata}")