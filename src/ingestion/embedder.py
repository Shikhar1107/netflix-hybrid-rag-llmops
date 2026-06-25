from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
from src.ingestion.loader import load_netflix_data
from src.ingestion.chunker import create_documents
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5",model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},encode_kwargs={"normalize_embeddings": True})

    return embeddings

# df = load_netflix_data()

# docs = create_documents(df)


# vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db", collection_name="netflix_shows")

# query = "thriller movies like Dark"
# results = vector_store.similarity_search(query, k=5)
# for r in results:
#     print(r.metadata['title'])
#     print(r.page_content[:200])
#     print("---")