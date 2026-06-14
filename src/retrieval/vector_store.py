from langchain_chroma import Chroma
from src.ingestion.embedder import get_embedding_model

CHROMA_DB_PATH = "chroma_db"

def build_vector_store(documents):
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents = documents,
        embedding = embeddings,
        persist_directory = CHROMA_DB_PATH,
    )

    return vector_store

def load_vector_store():
    embeddings = get_embedding_model()

    vector_store = Chroma(
        persist_directory = CHROMA_DB_PATH,
        embedding_function = embeddings,
    )
    return vector_store