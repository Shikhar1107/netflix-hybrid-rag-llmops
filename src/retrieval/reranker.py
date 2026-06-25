from sentence_transformers import CrossEncoder
from functools import lru_cache

RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

class Reranker:
    def __init__ (self):
        self .model = CrossEncoder(RERANKER_MODEL_NAME)
    
    def rerank(self, query, documents, top_k = 5):

        if not documents:
            return []
        
        # Create pairs of query and document content for scoring
        pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance scores for each pair
        scores = self.model.predict(pairs)

        # Combine documents with their scores and sort by score
        reranked_results = sorted(zip(scores, documents), key = lambda x: x[0], reverse=True)

        return [
            {
                "score": float(score),
                "document" : doc
            }
            for score,doc in reranked_results[:top_k]
        ]

@lru_cache(maxsize=1)
def get_reranker_model():
    return CrossEncoder(RERANKER_MODEL_NAME)