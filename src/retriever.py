from .embeddings import EmbeddingModel
from .models import EmbeddedChunk
from .vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[EmbeddedChunk, float]]:

        query_embedding = self.embedding_model.embed_query(query)

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )