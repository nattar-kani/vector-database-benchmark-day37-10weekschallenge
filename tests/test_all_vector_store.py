from src.chunkers import sentence_chunks
from src.embeddings import EmbeddingModel
from src.loaders import load_document
from src.vector_store import VectorStore
from src.chroma_store import ChromaStore
from src.qdrant_store import QdrantStore


def test_all_vector_stores():

    document = load_document(
        "data/raw/sample.txt"
    )

    chunks = sentence_chunks(document)

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.embed_chunks(
        chunks
    )

    query_embedding = embedding_model.embed_query(
        "What is the annual leave policy?"
    )

    stores = {
        "faiss": VectorStore(),
        "chroma": ChromaStore(
            collection_name="test_all_chroma",
        ),
        "qdrant": QdrantStore(
            collection_name="test_all_qdrant",
        ),
    }

    for name, store in stores.items():

        store.add(embedded_chunks)

        results = store.search(
            query_embedding,
            top_k=3,
        )

        assert len(results) > 0, (
            f"{name} returned no results"
        )

        first_chunk, score = results[0]

        assert first_chunk.chunk.content
        assert isinstance(score, float)