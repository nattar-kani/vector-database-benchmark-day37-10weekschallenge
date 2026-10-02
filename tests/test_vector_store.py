import numpy as np

from src.chunkers import sentence_chunks
from src.embeddings import EmbeddingModel
from src.loaders import load_document
from src.vector_store import VectorStore


def test_vector_store_search():
    document = load_document("data/raw/sample.txt")

    chunks = sentence_chunks(
        document,
        chunk_size=500,
    )

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.embed_chunks(chunks)

    store = VectorStore()
    store.add(embedded_chunks)

    query = "How many annual leave days do employees get?"

    query_embedding = embedding_model.embed_query(query)

    results = store.search(
        query_embedding,
        top_k=3,
    )

    assert len(results) > 0
    assert len(results) <= 3

    best_chunk, best_score = results[0]

    assert best_score > 0
    assert "18 days" in best_chunk.chunk.content