from src.chunkers import sentence_chunks
from src.embeddings import EmbeddingModel
from src.loaders import load_document
from src.retriever import Retriever
from src.vector_store import VectorStore


def test_retriever_returns_relevant_results():
    document = load_document("data/raw/sample.txt")

    chunks = sentence_chunks(
        document,
        chunk_size=500,
    )

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.embed_chunks(chunks)

    vector_store = VectorStore()
    vector_store.add(embedded_chunks)

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    results = retriever.retrieve(
        "How many annual leave days do employees get?",
        top_k=3,
    )

    assert len(results) > 0
    assert len(results) <= 3

    best_chunk, best_score = results[0]

    assert best_score > 0
    assert "18 days" in best_chunk.chunk.content