from src.chunkers import sentence_chunks
from src.embeddings import EmbeddingModel
from src.evaluation import evaluate_retrieval
from src.loaders import load_document
from src.retriever import Retriever
from src.vector_store import VectorStore


def test_evaluate_retrieval():

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

    results = evaluate_retrieval(
        retriever,
        "data/evaluation/retrieval_questions.csv",
        top_k=3,
    )

    assert len(results) == 5

    assert all(
        result["retrieved"]
        for result in results
    )