from .chunkers import (
    fixed_size_chunks,
    recursive_chunks,
    sentence_chunks,
    semantic_chunks,
)
from .embeddings import EmbeddingModel
from .evaluation import evaluate_retrieval
from .loaders import load_document
from .retriever import Retriever
from .vector_store import VectorStore


def build_retriever(
    document,
    chunker,
    embedding_model,
):
    chunks = chunker(document)

    embedded_chunks = embedding_model.embed_chunks(
        chunks
    )

    vector_store = VectorStore()
    vector_store.add(embedded_chunks)

    return Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    ), len(chunks)


def run_retrieval_benchmark():

    document = load_document(
        "data/raw/sample.txt"
    )

    embedding_model = EmbeddingModel()

    strategies = {
        "fixed_size": lambda doc: fixed_size_chunks(doc),
        "recursive": lambda doc: recursive_chunks(doc),
        "sentence": lambda doc: sentence_chunks(doc),
        "semantic": lambda doc: semantic_chunks(
            doc,
            model=embedding_model.model,
        ),
    }

    results = []

    for name, chunker in strategies.items():

        retriever, chunk_count = build_retriever(
            document,
            chunker,
            embedding_model,
        )

        evaluation_results = evaluate_retrieval(
            retriever,
            "data/evaluation/retrieval_questions.csv",
            top_k=3,
        )

        successful = sum(
            result["retrieved"]
            for result in evaluation_results
        )

        total = len(evaluation_results)

        recall = successful / total

        results.append(
            {
                "strategy": name,
                "chunk_count": chunk_count,
                "successful_queries": successful,
                "total_queries": total,
                "recall_at_3": round(recall, 3),
            }
        )

    return results


if __name__ == "__main__":

    results = run_retrieval_benchmark()

    print("\nRetrieval Benchmark\n")

    for result in results:
        print(result)