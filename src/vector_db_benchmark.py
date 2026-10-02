import time
import csv

from .loaders import load_document
from .chunkers import sentence_chunks
from .chroma_store import ChromaStore
from .embeddings import EmbeddingModel
from .qdrant_store import QdrantStore
from .vector_store import VectorStore


def build_benchmark_data():

    document = load_document(
        "data/raw/sample.txt"
    )

    chunks = sentence_chunks(document)

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.embed_chunks(
        chunks
    )

    return embedded_chunks, embedding_model


def benchmark_ingestion(
    store,
    embedded_chunks,
):

    start = time.perf_counter()

    store.add(embedded_chunks)

    end = time.perf_counter()

    return end - start


def benchmark_query_latency(
    store,
    query_embeddings,
    top_k=3,
):

    latencies = []

    for query_embedding in query_embeddings:

        start = time.perf_counter()

        store.search(
            query_embedding,
            top_k=top_k,
        )

        end = time.perf_counter()

        latencies.append(
            end - start
        )

    return sum(latencies) / len(latencies)


def benchmark_recall(
    store,
    embedding_model,
    evaluation_path,
    top_k=3,
):

    correct = 0
    total = 0

    with open(
        evaluation_path,
        "r",
        encoding="utf-8",
    ) as file:

        import csv

        reader = csv.DictReader(file)

        for row in reader:

            query = row["question"]
            expected_text = row["expected_text"]

            query_embedding = (
                embedding_model.embed_query(query)
            )

            results = store.search(
                query_embedding,
                top_k=top_k,
            )

            retrieved_text = " ".join(
                item.chunk.content
                for item, _ in results
            )

            if (
                expected_text.lower()
                in retrieved_text.lower()
            ):
                correct += 1

            total += 1

    return correct / total


def run_benchmark():

    embedded_chunks, embedding_model = (
        build_benchmark_data()
    )

    stores = {
        "faiss": VectorStore(),

        "chroma": ChromaStore(
            collection_name="benchmark_chroma"
        ),

        "qdrant": QdrantStore(
            collection_name="benchmark_qdrant"
        ),
    }

    results = []

    for name, store in stores.items():

        print(f"\nBenchmarking: {name}")

        ingestion_time = benchmark_ingestion(
            store,
            embedded_chunks,
        )

        query_embeddings = []

        evaluation_path = (
            "data/evaluation/retrieval_questions.csv"
        )

        with open(
            evaluation_path,
            "r",
            encoding="utf-8",
        ) as file:

            import csv

            reader = csv.DictReader(file)

            for row in reader:

                query_embeddings.append(
                    embedding_model.embed_query(
                        row["question"]
                    )
                )

        latency = benchmark_query_latency(
            store,
            query_embeddings,
            top_k=3,
        )

        recall = benchmark_recall(
            store,
            embedding_model,
            evaluation_path,
            top_k=3,
        )

        result = {
            "store": name,
            "chunk_count": len(embedded_chunks),
            "ingestion_time": round(
                ingestion_time,
                6,
            ),
            "avg_query_latency": round(
                latency,
                6,
            ),
            "recall_at_3": round(
                recall,
                3,
            ),
            "cost": "$0 external"
        }

        results.append(result)

        print(result)

    return results

def save_results(results):

    output_path = (
        "data/processed/vector_db_results.csv"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "store",
                "chunk_count",
                "ingestion_time",
                "avg_query_latency",
                "recall_at_3",
                "cost",
            ],
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

if __name__ == "__main__":

    results = run_benchmark()

    save_results(results)

    print(
        "\nResults saved to "
        "data/processed/vector_db_results.csv"
    )