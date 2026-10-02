import time
import csv

from sentence_transformers import SentenceTransformer

from .chunkers import (
    fixed_size_chunks,
    sentence_chunks,
    recursive_chunks,
    semantic_chunks
)
from .loaders import load_document

def benchmark_chunkers(document):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    strategies = {
        "fixed_size": lambda: fixed_size_chunks(document),
        "recursive": lambda: recursive_chunks(document),
        "sentence": lambda: sentence_chunks(document),
        "semantic": lambda: semantic_chunks(
            document,
            model=model,
        ),
    }

    results = []

    for name,chunker in strategies.items():
        start_time = time.perf_counter()

        chunks = chunker()

        elapsed = time.perf_counter() - start_time

        lengths = [len(chunk.content) for chunk in chunks]

        results.append(
            {
                "strategy": name,
                "chunk_count": len(chunks),
                "avg_chunk_size": round(sum(lengths) / len(lengths), 2),
                "min_chunk_size": min(lengths),
                "max_chunk_size": max(lengths),
                "processing_time": round(elapsed, 6)
            }
        )

    return results

if __name__ == "__main__":
    document = load_document("data/raw/sample.txt")

    results = benchmark_chunkers(document)

    output_path = "data/processed/benchmark_results.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("Results saved")