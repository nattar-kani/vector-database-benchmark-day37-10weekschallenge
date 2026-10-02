from pathlib import Path

from src.benchmark import benchmark_chunkers
from src.loaders import load_document


def test_benchmark_returns_all_strategies():

    document = load_document("data/raw/sample.txt")

    results = benchmark_chunkers(document)

    strategies = {
        result["strategy"]
        for result in results
    }

    assert strategies == {
        "fixed_size",
        "recursive",
        "sentence",
        "semantic",
    }


def test_benchmark_contains_required_metrics():
    document = load_document("data/raw/sample.txt")

    results = benchmark_chunkers(document)

    required_fields = {
        "strategy",
        "chunk_count",
        "avg_chunk_size",
        "min_chunk_size",
        "max_chunk_size",
        "processing_time",
    }

    for result in results:
        assert required_fields.issubset(result.keys())
        assert result["chunk_count"] > 0
        assert result["avg_chunk_size"] > 0
        assert result["min_chunk_size"] > 0
        assert result["max_chunk_size"] > 0
        assert result["processing_time"] >= 0