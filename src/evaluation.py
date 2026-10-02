import csv

from .retriever import Retriever


def evaluate_retrieval(
    retriever: Retriever,
    evaluation_path: str,
    top_k: int = 3,
) -> list[dict]:

    results = []

    with open(
        evaluation_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            question = row["question"]
            expected_text = row["expected_text"]

            retrieved = retriever.retrieve(
                question,
                top_k=top_k,
            )

            retrieved_text = " ".join(
                item.chunk.content
                for item, _ in retrieved
            )

            success = (
                expected_text.lower()
                in retrieved_text.lower()
            )

            results.append(
                {
                    "question": question,
                    "expected_text": expected_text,
                    "retrieved": success,
                    "result_count": len(retrieved),
                }
            )

    return results