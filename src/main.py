from .chunkers import sentence_chunks
from .embeddings import EmbeddingModel
from .loaders import load_document
from .retriever import Retriever
from .vector_store import VectorStore

def build_index(file_path:str):
    document = load_document(file_path=file_path)

    chunks = sentence_chunks(
        document,
        chunk_size=500
    )

    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.embed_chunks(chunks)

    vector_store = VectorStore()
    vector_store.add(embedded_chunks)

    return Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model
    )

def main():
    retriever = build_index(
        "data/raw/sample.txt"
    )

    query = "How many annual leave days do employees get?"

    results = retriever.retrieve(
        query,
        top_k=3,
    )

    print(f"\nQuery: {query}\n")
    print("Retrieved results:\n")

    for rank, (embedded_chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"Result {rank}")
        print(f"Score: {score:.4f}")
        print(f"Source: {embedded_chunk.chunk.source}")
        print(f"Content: {embedded_chunk.chunk.content}")
        print("-" * 60)


if __name__ == "__main__":
    main()