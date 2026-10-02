from sentence_transformers import SentenceTransformer

from src.loaders import load_document
from src.chunkers import semantic_chunks

document = load_document("data/raw/sample.txt")
model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = semantic_chunks(
    document,
    similarity_threshold=0.5,
    model=model
)


print(f"Total semantic chunks: {len(chunks)}")

for chunk in chunks:
    print(f"\n--- Chunk {chunk.chunk_index} ---")
    print(chunk.content)