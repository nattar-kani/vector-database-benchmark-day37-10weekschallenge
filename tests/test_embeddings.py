from src.embeddings import EmbeddingModel
from src.loaders import load_document
from src.chunkers import sentence_chunks
from src.models import EmbeddedChunk


def test_embed_chunks():
    document = load_document("data/raw/sample.txt")

    chunks = sentence_chunks(
        document,
        chunk_size=500,
    )

    embedding_model = EmbeddingModel()

    embedded_chunks = embedding_model.embed_chunks(chunks)

    assert len(embedded_chunks) == len(chunks)

    for embedded_chunk, original_chunk in zip(
        embedded_chunks,
        chunks,
    ):
        assert isinstance(embedded_chunk, EmbeddedChunk)
        assert embedded_chunk.chunk is original_chunk
        assert embedded_chunk.embedding.shape == (384,)