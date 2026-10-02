import chromadb

from .models import EmbeddedChunk, Chunk
from .vector_store_base import BaseVectorStore

class ChromaStore(BaseVectorStore):
    def __init__(self, collection_name: str ="day37"):
        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )


    def add(
            self,
            embedded_chunks: list[EmbeddedChunk]
    )->None:
        if not embedded_chunks:
            return

        self.collection.add(
            ids=[
                f"{item.chunk.source}_{item.chunk.chunk_index}"
                for item in embedded_chunks
            ],
            embeddings=[
                item.embedding.tolist()    
                for item in embedded_chunks
            ],
            documents=[
                item.chunk.content
                for item in embedded_chunks
            ],
            metadatas=[
                {
                    "source": item.chunk.source,
                    "chunk_index": item.chunk.chunk_index,
                    "strategy": item.chunk.strategy,
                    **item.chunk.metadata,
                }
                for item in embedded_chunks
            ]
        )
        
    def search(self, query_embedding, top_k = 3)->list[tuple[EmbeddedChunk,float]]:
        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document,metadata,distance in zip(
            documents,metadatas,distances
        ):
            chunk = EmbeddedChunk(
                chunk=Chunk(
                    content=document,
                    source=metadata["source"],
                    chunk_index=metadata["chunk_index"],
                    strategy=metadata["strategy"],
                    metadata=metadata,
                ),
                embedding=None,
            )

            score = 1 - distance

            retrieved.append(
                (chunk, score)
            )

        return retrieved