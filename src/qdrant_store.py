from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .models import Chunk, EmbeddedChunk
from .vector_store_base import BaseVectorStore

class QdrantStore(BaseVectorStore):

    def __init__(
        self,
        collection_name: str="day37",
        dimension: int=384
    ):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )

    def add(
            self,
            embedded_chunks: list[EmbeddedChunk]
    )->None:

        if not embedded_chunks:
            return

        points = []

        for index, item in enumerate(embedded_chunks):
            points.append(
                PointStruct(
                    id=index,
                    vector=item.embedding.tolist(),
                    payload={
                        "content": item.chunk.content,
                        "source": item.chunk.source,
                        "chunk_index": item.chunk.chunk_index,
                        "strategy": item.chunk.strategy,
                        "metadata": item.chunk.metadata,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_embedding,
        top_k:int=3
    )-> list[tuple[EmbeddedChunk,float]]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k
        ).points

        retrieved = []

        for result in results:
            payload = result.payload

            chunk = Chunk(
                content=payload["content"],
                source=payload["source"],
                chunk_index=payload["chunk_index"],
                strategy=payload["strategy"],
                metadata=payload["metadata"],
            )

            embedded_chunk = EmbeddedChunk(
                chunk=chunk,
                embedding=None
            )

            retrieved.append(
                (embedded_chunk,float(result.score))
            )

        return retrieved
        