from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.core.config import settings

class QdrantStore:
    def __init__(self, url: str = None, collection: str = None):
        self.url = url or settings.QDRANT_URL
        self.collection = collection or settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self, vector_size: int):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)