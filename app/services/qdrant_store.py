from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.core.config import settings


def build_filter(doc_id: str | None = None, tags: list[str] | None = None) -> Filter | None:
    must = []

    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

    if tags:
        for t in tags:
            must.append(FieldCondition(key="tags", match=MatchValue(value=t)))

    return Filter(must=must) if must else None


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

    def search(self, query_vector: list[float], top_k: int = 5, qfilter: Filter | None = None):
        """
        Compat wrapper for multiple qdrant-client versions:
        - some have: client.search(...)
        - some have: client.search_points(...).points
        - newer may have: client.query_points(...).points
        """
        # 1) Old/standard API
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )

        # 2) Some versions
        if hasattr(self.client, "search_points"):
            res = self.client.search_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )
            return res.points

        # 3) Newer query API
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qfilter,
            )
            return res.points

        raise RuntimeError(
            "Unsupported qdrant-client version: no search/search_points/query_points found. "
            "Please upgrade qdrant-client."
        )