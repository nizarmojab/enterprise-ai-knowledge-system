import requests
from app.core.config import settings

class OllamaEmbeddingsClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.model = model or settings.EMBED_MODEL

    def embed_one(self, text: str) -> list[float]:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]