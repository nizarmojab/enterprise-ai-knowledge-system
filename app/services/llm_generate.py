import requests
from app.core.config import settings

class OllamaLLMClient:
    def __init__(self, base_url: str = None, model: str = "llama3.1:8b"):
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 400) -> str:
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,   # ✅ limite longueur réponse
                    "temperature": 0.2,
                },
            },
            timeout=600,  # ✅ augmente timeout (10 min)
        )
        r.raise_for_status()
        return r.json().get("response", "")