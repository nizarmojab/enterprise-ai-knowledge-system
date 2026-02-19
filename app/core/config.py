from pydantic import BaseModel

class Settings(BaseModel):
    OLLAMA_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "nomic-embed-text"
    VISION_MODEL: str = "llava"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "docs"

    OCR_LANG: str = "eng+fra"
    PDF_DPI: int = 200

    DATA_DIR: str = "data"
    WORK_DIR: str = "data/work"
    UPLOAD_DIR: str = "data/uploads"

settings = Settings()
