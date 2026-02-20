from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    doc_id: Optional[str] = None           # optionnel: filtrer un document
    top_k: int = 5                         # nb de chunks à récupérer
    min_score: float = 0.0                 # optionnel
    tags: Optional[List[str]] = None       # optionnel: filtrer par tags

class SourceRef(BaseModel):
    doc_id: str
    page: int
    chunk_id: str
    extraction_type: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceRef]
    debug: Optional[Dict[str, Any]] = None