from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    source_id: str = Field(..., description="S1, S2...")
    doc_id: str
    page: int
    chunk_id: str

class KeyPoint(BaseModel):
    text: str
    citations: List[str]  # ex ["S1","S3"]

class AnalysisResponse(BaseModel):
    summary: str
    key_points: List[KeyPoint]
    concepts: List[str]
    limitations: List[str]
    confidence: float
    sources: List[Citation]