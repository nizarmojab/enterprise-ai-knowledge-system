from fastapi import APIRouter
from app.models.query import QueryRequest
from app.models.analysis import AnalysisResponse
from app.pipelines.analysis_rag import analysis_rag

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/analysis", response_model=AnalysisResponse)
def analysis_endpoint(req: QueryRequest):
    return analysis_rag(req)