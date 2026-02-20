from fastapi import APIRouter
from app.models.query import QueryRequest, QueryResponse
from app.pipelines.query_rag import query_rag

router = APIRouter(prefix="/query", tags=["query"])

@router.post("", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    return query_rag(req)