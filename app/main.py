from fastapi import FastAPI
from app.api.ingest import router as ingest_router

app = FastAPI(title="Enterprise AI Knowledge System")
app.include_router(ingest_router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Enterprise AI Knowledge System API. Go to /docs"}

from app.api.query import router as query_router
app.include_router(query_router)