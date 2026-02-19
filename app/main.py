from fastapi import FastAPI
from pydantic import BaseModel

from app.core.llm import get_llm

app = FastAPI(title="Enterprise AI Knowledge System")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Enterprise AI Knowledge System is running"}


@app.post("/ask")
def ask(req: AskRequest):
    llm = get_llm()
    resp = llm.invoke(req.question)
    return {"answer": resp.content}
