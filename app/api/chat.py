import time
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.pipelines.ingest_index import ingest_pdf_index
from app.pipelines.query_rag import query_rag
from app.models.query import QueryRequest

router = APIRouter(prefix="/chat", tags=["chat"])

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class ChatMessage(BaseModel):
    session_id: str
    doc_id: str
    message: str

@router.get("/ui", response_class=HTMLResponse)
def chat_ui():
    with open("app/web/chat.html", "r", encoding="utf-8") as f:
        return f.read()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    doc_id = f"doc_{session_id[:8]}"

    content = await file.read()
    pdf_path = DATA_DIR / f"{doc_id}.pdf"
    pdf_path.write_bytes(content)

    # Indexation (si c'est long, tu peux le mettre en background plus tard)
    ingest_pdf_index(str(pdf_path), doc_id=doc_id, source="web-ui", tags=[])

    return {"session_id": session_id, "doc_id": doc_id}

@router.post("/message")
def send_message(req: ChatMessage):
    # Logs de debug
    t0 = time.time()
    print("[chat] session_id:", req.session_id)
    print("[chat] doc_id:", req.doc_id)
    print("[chat] message:", req.message)

    try:
        t1 = time.time()
        print("[chat] calling query_rag...")
        result = query_rag(QueryRequest(question=req.message, doc_id=req.doc_id, top_k=6))
        print("[chat] query_rag done in", round(time.time() - t1, 2), "sec")

        # Assure serialization JSON des sources
        sources = result.sources
        if sources is None:
            sources = []
        else:
            fixed = []
            for s in sources:
                if hasattr(s, "model_dump"):
                    fixed.append(s.model_dump())
                elif isinstance(s, dict):
                    fixed.append(s)
                else:
                    # fallback: on évite le crash JSON
                    fixed.append({"source": str(s)})
            sources = fixed

        answer = result.answer if result.answer is not None else ""

        print("[chat] total", round(time.time() - t0, 2), "sec")
        return {"answer": answer, "sources": sources}

    except Exception as e:
        print("[chat] ERROR:", repr(e))
        return {"error": f"Server error: {type(e).__name__}: {e}"}