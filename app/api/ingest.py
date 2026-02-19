from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import uuid, shutil

from app.core.config import settings
from app.pipelines.ingest_report import ingest_pdf_report

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.post("/pdf-report")
async def pdf_report(doc_id: str, file: UploadFile = File(...)):
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    saved = Path(settings.UPLOAD_DIR) / f"{doc_id}_{uuid.uuid4().hex}.pdf"

    with open(saved, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return ingest_pdf_report(str(saved), doc_id=doc_id)
