from pathlib import Path
from app.core.config import settings
from app.services.pdf_txt import extract_text_per_page, needs_ocr
from app.services.pdf_images import render_pdf_to_images
from app.services.ocr import ocr_image
from app.services.chunking import chunk_page_text
from app.services.embeddings import OllamaEmbeddingsClient
from app.services.qdrant_store import QdrantStore

def ingest_pdf_index(pdf_path: str, doc_id: str, source: str | None = None, tags: list[str] | None = None):
    tags = tags or []
    work = Path(settings.WORK_DIR) / doc_id
    pages_dir = work / "pages"
    work.mkdir(parents=True, exist_ok=True)

    page_texts = extract_text_per_page(pdf_path)
    page_images = render_pdf_to_images(pdf_path, str(pages_dir), dpi=settings.PDF_DPI)

    all_chunks = []
    pages_ocr = 0

    for i, txt in enumerate(page_texts, start=1):
        img = page_images[i - 1]
        extraction_type = "text"
        ocr_conf = None
        final_text = txt

        if needs_ocr(txt):
            extraction_type = "ocr"
            final_text, ocr_conf = ocr_image(img, lang=settings.OCR_LANG)
            pages_ocr += 1

        base_meta = {"source": source, "tags": tags, "ocr_confidence": ocr_conf, "page_image_path": img}

        all_chunks.extend(
            chunk_page_text(
                doc_id=doc_id,
                page=i,
                text=final_text,
                extraction_type=extraction_type,
                base_metadata=base_meta,
            )
        )

    embedder = OllamaEmbeddingsClient()
    texts = [c["text"] for c in all_chunks]
    vectors = embedder.embed_batch(texts)

    vector_size = len(vectors[0]) if vectors else 0

    store = QdrantStore()
    store.ensure_collection(vector_size=vector_size)
    store.upsert(
        ids=[c["id"] for c in all_chunks],
        vectors=vectors,
        payloads=[{**c["payload"], "text": c["text"], "chunk_id": c["id"]} for c in all_chunks],
    )

    return {
        "doc_id": doc_id,
        "pages_total": len(page_texts),
        "pages_ocr": pages_ocr,
        "pages_text": len(page_texts) - pages_ocr,
        "chunks_indexed": len(all_chunks),
        "vector_size": vector_size,
        "collection": store.collection,
        "workdir": str(work),
    }