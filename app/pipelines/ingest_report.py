from pathlib import Path
from app.core.config import settings
from app.services.pdf_txt import extract_text_per_page, needs_ocr
from app.services.pdf_images import render_pdf_to_images
from app.services.ocr import ocr_image

def ingest_pdf_report(pdf_path: str, doc_id: str) -> dict:
    work = Path(settings.WORK_DIR) / doc_id
    pages_dir = work / "pages"
    work.mkdir(parents=True, exist_ok=True)

    page_texts = extract_text_per_page(pdf_path)
    page_images = render_pdf_to_images(pdf_path, str(pages_dir), dpi=settings.PDF_DPI)

    pages = []
    ocr_pages = 0

    for i, txt in enumerate(page_texts, start=1):
        img = page_images[i - 1]
        if needs_ocr(txt):
            ocr_txt, conf = ocr_image(img, lang=settings.OCR_LANG)
            pages.append({
                "page": i,
                "extraction_type": "ocr",
                "ocr_confidence": conf,
                "chars": len(ocr_txt),
            })
            ocr_pages += 1
        else:
            pages.append({
                "page": i,
                "extraction_type": "text",
                "ocr_confidence": None,
                "chars": len(txt),
            })

    return {
        "doc_id": doc_id,
        "pdf_path": pdf_path,
        "pages_total": len(page_texts),
        "pages_ocr": ocr_pages,
        "pages_text": len(page_texts) - ocr_pages,
        "pages_detail": pages,
        "workdir": str(work),
    }
