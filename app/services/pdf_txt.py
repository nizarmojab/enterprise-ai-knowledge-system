from pypdf import PdfReader

def extract_text_per_page(pdf_path: str) -> list[str]:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return texts

def needs_ocr(page_text: str, min_chars: int = 30) -> bool:
    cleaned = "".join(page_text.split())
    return len(cleaned) < min_chars
