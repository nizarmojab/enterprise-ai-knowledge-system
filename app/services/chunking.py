import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_page_text(
    doc_id: str,
    page: int,
    text: str,
    extraction_type: str,
    base_metadata: dict,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    text = text or ""
    parts = splitter.split_text(text)
    chunks = []
    cursor = 0

    for part in parts:
        idx = text.find(part, cursor)
        if idx == -1:
            idx = cursor

        chunk_id = str(uuid.uuid4())
        chunks.append(
            {
                "id": chunk_id,
                "text": part,
                "payload": {
                    "doc_id": doc_id,
                    "page": page,
                    "extraction_type": extraction_type,
                    "char_start": idx,
                    "char_end": idx + len(part),
                    **base_metadata,
                },
            }
        )
        cursor = idx + len(part)

    return chunks