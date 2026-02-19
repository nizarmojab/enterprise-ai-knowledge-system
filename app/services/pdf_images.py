from pdf2image import convert_from_path
from pathlib import Path

def render_pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    for i, img in enumerate(pages, start=1):
        p = Path(out_dir) / f"page_{i:04d}.png"
        img.save(p, "PNG")
        paths.append(str(p))
    return paths
