import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


def project_root() -> str:
    """
    Retourne le dossier racine du projet en partant de app/rag/pdf_loader.py
    app/rag/pdf_loader.py -> app/rag -> app -> (root)
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


if __name__ == "__main__":
    # ✅ Change uniquement le nom du fichier si besoin
    PDF_FILENAME = "ai_business.pdf"

    base_dir = project_root()
    pdf_path = os.path.join(base_dir, "data", PDF_FILENAME)

    print("=== PDF Loader Debug ===")
    print("Project root:", base_dir)
    print("PDF path:", pdf_path)

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"PDF introuvable: {pdf_path}\n"
            "➡️ Vérifie que le fichier est bien dans le dossier /data et que le nom est correct."
        )

    docs = load_pdf(pdf_path)

    print("\n✅ Loaded!")
    print("Pages loaded:", len(docs))

    # Afficher metadata + preview sur les 5 premières pages
    print("\n=== First 5 pages check ===")
    for i, doc in enumerate(docs[:5]):
        text = doc.page_content or ""
        print(f"\n--- Page {i + 1} ---")
        print("Metadata:", doc.metadata)
        print("Characters:", len(text))
        preview = text[:300].replace("\n", " ").strip()
        print("Preview:", preview if preview else "[EMPTY]")

    # Chercher quelques pages non vides (utile si le PDF est scanné / image-based)
    print("\n=== Searching for non-empty pages (up to 10) ===")
    found = 0
    for i, doc in enumerate(docs):
        text = (doc.page_content or "").strip()
        if len(text) > 50:
            preview = text[:250].replace("\n", " ").strip()
            print(f"\n✅ Non-empty page found: {i + 1}")
            print("Characters:", len(text))
            print("Preview:", preview)
            found += 1
            if found >= 10:
                break

    if found == 0:
        print(
            "\n⚠️ Aucun texte extractible détecté dans ce PDF.\n"
            "➡️ Il est probablement scanné (images) et nécessite OCR.\n"
            "➡️ Solution: utiliser un autre PDF texte OU ajouter une étape OCR plus tard."
        )
