from pathlib import Path
from typing import List
from pypdf import PdfReader


def load_pdf(file_path: Path) -> List[dict]:
    """
    Returns a list of objects: [{"page": 1, "text": "..."}, ...]
    """
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page": i + 1,
                "text": text
            })

    return pages


def clean_text(text: str) -> str:
    # basic normalization (we will improve later)
    text = text.replace("\x00", "")
    text = text.strip() 
    return text
