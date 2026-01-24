from pathlib import Path
from typing import List
from pypdf import PdfReader


def load_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def clean_text(text: str) -> str:
    # basic normalization (we will improve later)
    text = text.replace("\x00", "")
    text = text.strip() 
    return text
