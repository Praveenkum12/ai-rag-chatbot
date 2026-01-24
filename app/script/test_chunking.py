from pathlib import Path
from app.rag.chunker import chunk_text
from app.rag.loader import load_pdf, clean_text

pdf_path = Path("data/documents/YOUR_DOC_ID.pdf")

text = clean_text(load_pdf(pdf_path))
chunks = chunk_text(text)

for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} ---\n")
    print(c)
