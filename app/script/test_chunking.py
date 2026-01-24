from pathlib import Path
from app.rag.chunker import chunk_text
from app.rag.loader import load_pdf, clean_text

pdf_path = Path("data/documents/e69163b4-7ab3-47ca-ba3a-fc5a99c25465.pdf")

text = clean_text(load_pdf(pdf_path))
chunks = chunk_text(text)

for i, c in enumerate(chunks[:1000]):
    print(f"\n--- Chunk {i} ---\n")
    print(c)
