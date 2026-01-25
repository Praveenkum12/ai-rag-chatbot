from pathlib import Path
from app.rag.chunker import chunk_text
from app.rag.loader import load_pdf, clean_text

# Get the latest uploaded PDF automatically
pdf_files = sorted(Path("data/documents").glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
if not pdf_files:
    print("No PDF files found in data/documents/")
    exit()

pdf_path = pdf_files[0]
print(f"Testing chunking for: {pdf_path.name}")

text = clean_text(load_pdf(pdf_path))
chunks = chunk_text(text)

for i, c in enumerate(chunks[:1000]):
    print(f"\n--- Chunk {i} ---\n")
    print(c)
