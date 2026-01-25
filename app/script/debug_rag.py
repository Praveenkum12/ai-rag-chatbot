from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from app.rag.loader import load_pdf, clean_text
from app.rag.chunker import chunk_text
from app.rag.chroma_store import ChromaVectorStore
from app.rag.embedding import store_chunks
from app.rag.qa import answer_question

def debug():
    print("--- 1. Testing PDF Loading ---")
    pdf_path = Path("data/documents/e69163b4-7ab3-47ca-ba3a-fc5a99c25465.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return
    
    text = clean_text(load_pdf(pdf_path))
    print(f"Loaded text length: {len(text)} characters")
    if len(text) == 0:
        print("Error: Text extracted is empty. (Is it a scanned PDF?)")
        return

    print("\n--- 2. Testing Chunking ---")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    if not chunks:
        print("Error: No chunks created")
        return

    print("\n--- 3. Testing Chroma Store ---")
    vstore = ChromaVectorStore()
    doc_id = "test_debug_doc"
    chunk_list = [
        {
            "text": chunk,
            "metadata": {"doc_id": doc_id, "chunk_index": i},
            "id": f"{doc_id}_{i}"
        }
        for i, chunk in enumerate(chunks[:5])  # Just store 5 for test
    ]
    store_chunks(chunk_list, vstore)
    print("Stored 5 chunks in Chroma")

    print("\n--- 4. Testing Retrieval & QA ---")
    question = "What is this document about?"
    # Use the first chunk's text to ensure we find something
    if chunks:
        sample_text = chunks[0][:30]
        # Use repr() to safely print characters that might cause UnicodeEncodeError on Windows
        print(f"Using sample query from first chunk: {repr(sample_text)}...")
        result = answer_question(sample_text, vstore)
        # Safely print the answer
        print(f"Answer: {result['answer'].encode('utf-8', errors='replace').decode('utf-8')}")
        print(f"Sources: {len(result['sources'])} found")

if __name__ == "__main__":
    debug()
