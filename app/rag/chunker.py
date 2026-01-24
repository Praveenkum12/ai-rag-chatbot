from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150
) -> List[str]:
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())

        start = end - overlap
        if start < 0:
            start = 0

    return chunks

def create_chunk_records(
    chunks: List[str],
    document_id: str
) -> List[Dict]:
    records = []

    for i, chunk in enumerate(chunks):
        records.append({
            "id": f"{document_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "document_id": document_id,
                "chunk_index": i
            }
        })

    return records
