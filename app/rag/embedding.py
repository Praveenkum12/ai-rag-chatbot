from typing import List, Dict
from .chroma_store import ChromaVectorStore


def store_chunks(
    chunks: List[Dict],
    vector_store: ChromaVectorStore
) -> None:
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )
