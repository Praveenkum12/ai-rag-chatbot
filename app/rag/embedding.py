from typing import List, Dict
from .vector_store import VectorStore

def store_chunks(
    chunks: List[Dict],
    vector_store: VectorStore
) -> None:
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )
