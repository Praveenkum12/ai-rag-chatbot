from typing import List, Dict
from .chroma_store import ChromaVectorStore


def retrieve_context(
    query: str,
    vector_store: ChromaVectorStore,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k relevant chunks for a query.
    """
    return vector_store.similarity_search(query=query, k=k)
