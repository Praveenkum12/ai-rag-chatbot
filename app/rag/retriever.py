from typing import List, Dict
from .vector_store import VectorStore


def retrieve_context(
    query: str,
    vector_store: VectorStore,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k relevant chunks for a query.
    """
    return vector_store.similarity_search(query=query, k=k)
