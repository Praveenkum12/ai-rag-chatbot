from abc import ABC, abstractmethod
from typing import List, Dict


class VectorStore(ABC):
    """
    Abstract interface for a vector database.
    """

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        """
        Store texts and their embeddings.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve top-k similar texts for a query.
        """
        pass
