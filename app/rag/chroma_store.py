import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from typing import List, Dict
from .vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path="data/chroma")

        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]
