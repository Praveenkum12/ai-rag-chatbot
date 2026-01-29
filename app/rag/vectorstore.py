from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectordb = Chroma(
        persist_directory="data/chroma",
        embedding_function=embeddings
    )

    return vectordb


def clear_vectorstore(vectordb: Chroma):
    """
    Clears the Chroma vector store by deleting the collection and 
    effectively starting fresh.
    """
    vectordb.delete_collection()
    # It's often good practice to return a fresh instance or 
    # the user might need to re-initialize if they want to use it immediately.
    # However, delete_collection() usually means the object is no longer valid 
    # for certain operations until re-created.
    # We will just return a new instance from here.
    return get_vectorstore()
