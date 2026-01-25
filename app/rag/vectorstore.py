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
