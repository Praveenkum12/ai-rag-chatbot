from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.2
    )

def get_prompt():
    return ChatPromptTemplate.from_template(
        """
        You are a helpful AI Knowledge Assistant. Use the provided context documents to answer the user's question.

        CONVERSATION HISTORY:
        {chat_history}

        CONTEXT DOCUMENTS:
        {context}

        INSTRUCTIONS:
        1. Answer the Question based ONLY on the Context Documents.
        2. If you find the answer in the documents, cite the source using [1], [2], etc.
        3. If the answer is NOT in the documents, say "I don't know."
        4. Use the Conversation History to understand follow-up questions (like "who is he?").

        Question: {question}
        Answer:
        """
    )

def get_hybrid_retriever(vectordb, search_kwargs={"k": 5}):
    # 1. Get the Vector (Semantic) Retriever
    vector_retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    
    # 2. Build the Keyword (BM25) Retriever with the SAME filters
    # We must restrict BM25 documents to match the metadata filter
    filters = search_kwargs.get("filter")
    
    # Fetch content from Chroma
    if filters:
        # Chroma .get() supports complex filters
        all_data = vectordb.get(where=filters, include=["documents", "metadatas"])
    else:
        all_data = vectordb.get(include=["documents", "metadatas"])
        
    all_docs = []
    from langchain_core.documents import Document
    
    if all_data and "ids" in all_data:
        for i in range(len(all_data["ids"])):
            all_docs.append(Document(
                page_content=all_data["documents"][i],
                metadata=all_data["metadatas"][i]
            ))
    
    # If no docs match the filter, just return the vector retriever 
    # (which will also return nothing, but safely)
    if not all_docs:
        return vector_retriever

    keyword_retriever = BM25Retriever.from_documents(all_docs)
    keyword_retriever.k = search_kwargs.get("k", 5)
    
    # 3. Combine them (Ensemble)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever


def get_qa_chain(vectordb):
    retriever = get_hybrid_retriever(vectordb)
    llm = get_llm()
    prompt = get_prompt()

    # This is the "Full Auto" chain used at startup
    chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough()
        }).assign(
            answer=(
                prompt
                | llm
                | StrOutputParser()
            )
        )
    )

    return chain
