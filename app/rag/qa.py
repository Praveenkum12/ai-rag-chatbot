from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from typing import Dict, Optional
import hashlib

# Global BM25 cache: {user_id: {filter_hash: BM25Retriever}}
_bm25_cache: Dict[str, Dict[str, BM25Retriever]] = {}

def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.2
    )

def get_prompt():
    return ChatPromptTemplate.from_template(
        """
        You are a helpful AI Knowledge Assistant. Use the provided context documents to answer the user's question.

        LONG-TERM MEMORIES:
        {user_memories}

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

def _hash_filter(filters: Optional[dict]) -> str:
    """Create a stable hash for filter dictionaries."""
    if not filters:
        return "no_filter"
    import json
    return hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()

def invalidate_bm25_cache(user_id: str):
    """Invalidate BM25 cache for a specific user (call on upload/delete)."""
    if user_id in _bm25_cache:
        del _bm25_cache[user_id]
        print(f"DEBUG: BM25 cache invalidated for user {user_id}")

def get_hybrid_retriever(vectordb, search_kwargs={"k": 5}, user_id: Optional[str] = None):
    """
    Optimized hybrid retriever with BM25 caching.
    Cache key: user_id + filter_hash
    """
    # 1. Get the Vector (Semantic) Retriever
    vector_retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    
    # 2. Build or retrieve cached BM25 Retriever
    filters = search_kwargs.get("filter")
    filter_hash = _hash_filter(filters)
    
    # Initialize user cache if needed
    if user_id and user_id not in _bm25_cache:
        _bm25_cache[user_id] = {}
    
    # Check cache
    if user_id and filter_hash in _bm25_cache[user_id]:
        print(f"DEBUG: Using cached BM25 for user {user_id}, filter {filter_hash[:8]}")
        keyword_retriever = _bm25_cache[user_id][filter_hash]
        keyword_retriever.k = search_kwargs.get("k", 5)  # Update k dynamically
    else:
        # Build BM25 from scratch
        print(f"DEBUG: Building new BM25 for user {user_id}, filter {filter_hash[:8]}")
        
        # Fetch content from Chroma
        if filters:
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
        
        # If no docs match the filter, return vector retriever only
        if not all_docs:
            return vector_retriever

        keyword_retriever = BM25Retriever.from_documents(all_docs)
        keyword_retriever.k = search_kwargs.get("k", 5)
        
        # Cache it
        if user_id:
            _bm25_cache[user_id][filter_hash] = keyword_retriever
    
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
