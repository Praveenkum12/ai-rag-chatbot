from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List, Optional
from pathlib import Path
import uuid
import tiktoken

from app.rag.ingest import load_and_split_document
from app.rag.vectorstore import clear_vectorstore
from app.rag.qa import get_qa_chain
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

from datetime import datetime, timedelta

class ChatRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = []
    file_type: Optional[str] = None
    date_filter: Optional[str] = None # 'today', 'week', 'any'
    history: Optional[List[dict]] = [] # [{"role": "user", "content": "..."}, ...]

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_ids: List[str] = None
    file_type: Optional[str] = None
    date_filter: Optional[str] = None


@router.post("/documents/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(allowed_extensions)}"
        )

    try:
        doc_id = str(uuid.uuid4())
        file_path = DATA_DIR / f"{doc_id}{file_ext}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = load_and_split_document(str(file_path))
        
        if not chunks:
            raise ValueError("No text content could be extracted from this document.")

        from datetime import datetime
        
        # Inject cleaner metadata for easier retrieval/filtering
        for chunk in chunks:
            chunk.metadata["filename"] = file.filename
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["file_type"] = file_ext.replace(".", "")
            chunk.metadata["processed_at"] = datetime.now().isoformat()
            chunk.metadata["source"] = doc_id 

        request.app.state.vectordb.add_documents(chunks)

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents")
async def list_documents(request: Request):
    """
    Returns a list of unique documents by querying the vector database's metadata.
    """
    try:
        # Get all metadata from Chroma
        # We need to know which doc_id belongs to which original filename
        data = request.app.state.vectordb.get(include=["metadatas"])
        
        unique_docs = {}
        if data and "metadatas" in data:
            for meta in data["metadatas"]:
                d_id = meta.get("doc_id")
                fname = meta.get("filename", "Unknown")
                ftype = meta.get("file_type", "txt")
                
                if d_id and d_id not in unique_docs:
                    unique_docs[d_id] = {
                        "id": d_id,
                        "name": fname,
                        "type": ftype
                    }
        
        return list(unique_docs.values())
    except Exception as e:
        print(f"Error listing documents: {e}")
        return []


@router.post("/chat")
def chat(request: Request, chat_request: ChatRequest):
    try:
        # Construct filters for retrieval
        meta_filters = []
        
        # 1. Doc IDs filter
        if chat_request.doc_ids:
            if len(chat_request.doc_ids) == 1:
                meta_filters.append({"doc_id": {"$eq": chat_request.doc_ids[0]}})
            else:
                meta_filters.append({"doc_id": {"$in": chat_request.doc_ids}})
        
        # 2. File Type filter
        if chat_request.file_type and chat_request.file_type != "all":
            meta_filters.append({"file_type": {"$eq": chat_request.file_type}})
            
        # 3. Date filter
        if chat_request.date_filter and chat_request.date_filter != "any":
            now = datetime.now()
            if chat_request.date_filter == "today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif chat_request.date_filter == "week":
                cutoff = now - timedelta(days=7)
            
            meta_filters.append({"processed_at": {"$gte": cutoff.isoformat()}})

        # Combine filters with $and if multiple exist
        filters = None
        if len(meta_filters) == 1:
            filters = meta_filters[0]
        elif len(meta_filters) > 1:
            filters = {"$and": meta_filters}

        from app.rag.qa import get_hybrid_retriever, get_llm, get_prompt
        from langchain_core.output_parsers import StrOutputParser
        
        # 1. Get Hybrid Retriever
        retriever = get_hybrid_retriever(
            request.app.state.vectordb, 
            search_kwargs={"k": 5, "filter": filters if filters else None}
        )
        
        print(f"DEBUG: Active filters: {filters}")
        
        # 2. Invoke retriever
        docs = retriever.invoke(chat_request.question)
        print(f"DEBUG: Retrieved {len(docs)} docs for question: '{chat_request.question}'")

        # --- RELEVANCE FILTERING REMOVED ---
        # Passing all retrieved documents to the LLM regardless of relevance score as requested.
        # ---------------------------

        # 3. Check if it's a greeting
        is_greeting = chat_request.question.lower() in ["hi", "hello", "hey", "greetings"]
        
        # 4. Run LLM
        llm = get_llm()
        prompt = get_prompt()
        answer_chain = prompt | llm | StrOutputParser()
        
        # Clean and Label Docs
        # We also perform a quick scoring pass to get confidence percentages
        # Match each doc to its distance score from the vector store
        context_parts = []
        source_data = []

        # Get scores for the top docs in this specific search
        # Note: We use search_with_score to get the raw numbers
        scored_results = {
            doc.page_content: score 
            for doc, score in request.app.state.vectordb.similarity_search_with_score(
                chat_request.question, k=10, filter=filters if filters else None
            )
        }
        for i, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "Unknown File")
            clean_content = doc.page_content.replace("\x00", "").replace("♂", "").replace("¶", "").replace("•", "-")
            context_parts.append(f"[Source {i+1}]: From {filename}\n{clean_content}")
            
            # Calculate Percentage: 0.0 distance is 100%, 1.2+ is ~0%
            raw_score = scored_results.get(doc.page_content, 1.0)
            confidence = max(0, min(100, round((1 - (raw_score / 1.5)) * 100)))
            
            source_data.append({
                "content": doc.page_content,
                "metadata": {**doc.metadata, "confidence": confidence}
            })
        
        context_text = "\n\n".join(context_parts)
        
        # 5. Format History for the Prompt
        history_text = ""
        for msg in chat_request.history:
            role = "HUMAN" if msg.get("role") == "user" else "AI"
            history_text += f"{role}: {msg.get('content')}\n"

        if not history_text:
            history_text = "No previous conversation."

        print(f"DEBUG: Context length: {len(context_text)}")
        
        answer = answer_chain.invoke({
            "question": chat_request.question,
            "context": context_text if context_text else "No context documents found.",
            "chat_history": history_text
        })
        
        # Hide sources if it's a greeting OR if the AI says "I don't know"
        hide_sources = is_greeting or "i don't know" in answer.lower()
        
        # 6. Calculate Tokens for tracking
        total_tokens = count_tokens(chat_request.question) + count_tokens(context_text) + count_tokens(history_text)
        print(f"DEBUG: Token usage for this request: {total_tokens}")

        return {
            "answer": answer,
            "sources": source_data if not hide_sources and docs else [],
            "token_usage": total_tokens
        }
    except Exception as e:
        print(f"CRITICAL CHAT ERROR: {str(e)}")
        return {
            "answer": "I don't know.",
            "sources": []
        }


@router.post("/search")
def search(request: Request, search_request: SearchRequest):
    filter_dict = None
    if search_request.doc_ids:
        if len(search_request.doc_ids) == 1:
            filter_dict = {"doc_id": search_request.doc_ids[0]}
        else:
            filter_dict = {"doc_id": {"$in": search_request.doc_ids}}

    results = request.app.state.vectordb.similarity_search(
        search_request.query, 
        k=search_request.k,
        filter=filter_dict
    )
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    """
    Deletes a single document from both Chroma and the disk.
    """
    try:
        # 1. Delete from Chroma
        # We use the doc_id metadata filter to find and remove all associated chunks
        request.app.state.vectordb.delete(where={"doc_id": doc_id})
        print(f"Deleted chunks for doc_id: {doc_id} from Chroma.")

        # 2. Delete from Disk
        # We search for any file starting with this doc_id (could be .pdf, .txt, .md)
        deleted_from_disk = False
        for file_path in DATA_DIR.glob(f"{doc_id}.*"):
            try:
                file_path.unlink()
                deleted_from_disk = True
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        if not deleted_from_disk:
            print(f"Warning: No file found on disk for ID {doc_id}")

        return {"message": f"Document {doc_id} deleted successfully."}
    except Exception as e:
        print(f"Error deleting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/documents/clear")
async def clear_documents(request: Request):
    """
    Clears the Chroma vector store and resets the QA chain.
    """
    print("Received request to clear documents and vector store...")
    try:
        # Clear Chroma
        request.app.state.vectordb = clear_vectorstore(request.app.state.vectordb)
        print("Chroma collection deleted and re-initialized.")
        
        # Re-initialize QA chain with the new (empty) vectordb
        request.app.state.qa_chain = get_qa_chain(request.app.state.vectordb)
        print("QA chain re-initialized.")
        
        # Delete files in DATA_DIR
        deleted_count = 0
        for file_path in DATA_DIR.glob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except PermissionError:
                    print(f"Warning: Could not delete {file_path} because it is in use.")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        print(f"Cleaned up {deleted_count} files from disk.")
        return {"message": "Success! Chroma and document storage cleared."}
    except Exception as e:
        print(f"CRITICAL ERROR during clear: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@router.get("/documents/inspect")
async def inspect_chroma(request: Request, limit: int = 10):
    """
    Returns the raw structure of chunks stored in Chroma for inspection.
    """
    try:
        # Get data from Chroma
        # We use the underlying collection to get raw data
        db = request.app.state.vectordb
        data = db.get(limit=limit, include=["documents", "metadatas"])
        
        inspection_results = []
        for i in range(len(data["ids"])):
            inspection_results.append({
                "id": data["ids"][i],
                "metadata": data["metadatas"][i],
                "content_preview": data["documents"][i][:200] + "..." if data["documents"][i] else ""
            })
            
        return {
            "total_in_db": len(db.get()["ids"]),
            "limit": limit,
            "samples": inspection_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")
