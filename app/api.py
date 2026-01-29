from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from typing import List
from pathlib import Path
import uuid

from app.rag.ingest import load_and_split_document
from app.rag.vectorstore import clear_vectorstore
from app.rag.qa import get_qa_chain
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class ChatRequest(BaseModel):
    question: str
    doc_ids: List[str] = None
    file_type: str = None

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_ids: List[str] = None


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
async def list_documents():
    files = []
    for path in DATA_DIR.glob("*"):
        if path.is_file():
            files.append({
                "id": path.stem,
                "name": path.name,
                "type": path.suffix
            })
    return files


@router.post("/chat")
def chat(request: Request, chat_request: ChatRequest):
    # Construct filters for retrieval
    filters = {}
    if chat_request.doc_ids:
        if len(chat_request.doc_ids) == 1:
            filters["doc_id"] = chat_request.doc_ids[0]
        else:
            filters["doc_id"] = {"$in": chat_request.doc_ids}
            
    if chat_request.file_type:
        filters["file_type"] = chat_request.file_type

    # If we have filters, we must use manual retrieval
    if filters:
        print(f"Applying filters: {filters}")
        # Search manually with filters
        retriever = request.app.state.vectordb.as_retriever(
            search_kwargs={"k": 5, "filter": filters}
        )
        docs = retriever.invoke(chat_request.question)
        
        # Manually run the "Brain" part (Prompt + LLM)
        from app.rag.qa import get_llm, get_prompt
        from langchain_core.output_parsers import StrOutputParser
        
        llm = get_llm()
        prompt = get_prompt()
        answer_chain = prompt | llm | StrOutputParser()
        
        answer = answer_chain.invoke({
            "question": chat_request.question,
            "context": docs
        })
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }
    
    # Otherwise, use the optimized global chain (Global RAG)
    else:
        result = request.app.state.qa_chain.invoke(chat_request.question)
        return {
            "answer": result["answer"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result["context"]
            ]
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
