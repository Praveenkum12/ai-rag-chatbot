from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pathlib import Path
import uuid

from app.rag.ingest import load_and_split_document
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class ChatRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str
    k: int = 5


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
        request.app.state.vectordb.add_documents(chunks)

        return {
            "document_id": doc_id,
            "chunks_added": len(chunks),
            "filename": file.filename
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
    results = request.app.state.vectordb.similarity_search(
        search_request.query, 
        k=search_request.k
    )
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results
    ]
