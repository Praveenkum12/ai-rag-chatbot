from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pathlib import Path
import uuid

from app.rag.ingest import load_and_split_pdf
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class ChatRequest(BaseModel):
    question: str


@router.post("/documents/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    try:
        doc_id = str(uuid.uuid4())
        file_path = DATA_DIR / f"{doc_id}.pdf"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = load_and_split_pdf(str(file_path))
        request.app.state.vectordb.add_documents(chunks)

        return {
            "document_id": doc_id,
            "chunks_added": len(chunks),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/chat")
def chat(request: Request, chat_request: ChatRequest):
    answer = request.app.state.qa_chain.invoke(chat_request.question)
    return {
        "answer": answer
    }
