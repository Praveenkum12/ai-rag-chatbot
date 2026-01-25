from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
from pydantic import BaseModel
from app.rag.chroma_store import ChromaVectorStore
from app.rag.qa import answer_question

from app.rag.loader import load_pdf, clean_text

router = APIRouter()
vector_store = ChromaVectorStore()

class QuestionRequest(BaseModel):
    question: str

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    doc_id = str(uuid.uuid4())
    file_path = DATA_DIR / f"{doc_id}.pdf"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    raw_text = load_pdf(file_path)
    cleaned_text = clean_text(raw_text)

    return {
        "document_id": doc_id,
        "characters": len(cleaned_text),
        "preview": cleaned_text[:500]
    }

@router.post("/chat")
def chat(req: QuestionRequest):
    result = answer_question(req.question, vector_store)
    return result
