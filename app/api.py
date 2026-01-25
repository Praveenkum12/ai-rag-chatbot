from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
from pydantic import BaseModel
from app.rag.chroma_store import ChromaVectorStore
from app.rag.qa import answer_question
from app.rag.chunker import chunk_text, create_chunk_records
from app.rag.embedding import store_chunks

from app.rag.loader import load_pdf, clean_text

import logging

# Configure logging to handle Unicode safely on Windows
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
try:
    vector_store = ChromaVectorStore()
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None

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

    # Load pages as objects
    pages = load_pdf(file_path)
    
    all_chunks_to_store = []
    
    for page_obj in pages:
        page_num = page_obj["page"]
        text = clean_text(page_obj["text"])
        
        # Chunk this page
        page_chunks = chunk_text(text)
        
        # Create records for this page
        for i, chunk_text_content in enumerate(page_chunks):
            all_chunks_to_store.append({
                "id": f"{doc_id}_p{page_num}_c{i}",
                "text": chunk_text_content,
                "metadata": {
                    "document_id": doc_id,
                    "filename": file.filename,
                    "page": page_num,
                    "chunk_index": i
                }
            })
    
    store_chunks(all_chunks_to_store, vector_store)

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "pages_processed": len(pages),
        "chunks_stored": len(all_chunks_to_store)
    }

@router.post("/chat")
def chat(req: QuestionRequest):
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized. Check your API keys.")
    
    try:
        result = answer_question(req.question, vector_store)
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/status")
def get_status():
    return {
        "vector_store_ready": vector_store is not None,
        "document_count": len(list(DATA_DIR.glob("*.pdf"))),
        "chroma_path": "data/chroma"
    }
