from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid

from app.rag.ingest import load_and_split_pdf
from app.rag.vectorstore import get_vectorstore
from app.rag.qa import get_qa_chain

router = APIRouter()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize vector DB once
vectordb = get_vectorstore()
qa_chain = get_qa_chain(vectordb)


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    doc_id = str(uuid.uuid4())
    file_path = DATA_DIR / f"{doc_id}.pdf"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # LangChain ingestion
    chunks = load_and_split_pdf(str(file_path))
    vectordb.add_documents(chunks)
    vectordb.persist()

    return {
        "document_id": doc_id,
        "chunks_added": len(chunks)
    }


@router.post("/chat")
def chat(question: str):
    result = qa_chain.invoke(question)
    return {
        "answer": result["result"],
        "sources": [
            doc.metadata for doc in result["source_documents"]
        ]
    }
