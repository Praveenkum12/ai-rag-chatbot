from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router
from contextlib import asynccontextmanager
from app.rag.vectorstore import get_vectorstore
from app.rag.qa import get_qa_chain
import os

# In main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.vectordb = get_vectorstore()
    app.state.qa_chain = get_qa_chain(app.state.vectordb)
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(title="LangChain RAG", lifespan=lifespan)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}
