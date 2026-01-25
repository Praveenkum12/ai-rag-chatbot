from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
from app.api import router
from contextlib import asynccontextmanager
from app.rag.vectorstore import get_vectorstore
from app.rag.qa import get_qa_chain

# In main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.vectordb = get_vectorstore()
    app.state.qa_chain = get_qa_chain(app.state.vectordb)
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(title="LangChain RAG", lifespan=lifespan)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
