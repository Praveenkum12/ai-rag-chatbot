from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router
from contextlib import asynccontextmanager
from app.rag.vectorstore import get_vectorstore
from app.rag.qa import get_qa_chain
from app.database import engine
from app.models import Base
import os

# In main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Connecting to MySQL...")
    try:
        # Create tables inside lifespan to avoid Windows reload issues
        Base.metadata.create_all(bind=engine)
        print("Database tables verified.")
    except Exception as e:
        print(f"DATABASE ERROR: {e}")

    app.state.vectordb = get_vectorstore()
    app.state.qa_chain = get_qa_chain(app.state.vectordb)
    yield
    # Shutdown (cleanup if needed)

from app.auth_api import router as auth_router

app = FastAPI(title="LangChain RAG", lifespan=lifespan)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router)
app.include_router(auth_router)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/WHO-ARE-YOU")
def who_are_you():
    return {"message": "I am the latest Antigravity-patched server!"}
