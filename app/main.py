from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
from app.api import router

app = FastAPI(title="LangChain RAG")

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
