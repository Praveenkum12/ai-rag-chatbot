# AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot implementation using ChromaDB for vector storage.

## Project Structure

```
ai-rag-chatbot/
├── app/
│   ├── main.py          # Main application entry point
│   ├── api.py           # API endpoints
│   ├── config.py        # Configuration settings
│   └── rag/
│       ├── __init__.py
│       ├── vector_store.py    # Vector store implementation
│       └── chroma_store.py    # ChromaDB integration
├── data/
│   └── documents/       # Document storage directory
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your documents to the `data/documents/` directory

3. Run the application:
   ```bash
   python app/main.py
   ```

## Features

- RAG-based question answering
- ChromaDB vector storage
- FastAPI REST API
- Document ingestion and retrieval
