from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from typing import List, Optional
from pathlib import Path
import uuid
import tiktoken

from app.rag.ingest import load_and_split_document
from app.rag.vectorstore import clear_vectorstore
from app.rag.qa import get_qa_chain
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Conversation, Message, UserMemory
from fastapi import Depends

router = APIRouter()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration for token management
MODEL_NAME = "gpt-4.1-nano" # or whatever model you use
MODEL_TOKEN_LIMIT = 3000 # Example limit for older/local models
SUMMARIZATION_THRESHOLD = int(MODEL_TOKEN_LIMIT * 0.75)

from datetime import datetime, timedelta

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None # Added for session persistence
    doc_ids: Optional[List[str]] = []
    file_type: Optional[str] = None
    date_filter: Optional[str] = None # 'today', 'week', 'any'
    history: Optional[List[dict]] = [] # [{"role": "user", "content": "..."}, ...]

def count_tokens(text: str, model: str = "gpt-4.1-nano") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

async def summarize_chat(history: List[dict]) -> str:
    """Uses LLM to summarize a segment of chat history."""
    from app.rag.qa import get_llm
    llm = get_llm()
    
    text_to_summarize = ""
    for msg in history:
        text_to_summarize += f"{msg['role'].upper()}: {msg['content']}\n"
        
    prompt = f"Extract the key facts and decisions from the following conversation history into a concise bulleted list. Keep it informative but brief:\n\n{text_to_summarize}"
    
    summary = llm.invoke(prompt)
    return summary.content

async def generate_title(first_question: str) -> str:
    """Uses LLM to generate a short, descriptive title for the conversation."""
    try:
        from app.rag.qa import get_llm
        llm = get_llm()
        
        prompt = f"Generate a very short (max 4-5 words) descriptive title for a conversation starting with this question: '{first_question}'. Respond ONLY with the title text. No quotes, no prefix like 'Title:', no period."
        
        response = llm.invoke(prompt)
        return response.content.strip().replace('"', '')
    except Exception as e:
        print(f"DEBUG: Title generation failed: {e}")
        return first_question[:40] + "..."

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_ids: List[str] = None
    file_type: Optional[str] = None
    date_filter: Optional[str] = None


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
        # Get User from Token
        auth_header = request.headers.get("Authorization")
        user_id = "guest"
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            from app.auth import decode_access_token
            payload = decode_access_token(token)
            if payload:
                user_id = str(payload.get("sub"))
        
        print(f"DEBUG: Uploading document for User: {user_id}")

        doc_id = str(uuid.uuid4())
        file_path = DATA_DIR / f"{doc_id}{file_ext}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = load_and_split_document(str(file_path))
        
        if not chunks:
            raise ValueError("No text content could be extracted from this document.")

        from datetime import datetime
        
        # Inject cleaner metadata for easier retrieval/filtering
        for chunk in chunks:
            chunk.metadata["filename"] = file.filename
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["user_id"] = user_id # LINKED TO USER
            chunk.metadata["file_type"] = file_ext.replace(".", "")
            chunk.metadata["processed_at"] = datetime.now().isoformat()
            chunk.metadata["source"] = doc_id 

        request.app.state.vectordb.add_documents(chunks)

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents")
async def list_documents(request: Request):
    """
    Returns a list of unique documents by querying the vector database's metadata.
    Filtered by the currently logged-in user.
    """
    try:
        # Get User from Token
        auth_header = request.headers.get("Authorization")
        user_id = "guest"
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            from app.auth import decode_access_token
            payload = decode_access_token(token)
            if payload:
                user_id = str(payload.get("sub"))
        
        print(f"DEBUG: Listing documents for User ID: {user_id}")

        # Query metadata filtered by user_id
        # Note: documents uploaded before the security update will be hidden
        data = request.app.state.vectordb.get(
            where={"user_id": {"$eq": user_id}}, 
            include=["metadatas"]
        )
        
        unique_docs = {}
        if data and "metadatas" in data:
            for meta in data["metadatas"]:
                d_id = meta.get("doc_id")
                fname = meta.get("filename", "Unknown")
                ftype = meta.get("file_type", "txt")
                p_at = meta.get("processed_at", "")
                
                if d_id and d_id not in unique_docs:
                    unique_docs[d_id] = {
                        "id": d_id,
                        "name": fname,
                        "type": ftype,
                        "processed_at": p_at
                    }
        
        # Sort by processed_at descending (Most Recent First)
        sorted_docs = sorted(
            unique_docs.values(), 
            key=lambda x: x.get("processed_at", ""), 
            reverse=True
        )
        return sorted_docs
    except Exception as e:
        print(f"Error listing documents: {e}")
        return []


@router.post("/chat")
async def chat(request: Request, chat_request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # 0. Get User from Token (Optional for now, but good practice)
        auth_header = request.headers.get("Authorization")
        user_id = "guest"  # Default to 'guest' if not authenticated
        
        print(f"DEBUG: Authorization header present: {bool(auth_header)}")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            print(f"DEBUG: Token extracted: {token[:20]}...")
            
            from app.auth import decode_access_token
            payload = decode_access_token(token)
            
            print(f"DEBUG: Decoded payload: {payload}")
            
            if payload:
                user_id = str(payload.get("sub"))
                print(f"DEBUG: Extracted user_id from token: {user_id}")
        
        print(f"DEBUG: Final user_id for this request: {user_id}")

        # 1. Handle Conversation Persistence
        conv_id = chat_request.conversation_id
        if not conv_id:
            # Generate a smart title using LLM
            smart_title = await generate_title(chat_request.question)
            
            # Create new conversation if none provided
            new_conv = Conversation(
                title=smart_title,
                user_id=user_id # LINKED TO LOGGED IN USER
            )
            db.add(new_conv)
            db.commit()
            db.refresh(new_conv)
            conv_id = new_conv.id
        
        # Save User Message to DB
        user_msg = Message(
            conversation_id=conv_id,
            role="user",
            content=chat_request.question,
            token_count=count_tokens(chat_request.question)
        )
        db.add(user_msg)
        db.commit()

        # 1.5. Check for "Remember This" command (flexible: with or without colon)
        question_lower = chat_request.question.lower().strip()
        print(f"DEBUG: Checking question: '{question_lower}'")
        print(f"DEBUG: Starts with 'remember this'? {question_lower.startswith('remember this')}")
        
        if question_lower.startswith("remember this"):
            print("DEBUG: Remember This command detected!")
            # Extract fact - handle both "remember this:" and "remember this"
            if ":" in chat_request.question:
                fact = chat_request.question.split(":", 1)[1].strip()
            else:
                fact = chat_request.question[len("remember this"):].strip()
            
            print(f"DEBUG: Extracted fact: '{fact}'")
            
            if fact:
                try:
                    new_memory = UserMemory(user_id=user_id, fact=fact)
                    db.add(new_memory)
                    db.commit()
                    print(f"DEBUG: Memory saved successfully for user {user_id}")
                    return {
                        "answer": f"✓ Got it! I'll remember: \"{fact}\"",
                        "conversation_id": conv_id,
                        "sources": [],
                        "token_usage": 0
                    }
                except Exception as e:
                    print(f"ERROR: Failed to save memory: {e}")
                    return {
                        "answer": f"Sorry, I couldn't save that memory. Error: {str(e)}",
                        "conversation_id": conv_id,
                        "sources": [],
                        "token_usage": 0
                    }

        # Construct filters for retrieval
        meta_filters = []
        
        # 0. User Filter (MANDATORY)
        current_identity = user_id if user_id else "guest"
        meta_filters.append({"user_id": {"$eq": current_identity}})
        print(f"DEBUG: Chatting as {current_identity}")

        # 1. Doc IDs filter (only if specific documents are selected)
        if chat_request.doc_ids and len(chat_request.doc_ids) > 0:
            if len(chat_request.doc_ids) == 1:
                meta_filters.append({"doc_id": {"$eq": chat_request.doc_ids[0]}})
            else:
                meta_filters.append({"doc_id": {"$in": chat_request.doc_ids}})
            print(f"DEBUG: Filtering to specific documents: {chat_request.doc_ids}")
        else:
            print(f"DEBUG: No specific documents selected - searching ALL user documents")
        
        
        # 2. File Type filter
        if chat_request.file_type and chat_request.file_type != "all":
            meta_filters.append({"file_type": {"$eq": chat_request.file_type}})
            
        # 3. Date filter
        if chat_request.date_filter and chat_request.date_filter != "any":
            now = datetime.now()
            if chat_request.date_filter == "today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif chat_request.date_filter == "week":
                cutoff = now - timedelta(days=7)
            
            meta_filters.append({"processed_at": {"$gte": cutoff.isoformat()}})

        # Combine filters with $and if multiple exist
        filters = None
        if len(meta_filters) == 1:
            filters = meta_filters[0]
        elif len(meta_filters) > 1:
            filters = {"$and": meta_filters}

        from app.rag.qa import get_hybrid_retriever, get_llm, get_prompt
        from langchain_core.output_parsers import StrOutputParser
        
        # 0. Check if it's a greeting (Skip retrieval to save tokens)
        clean_q = chat_request.question.lower().strip().strip('?!.')
        is_greeting = clean_q in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "yo", "hi there"]
        
        if is_greeting:
            print("DEBUG: Greeting detected. Skipping document retrieval.")
            docs = []
            filters = None
        else:
            # 1. Get Hybrid Retriever
            # Optimized: Reducing k from 5 to 3 to save tokens/cost (3+3 = max 6 docs)
            retriever = get_hybrid_retriever(
                request.app.state.vectordb, 
                search_kwargs={"k": 3, "filter": filters if filters else None}
            )
            
            print(f"DEBUG: Active filters: {filters}")
            
            # 2. Invoke retriever
            docs = retriever.invoke(chat_request.question)
            print(f"DEBUG: Retrieved {len(docs)} docs for question: '{chat_request.question}'")

        # --- RELEVANCE FILTERING REMOVED ---
        # Passing all retrieved documents to the LLM regardless of relevance score as requested.
        # ---------------------------
        
        # 3. Clean and Label Docs
        context_parts = []
        source_data = []
        
        # Get scores for the top docs in this specific search (for confidence metrics)
        scored_results = {}
        if not is_greeting and len(docs) > 0:
            try:
                scored_results = {
                    doc.page_content: score 
                    for doc, score in request.app.state.vectordb.similarity_search_with_score(
                        chat_request.question, k=10, filter=filters if filters else None
                    )
                }
            except Exception as e:
                print(f"DEBUG: Scoring failed: {e}")

        for i, doc in enumerate(docs):
            filename = doc.metadata.get("filename", "Unknown File")
            # Sanitize content for LLM
            clean_content = doc.page_content.replace("\x00", "").replace("♂", "").replace("¶", "").replace("•", "-")
            context_parts.append(f"[Source {i+1}]: From {filename}\n{clean_content}")
            
            # Calculate Percentage: 0.0 distance is 100%, 1.2+ is ~0%
            raw_score = scored_results.get(doc.page_content, 1.0)
            confidence = max(0, min(100, round((1 - (raw_score / 1.5)) * 100)))
            
            source_data.append({
                "content": doc.page_content,
                "metadata": {**doc.metadata, "confidence": confidence}
            })
        
        context_text = "\n\n".join(context_parts)

        # 4. Early return if no documents and not a greeting/tool-friendly query
        # We allow the LLM to proceed if the question mentions tools or is a greeting
        tool_keywords = [
            "weather", "time", "date", "today", "now", "where am i", "location", "place",
            "email", "mail", "send", "message", "subject", "content", "task", "todo", "reminder"
        ]
        is_tool_query = any(word in clean_q for word in tool_keywords)
        
        if not is_greeting and not is_tool_query and not context_text:
            print("DEBUG: No documents retrieved, not a greeting, and not a tool query. Returning 'I don't know.'")
            
            # Save AI Response to DB
            ai_msg = Message(
                conversation_id=conv_id,
                role="assistant",
                content="I don't know.",
                token_count=count_tokens("I don't know.")
            )
            db.add(ai_msg)
            db.commit()
            
            return {
                "answer": "I don't know.",
                "conversation_id": conv_id,
                "sources": [],
                "token_usage": 0,
                "tools_used": []
            }
        
        # 5. Run LLM
        llm = get_llm()
        
        # 5. Format History for the Prompt
        history_text = ""
        for msg in chat_request.history:
            role_raw = msg.get("role", "user")
            if role_raw == "system":
                role = "SUMMARY"
            elif role_raw == "user":
                role = "HUMAN"
            else:
                role = "AI"
            history_text += f"{role}: {msg.get('content')}\n"

        if not history_text:
            history_text = "No previous conversation."

        # 5.5 Fetch Long-Term Memories
        memories = db.query(UserMemory).filter(UserMemory.user_id == user_id).all() if user_id else []
        memory_text = "\n".join([f"- {m.fact}" for m in memories]) if memories else "No previous memories stored."

        print(f"DEBUG: Context length: {len(context_text)}")
        
        # 6. Use OpenAI with Tools
        from openai import OpenAI
        from app.tools import AVAILABLE_TOOLS, execute_function
        import json
        
        # Build messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": f"""You are a Knowledge Assistant. Use the provided context to answer questions accurately.

PRIMARY DIRECTIVE:
1. **TOOLS FIRST**: If the user asks about weather, time, or location, ALWAYS use the relevant tool immediately.
2. **KNOWLEDGE BASE**: Use the "Context from knowledge base" section to answer. If the information is present (e.g., in a resume), synthesize an answer even if not explicitly stated as a fact.
3. **STRICT DISCLOSURE**: Only if the answer is completely missing from the documents AND no tool can answer it, say "I don't know."

SPECIFIC RULES:
- **Weather** → Use `get_weather`.
- **Time/Date** → Use `get_datetime`.
- **Location** → Use `get_user_location`.
- **Greetings** → Respond warmly.
- **Identity** → If the context contains a person's name (like a resume), use that information to describe who they are.

Context from knowledge base:
{context_text if context_text else "No documents available"}

Chat History:
{history_text}

User Memories:
{memory_text}

Question: {chat_request.question}"""
            }
        ]
        
        # Add current question
        messages.append({
            "role": "user",
            "content": chat_request.question
        })
        
        # Initialize OpenAI client
        client = OpenAI()
        
        # First API call with tools
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto"
        )
        
        # Initial tool check
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # Track tool usage and iterations
        tools_used = []
        max_iterations = 5
        current_iteration = 0
        total_tokens = response.usage.total_tokens
        answer = response_message.content or ""

        # Function Chaining Loop
        while tool_calls and current_iteration < max_iterations:
            current_iteration += 1
            print(f"DEBUG: Processing tool call iteration {current_iteration}")
            
            # Add the assistant's response (containing tool_calls) to messages
            messages.append(response_message)
            
            # Execute each tool call in the current turn
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"DEBUG: Calling {function_name} with args: {function_args}")
                tools_used.append({"name": function_name, "args": function_args})
                
                # Execute the function
                function_response = await execute_function(function_name, function_args)
                
                # Include tool usage in sources for citation
                source_data.append({
                    "content": function_response,
                    "metadata": {
                        "filename": f"Tool: {function_name}",
                        "doc_id": f"tool_{function_name}",
                        "confidence": 100,
                        "type": "tool"
                    }
                })
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
            
            # Call AI again with tool results to see if more tools are needed
            next_response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto"
            )
            
            response_message = next_response.choices[0].message
            tool_calls = response_message.tool_calls
            total_tokens += next_response.usage.total_tokens
            answer = response_message.content or ""
            
            # If no more tool calls, we are done
            if not tool_calls:
                break
        else:
            # If loop ends or wasn't entered
            if not answer and response_message.content:
                answer = response_message.content
            if current_iteration >= max_iterations:
                print("DEBUG: Max tool iterations reached. Protecting from infinite loop.")
        
        # 7. Calculate Tokens for tracking
        print(f"DEBUG: Token usage for this request: {total_tokens} (Threshold: {SUMMARIZATION_THRESHOLD})")

        # 8. Sliding Window Logic (75% Rule)
        # If tokens are high, summarize the older parts for the next turn
        new_summary = None
        if total_tokens > SUMMARIZATION_THRESHOLD and len(chat_request.history) > 10:
            print(f"DEBUG: Tokens exceeded 75% of limit ({total_tokens}/{MODEL_TOKEN_LIMIT}). Summarizing...")
            
            # Keep the last 10 messages as "fresh" context
            to_summarize = chat_request.history[:-10] 
            new_summary = await summarize_chat(to_summarize)
            print(f"DEBUG: Generated Summary for older context: {new_summary}")

        # Hide sources if it's a greeting OR if the AI says "I don't know"
        hide_sources = is_greeting or (answer and "i don't know" in answer.lower())

        # Save AI Response to DB
        ai_msg = Message(
            conversation_id=conv_id,
            role="assistant",
            content=answer,
            token_count=count_tokens(answer)
        )
        db.add(ai_msg)
        db.commit()

        return {
            "answer": answer,
            "conversation_id": conv_id, # Return the ID so frontend can persist session
            "sources": source_data if not hide_sources and docs else [],
            "token_usage": total_tokens,
            "new_summary": new_summary,
            "tools_used": tools_used  # Add tools_used to response
        }
    except Exception as e:
        print(f"CRITICAL CHAT ERROR: {str(e)}")
        # Try to get conv_id if possible, otherwise None is okay
        try:
            current_conv_id = conv_id if 'conv_id' in locals() else chat_request.conversation_id
        except:
            current_conv_id = None
            
        return {
            "answer": "I don't know.",
            "conversation_id": current_conv_id,
            "sources": [],
            "token_usage": 0
        }



@router.post("/search")
def search(request: Request, search_request: SearchRequest):
    filter_dict = None
    if search_request.doc_ids:
        if len(search_request.doc_ids) == 1:
            filter_dict = {"doc_id": search_request.doc_ids[0]}
        else:
            filter_dict = {"doc_id": {"$in": search_request.doc_ids}}

    results = request.app.state.vectordb.similarity_search(
        search_request.query, 
        k=search_request.k,
        filter=filter_dict
    )
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    """
    Deletes a single document from both Chroma and the disk.
    Protected: Only allows deleting if it belongs to the user.
    """
    try:
        # Get User from Token
        auth_header = request.headers.get("Authorization")
        user_id = "guest"
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            from app.auth import decode_access_token
            payload = decode_access_token(token)
            if payload:
                user_id = str(payload.get("sub"))
        
        print(f"DEBUG: Delete request from user_id: {user_id} for doc: {doc_id}")

        # 1. Delete from Chroma (with user_id filter for safety)
        request.app.state.vectordb.delete(where={"$and": [{"doc_id": doc_id}, {"user_id": user_id}]})
        print(f"Deleted chunks for doc_id: {doc_id} (User: {user_id}) from Chroma.")

        # 2. Delete from Disk
        # (Disk cleanup remains global but doc_id is unique, 
        # but technically we should verify user owns the file before unlinking)
        # For simplicity, since doc_id is random UUID, collisions are unlikely.
        
        deleted_from_disk = False
        for file_path in DATA_DIR.glob(f"{doc_id}.*"):
            try:
                file_path.unlink()
                deleted_from_disk = True
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        return {"message": f"Document {doc_id} deleted successfully."}
    except Exception as e:
        print(f"Error deleting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/documents/clear")
async def clear_documents(request: Request):
    """
    Clears ONLY the current user's documents from the vector store.
    """
    try:
        # Get User from Token
        auth_header = request.headers.get("Authorization")
        user_id = "guest"
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            from app.auth import decode_access_token
            payload = decode_access_token(token)
            if payload:
                user_id = str(payload.get("sub"))
        
        print(f"DEBUG: Clear request from user_id: {user_id}")

        # Clear just this user's data from Chroma
        request.app.state.vectordb.delete(where={"user_id": user_id})
        print(f"Cleared vectordb for user: {user_id}")
        
        # Disk cleanup is more complex since we'd need to know which files belong to whom.
        # For now, we'll focus on the Vector store partitioning.
        
        return {"message": f"Success! Your private document library has been cleared."}
    except Exception as e:
        print(f"CRITICAL ERROR during clear: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@router.get("/conversations")
async def list_conversations(request: Request, db: Session = Depends(get_db)):
    """Returns a list of all chat sessions for the current user."""
    # Get User from Token
    auth_header = request.headers.get("Authorization")
    user_id = "guest"
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        from app.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            user_id = str(payload.get("sub"))
    
    print(f"DEBUG: List conversations for user_id: {user_id}")

    return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()).all()

@router.get("/conversations-search")
async def search_conversations(q: str, request: Request, db: Session = Depends(get_db)):
    """Searches conversations by title or message content."""
    auth_header = request.headers.get("Authorization")
    user_id = "guest"
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        from app.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            user_id = str(payload.get("sub"))
    
    # Search in titles
    conv_query = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.title.like(f"%{q}%")
    )
    
    # Search in message contents
    msg_query = db.query(Conversation).join(Message).filter(
        Conversation.user_id == user_id,
        Message.content.like(f"%{q}%")
    )
    
    # Combine and deduplicate
    results = conv_query.union(msg_query).order_by(Conversation.updated_at.desc()).all()
    
    return results

@router.get("/memories")
async def get_memories(request: Request, db: Session = Depends(get_db)):
    """Returns all memories for the logged-in user."""
    auth_header = request.headers.get("Authorization")
    user_id = "guest"
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        from app.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            user_id = str(payload.get("sub"))
            
    return db.query(UserMemory).filter(UserMemory.user_id == user_id).all()

@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request, db: Session = Depends(get_db)):
    """Deletes a specific memory."""
    auth_header = request.headers.get("Authorization")
    user_id = "guest"
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        from app.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            user_id = str(payload.get("sub"))
            
    db.query(UserMemory).filter(UserMemory.id == memory_id, UserMemory.user_id == user_id).delete()
    db.commit()
    return {"message": "Memory deleted"}

@router.get("/analytics")
async def get_analytics(request: Request, db: Session = Depends(get_db)):
    """Calculates chat and memory statistics."""
    auth_header = request.headers.get("Authorization")
    user_id = "guest"
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        from app.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            user_id = str(payload.get("sub"))

    # Stats calculation
    total_convs = db.query(Conversation).filter(Conversation.user_id == user_id).count()
    total_msgs = db.query(Message).join(Conversation).filter(Conversation.user_id == user_id).count()
    total_memories = db.query(UserMemory).filter(UserMemory.user_id == user_id).count()
    
    avg_msgs = round(total_msgs / total_convs, 1) if total_convs > 0 else 0
    
    # Get top 5 recent conversations (ordered by recency like history)
    from sqlalchemy import func
    top_chats = db.query(Conversation.title, func.count(Message.id).label('msg_count'))\
        .join(Message).filter(Conversation.user_id == user_id)\
        .group_by(Conversation.id, Conversation.updated_at)\
        .order_by(Conversation.updated_at.desc()).limit(5).all()

    return {
        "total_conversations": total_convs,
        "total_messages": total_msgs,
        "total_memories": total_memories,
        "avg_messages_per_chat": avg_msgs,
        "top_conversations": [{"title": c[0], "count": c[1]} for c in top_chats]
    }

@router.post("/clear-chat-history")
async def clear_all_conversations(request: Request, db: Session = Depends(get_db)):
    """Deletes all chat history for the current user."""
    # Get User from Token
    auth_header = request.headers.get("Authorization")
    user_id = "guest"
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        from app.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            user_id = str(payload.get("sub"))
    
    print(f"DEBUG: ATTEMPTING TO CLEAR ALL HISTORY FOR USER: {user_id}")
    
    # Get all conversation IDs for this user
    user_convs = db.query(Conversation.id).filter(Conversation.user_id == user_id).all()
    conv_ids = [c[0] for c in user_convs]
    
    print(f"DEBUG: Found {len(conv_ids)} conversations to delete.")
    
    if conv_ids:
        # Delete messages in batches or all together
        db.query(Message).filter(Message.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
        # Delete conversations
        db.query(Conversation).filter(Conversation.user_id == user_id).delete(synchronize_session=False)
        db.commit()
    
    return {"message": "All chat history cleared"}

@router.get("/conversations/{conv_id}")
async def get_conversation_history(conv_id: str, db: Session = Depends(get_db)):
    """Returns the message history for a specific conversation ID."""
    messages = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.created_at.asc()).all()
    return [{
        "role": msg.role,
        "content": msg.content
    } for msg in messages]

@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, db: Session = Depends(get_db)):
    """Deletes a conversation and all its messages."""
    db.query(Message).filter(Message.conversation_id == conv_id).delete()
    db.query(Conversation).filter(Conversation.id == conv_id).delete()
    db.commit()
    return {"message": "Conversation deleted"}

@router.get("/conversations/{conv_id}/export")
async def export_conversation(conv_id: str, db: Session = Depends(get_db)):
    """Exports the conversation history as a text file."""
    try:
        print(f"DEBUG: Export requested for conversation: {conv_id}")
        # Fetch the conversation title
        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
        if not conv:
            print(f"DEBUG: Conversation {conv_id} not found in DB")
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        messages = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.created_at.asc()).all()
        print(f"DEBUG: Found {len(messages)} messages for export")
        
        export_content = f"# Chat Export: {conv.title}\n"
        export_content += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_content += "-" * 40 + "\n\n"
        
        for msg in messages:
            role = "Human" if msg.role == "user" else "AI Assistant"
            export_content += f"[{role}]:\n{msg.content}\n\n"
            
        filename = f"chat_export_{conv_id[:8]}.txt"
        
        return PlainTextResponse(
            content=export_content,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/documents/inspect")
async def inspect_chroma(request: Request, limit: int = 10):
    """
    Returns the raw structure of chunks stored in Chroma for inspection.
    """
    try:
        # Get data from Chroma
        # We use the underlying collection to get raw data
        db = request.app.state.vectordb
        data = db.get(limit=limit, include=["documents", "metadatas"])
        
        inspection_results = []
        for i in range(len(data["ids"])):
            inspection_results.append({
                "id": data["ids"][i],
                "metadata": data["metadatas"][i],
                "content_preview": data["documents"][i][:200] + "..." if data["documents"][i] else ""
            })
            
        return {
            "total_in_db": len(db.get()["ids"]),
            "limit": limit,
            "samples": inspection_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")
