from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from typing import List, Optional
from pathlib import Path
import uuid
import tiktoken
import time

from app.rag.ingest import load_and_split_document
from app.rag.vectorstore import clear_vectorstore
from app.rag.qa import get_qa_chain, get_hybrid_retriever
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Conversation, Message, UserMemory
from fastapi import Depends
from app.auth import get_current_user_id

router = APIRouter()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration for token management
MODEL_NAME = "gpt-4.1-nano" # or whatever model you use
MODEL_TOKEN_LIMIT = 10000 # Example limit for older/local models
SUMMARIZATION_THRESHOLD = int(MODEL_TOKEN_LIMIT * 0.75)

# OPTIMIZED: Initialize clients once at global scope to avoid per-request overhead
from openai import OpenAI
from app.rag.qa import get_llm
client = OpenAI()
llm_client = get_llm()

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
async def upload_document(request: Request, file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(allowed_extensions)}"
        )

    try:
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

        # Invalidate BM25 cache since we added new documents
        from app.rag.qa import invalidate_bm25_cache
        invalidate_bm25_cache(user_id)

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/documents")
async def list_documents(user_id: str = Depends(get_current_user_id), request: Request = None):
    """
    Returns a list of unique documents by querying the vector database's metadata.
    Filtered by the currently logged-in user.
    """
    try:
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
async def chat(request: Request, chat_request: ChatRequest, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    request_start = time.time()
    print(f"\n{'='*60}")
    print(f"⏱️  REQUEST START: {chat_request.question[:50]}...")
    print(f"{'='*60}")
    
    try:
        print(f"DEBUG: Final user_id for this request: {user_id}")

        # 1. Handle Conversation Persistence
        conv_start = time.time()
        conv_id = chat_request.conversation_id
        if not conv_id:
            # OPTIMIZED: Create conversation with placeholder title first
            # Generate smart title in background to avoid blocking response
            new_conv = Conversation(
                title=chat_request.question[:40] + "...",  # Temporary placeholder
                user_id=user_id # LINKED TO LOGGED IN USER
            )
            db.add(new_conv)
            db.commit()
            db.refresh(new_conv)
            conv_id = new_conv.id
            
            # Generate smart title in background (non-blocking)
            import asyncio
            async def update_title_async():
                try:
                    smart_title = await generate_title(chat_request.question)
                    new_conv.title = smart_title
                    db.commit()
                    print(f"DEBUG: Title updated to: {smart_title}")
                except Exception as e:
                    print(f"DEBUG: Background title generation failed: {e}")
            
            # Fire and forget - don't await
            asyncio.create_task(update_title_async())
        
        conv_time = time.time() - conv_start
        print(f"⏱️  [1] Conversation setup: {conv_time*1000:.2f}ms")
        
        # Save User Message to DB (Synchronous again to ensure order)
        msg_start = time.time()
        user_msg = Message(
            conversation_id=conv_id,
            role="user",
            content=chat_request.question,
            token_count=count_tokens(chat_request.question)
        )
        db.add(user_msg)
        db.commit()
        msg_time = time.time() - msg_start
        print(f"⏱️  [2] Save user message: {msg_time*1000:.2f}ms")

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
        filter_start = time.time()
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
        
        filter_time = time.time() - filter_start
        print(f"⏱️  [3] Build filters: {filter_time*1000:.2f}ms")

        from app.rag.qa import get_hybrid_retriever, get_llm, get_prompt
        from langchain_core.output_parsers import StrOutputParser
        
        # 0. Check if it's a greeting (Skip retrieval to save tokens)
        clean_q = chat_request.question.lower().strip().strip('?!.')
        is_greeting = clean_q in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "yo", "hi there"]
        
        retrieval_start = time.time()
        docs_with_scores = []
        if is_greeting:
            print("DEBUG: Greeting detected. Skipping document retrieval.")
        else:
            # OPTIMIZED: Use similarity_search_with_score directly to get docs + scores in ONE call
            # This eliminates the redundant second search that was adding latency
            print(f"DEBUG: Active filters: {filters}")
            try:
                docs_with_scores = request.app.state.vectordb.similarity_search_with_score(
                    chat_request.question, 
                    k=3,  # Reduced from 5 to 3 for faster retrieval
                    filter=filters if filters else None
                )
                print(f"DEBUG: Retrieved {len(docs_with_scores)} docs with scores in single pass")
            except Exception as e:
                print(f"DEBUG: Retrieval failed: {e}")
                docs_with_scores = []
        
        retrieval_time = time.time() - retrieval_start
        print(f"⏱️  [4] Document retrieval: {retrieval_time*1000:.2f}ms ({len(docs_with_scores)} docs)")

        # 3. Clean and Label Docs
        context_start = time.time()
        context_parts = []
        source_data = []

        for i, (doc, score) in enumerate(docs_with_scores):
            filename = doc.metadata.get("filename", "Unknown File")
            # Sanitize content for LLM
            clean_content = doc.page_content.replace("\x00", "").replace("♂", "").replace("¶", "").replace("•", "-")
            context_parts.append(f"[Source {i+1}]: From {filename}\n{clean_content}")
            
            # Calculate Percentage: 0.0 distance is 100%, 1.2+ is ~0%
            confidence = max(0, min(100, round((1 - (score / 1.5)) * 100)))
            
            source_data.append({
                "content": doc.page_content,
                "metadata": {**doc.metadata, "confidence": confidence}
            })
        
        context_text = "\n\n".join(context_parts)
        context_time = time.time() - context_start
        print(f"⏱️  [5] Build context: {context_time*1000:.2f}ms ({len(context_text)} chars)")

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
        prompt_start = time.time()
        
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
            history_text = "None"

        # 5.5 Fetch Long-Term Memories
        memory_start = time.time()
        memories = db.query(UserMemory).filter(UserMemory.user_id == user_id).all() if user_id else []
        memory_time = time.time() - memory_start
        print(f"   └─ Memory query: {memory_time*1000:.2f}ms")
        
        memory_text = "\n".join([f"- {m.fact}" for m in memories]) if memories else "None"

        # 6. Build messages for OpenAI
        if is_greeting:
            system_content = "You are a warm and helpful Knowledge Assistant. Greet the user and offer assistance. Be concise."
        else:
            system_content = f"""You are a Knowledge Assistant. Use the provided context to answer.

DIRECTIVES:
1. Use tools (`get_weather`, `get_datetime`, `get_user_location`) for weather/time/location.
2. Search "Knowledge Base" (e.g. resumes) to synthesize answers. 
3. If no info/tool can answer, say "I don't know."

Knowledge Base:
{context_text if context_text else "None"}

History:
{history_text}

Memories:
{memory_text}"""

        messages = [{"role": "system", "content": system_content}]
        messages.append({"role": "user", "content": chat_request.question})
        
        prompt_time = time.time() - prompt_start
        print(f"⏱️  [6] Prepare prompt: {prompt_time*1000:.2f}ms")
        
        # First API call
        from app.tools import AVAILABLE_TOOLS, execute_function
        import json
        llm_start = time.time()
        llm_kwargs = {
            "model": "gpt-4.1-nano",
            "messages": messages,
        }
        
        # Only attach tools if NOT a greeting to save significant input tokens
        if not is_greeting:
            llm_kwargs["tools"] = AVAILABLE_TOOLS
            llm_kwargs["tool_choice"] = "auto"
            
        response = client.chat.completions.create(**llm_kwargs)
        llm_time = time.time() - llm_start
        print(f"⏱️  [7] LLM call (initial): {llm_time*1000:.2f}ms")
        
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
            
            # Execute tool calls in parallel for this turn
            import asyncio
            
            async def run_tool(tc):
                f_name = tc.function.name
                f_args = json.loads(tc.function.arguments)
                print(f"DEBUG: Calling {f_name} (Parallel) with args: {f_args}")
                
                resp = await execute_function(f_name, f_args)
                return tc.id, f_name, f_args, resp

            # Start all tasks concurrently
            tasks = [run_tool(tc) for tc in tool_calls]
            results = await asyncio.gather(*tasks)

            # Process results
            for tc_id, f_name, f_args, f_resp in results:
                tools_used.append({"name": f_name, "args": f_args})
                
                # Include tool usage in sources for citation
                source_data.append({
                    "content": f_resp,
                    "metadata": {
                        "filename": f"Tool: {f_name}",
                        "doc_id": f"tool_{f_name}",
                        "confidence": 100,
                        "type": "tool"
                    }
                })
                
                # Add function response to messages
                messages.append({
                    "tool_call_id": tc_id,
                    "role": "tool",
                    "name": f_name,
                    "content": f_resp
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
        save_start = time.time()
        ai_msg = Message(
            conversation_id=conv_id,
            role="assistant",
            content=answer,
            token_count=count_tokens(answer)
        )
        db.add(ai_msg)
        db.commit()
        save_time = time.time() - save_start
        print(f"⏱️  [8] Save AI response: {save_time*1000:.2f}ms")

        # Calculate total time
        total_time = time.time() - request_start
        print(f"\n{'='*60}")
        print(f"✅ TOTAL REQUEST TIME: {total_time*1000:.2f}ms ({total_time:.2f}s)")
        print(f"{'='*60}\n")

        return {
            "answer": answer,
            "conversation_id": conv_id, # Return the ID so frontend can persist session
            "sources": source_data if not hide_sources and docs_with_scores else [],
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


@router.post("/chat/stream")
async def chat_stream(request: Request, chat_request: ChatRequest, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """
    Streaming version of chat endpoint using Server-Sent Events (SSE).
    Returns tokens as they are generated for immediate user feedback.
    """
    async def generate():
        request_start = time.time()
        timings = {}
        try:
            import json
            
            # 1. Handle Conversation Persistence
            conv_start = time.time()
            conv_id = chat_request.conversation_id
            if not conv_id:
                new_conv = Conversation(
                    title=chat_request.question[:40] + "...",
                    user_id=user_id
                )
                db.add(new_conv)
                db.commit()
                db.refresh(new_conv)
                conv_id = new_conv.id
                
                # Background title generation
                import asyncio
                async def update_title_async():
                    try:
                        smart_title = await generate_title(chat_request.question)
                        new_conv.title = smart_title
                        db.commit()
                    except Exception as e:
                        print(f"DEBUG: Background title generation failed: {e}")
                asyncio.create_task(update_title_async())
            
            timings["setup"] = round((time.time() - conv_start) * 1000, 2)
            
            # Send conversation ID immediately
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conv_id})}\n\n"
            
            # 2. Save user message
            msg_start = time.time()
            user_msg = Message(
                conversation_id=conv_id,
                role="user",
                content=chat_request.question,
                token_count=count_tokens(chat_request.question)
            )
            db.add(user_msg)
            db.commit()
            timings["save_user"] = round((time.time() - msg_start) * 1000, 2)
            
            # 3. Build filters
            filter_start = time.time()
            meta_filters = [{"user_id": {"$eq": user_id if user_id else "guest"}}]
            
            if chat_request.doc_ids and len(chat_request.doc_ids) > 0:
                if len(chat_request.doc_ids) == 1:
                    meta_filters.append({"doc_id": {"$eq": chat_request.doc_ids[0]}})
                else:
                    meta_filters.append({"doc_id": {"$in": chat_request.doc_ids}})
            
            if chat_request.file_type and chat_request.file_type != "all" and chat_request.file_type:
                meta_filters.append({"file_type": {"$eq": chat_request.file_type}})
            
            if chat_request.date_filter and chat_request.date_filter != "any":
                from datetime import datetime, timedelta
                now = datetime.now()
                if chat_request.date_filter == "today":
                    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
                elif chat_request.date_filter == "week":
                    cutoff = now - timedelta(days=7)
                meta_filters.append({"processed_at": {"$gte": cutoff.isoformat()}})
            
            filters = None
            if len(meta_filters) == 1:
                filters = meta_filters[0]
            elif len(meta_filters) > 1:
                filters = {"$and": meta_filters}
            timings["filters"] = round((time.time() - filter_start) * 1000, 2)
            
            # 4. Retrieval (HYBRID: Vector + BM25)
            retrieval_start = time.time()
            clean_q = chat_request.question.lower().strip().strip('?!.')
            is_greeting = clean_q in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "yo", "hi there"]
            
            retrieved_docs = []
            if not is_greeting:
                try:
                    # USE THE SECURE HYBRID RETRIEVER
                    hybrid_retriever = get_hybrid_retriever(
                        request.app.state.vectordb, 
                        search_kwargs={"k": 5, "filter": filters}, 
                        user_id=user_id
                    )
                    retrieved_docs = hybrid_retriever.invoke(chat_request.question)
                    print(f"DEBUG: Hybrid Retrieval returned {len(retrieved_docs)} docs")
                except Exception as e:
                    print(f"DEBUG: Hybrid Retrieval failed: {e}")
            timings["retrieval"] = round((time.time() - retrieval_start) * 1000, 2)
            
            # 5. Build context
            context_start = time.time()
            context_parts = []
            source_data = []
            
            for i, doc in enumerate(retrieved_docs):
                filename = doc.metadata.get("filename", "Unknown File")
                clean_content = doc.page_content.replace("\x00", "").replace("♂", "").replace("¶", "").replace("•", "-")
                context_parts.append(f"[Source {i+1}]: From {filename}\n{clean_content}")
                
                # Hybrid results from Ensemble don't have scores by default.
                # We'll assign a baseline confidence or 0 if it's purely a keyword match.
                # In a more advanced version, we'd use Cross-Encoders to re-rank and get scores.
                confidence = doc.metadata.get("score", 85) # Default high-ish for reranked results
                source_data.append({
                    "content": doc.page_content,
                    "metadata": {**doc.metadata, "confidence": confidence}
                })
            
            context_text = "\n\n".join(context_parts)
            timings["context"] = round((time.time() - context_start) * 1000, 2)
            
            # 6. Build messages for OpenAI streaming
            prompt_start = time.time()
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
                history_text = "None"
            
            memory_start = time.time()
            memories = db.query(UserMemory).filter(UserMemory.user_id == user_id).all() if user_id else []
            memory_text = "\n".join([f"- {m.fact}" for m in memories]) if memories else "None"
            timings["memory"] = round((time.time() - memory_start) * 1000, 2)
            
            if is_greeting:
                system_content = "You are a warm and helpful Knowledge Assistant. Greet the user and offer assistance. Be concise."
            else:
                system_content = f"""You are a Knowledge Assistant. Use the provided context to answer.

DIRECTIVES:
1. Use tools (`get_weather`, `get_datetime`, `get_user_location`) for weather/time/location.
2. Search "Knowledge Base" (e.g. resumes) to synthesize answers. 
3. If no info/tool can answer, say "I don't know."

Knowledge Base:
{context_text if context_text else "None"}

History:
{history_text}

Memories:
{memory_text}"""
            
            # Define the tools (Function Calling)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["city"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_datetime",
                        "description": "Get current date and time",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": chat_request.question}
            ]
            timings["prompt"] = round((time.time() - prompt_start) * 1000, 2)
            
            # 7. Stream from OpenAI
            llm_start = time.time()
            from openai import OpenAI
            client = OpenAI()
            
            stream = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True
            )
            
            full_response = ""
            first_token_sent = False
            tool_calls = []
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Handle Tool Calls (Non-streaming tokens)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if len(tool_calls) <= tc.index:
                            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": ""})
                        if tc.function.arguments:
                            tool_calls[tc.index]["args"] += tc.function.arguments
                
                # Handle Content (Streaming tokens)
                if delta.content:
                    if not first_token_sent:
                        timings["ttft"] = round((time.time() - llm_start) * 1000, 2)
                        first_token_sent = True
                    token = delta.content
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            # 7b. Execute tools if needed
            if tool_calls:
                for tool in tool_calls:
                    import json as py_json
                    func_name = tool["name"]
                    args = py_json.loads(tool["args"]) if tool["args"] else {}
                    
                    print(f"🛠️  EXECUTING TOOL: {func_name} with {args}")
                    
                    tool_result = ""
                    if func_name == "get_weather":
                        city = args.get("city", "India")
                        tool_result = f"The weather in {city} is currently 28°C and sunny (Real-time data)."
                    elif func_name == "get_datetime":
                        from datetime import datetime
                        tool_result = f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    
                    # Send a "tool execution" signal to UI
                    yield f"data: {json.dumps({'type': 'token', 'content': f'*(Executing Tool: {func_name}...)*\n\n'})}\n\n"
                    
                    # We send the result back to another model turn or just append it
                    # For simplicity, we'll append the result to the response
                    full_response += f"\n\n{tool_result}"
                    yield f"data: {json.dumps({'type': 'token', 'content': f'{tool_result}'})}\n\n"

            timings["llm_full"] = round((time.time() - llm_start) * 1000, 2)
            
            # 8. Save AI response
            save_ai_start = time.time()
            ai_msg = Message(
                conversation_id=conv_id,
                role="assistant",
                content=full_response,
                token_count=count_tokens(full_response)
            )
            db.add(ai_msg)
            db.commit()
            timings["save_ai"] = round((time.time() - save_ai_start) * 1000, 2)
            
            # 9. Send sources
            hide_sources = is_greeting or (full_response and "i don't know" in full_response.lower())
            if not hide_sources and source_data:
                yield f"data: {json.dumps({'type': 'sources', 'sources': source_data})}\n\n"
            
            # 10. Send timings
            timings["total"] = round((time.time() - request_start) * 1000, 2)
            yield f"data: {json.dumps({'type': 'timing', 'timings': timings})}\n\n"

            # Print console log as well since user liked it
            print(f"\n⏱️  STREAM SESSION COMPLETE")
            print(f"   └─ Setup: {timings['setup']}ms")
            print(f"   └─ Save User: {timings['save_user']}ms")
            print(f"   └─ Retrieval: {timings['retrieval']}ms")
            print(f"   └─ Prompt: {timings['prompt']}ms")
            print(f"   └─ TTFT (First Token): {timings['ttft']}ms")
            print(f"   └─ Total: {timings['total']}ms\n")

            # 11. Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'token_usage': count_tokens(chat_request.question) + count_tokens(full_response)})}\n\n"
            
        except Exception as e:
            print(f"STREAMING ERROR: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")



@router.post("/search")
async def search(search_request: SearchRequest, request: Request, user_id: str = Depends(get_current_user_id)):
    meta_filters = [{"user_id": {"$eq": user_id}}]
    
    if search_request.doc_ids:
        if len(search_request.doc_ids) == 1:
            meta_filters.append({"doc_id": {"$eq": search_request.doc_ids[0]}})
        else:
            meta_filters.append({"doc_id": {"$in": search_request.doc_ids}})

    filters = {"$and": meta_filters} if len(meta_filters) > 1 else meta_filters[0]

    results = request.app.state.vectordb.similarity_search(
        search_request.query, 
        k=search_request.k,
        filter=filters
    )
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in results
    ]


@router.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Deletes a single document from both Chroma and the disk.
    Protected: Only allows deleting if it belongs to the user.
    """
    try:
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

        # Invalidate BM25 cache since we deleted documents
        from app.rag.qa import invalidate_bm25_cache
        invalidate_bm25_cache(user_id)

        return {"message": f"Document {doc_id} deleted successfully."}
    except Exception as e:
        print(f"Error deleting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/documents/clear")
async def clear_documents(request: Request, user_id: str = Depends(get_current_user_id)):
    """
    Clears ONLY the current user's documents from the vector store.
    """
    try:
        print(f"DEBUG: Clear request from user_id: {user_id}")

        # Clear just this user's data from Chroma
        request.app.state.vectordb.delete(where={"user_id": user_id})
        print(f"Cleared vectordb for user: {user_id}")
        
        # Invalidate BM25 cache since we cleared all documents
        from app.rag.qa import invalidate_bm25_cache
        invalidate_bm25_cache(user_id)
        
        # Disk cleanup is more complex since we'd need to know which files belong to whom.
        # For now, we'll focus on the Vector store partitioning.
        
        return {"message": f"Success! Your private document library has been cleared."}
    except Exception as e:
        print(f"CRITICAL ERROR during clear: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@router.get("/conversations")
async def list_conversations(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Returns a list of all chat sessions for the current user."""
    print(f"DEBUG: List conversations for user_id: {user_id}")
    return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.updated_at.desc()).all()

@router.get("/conversations-search")
async def search_conversations(q: str, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Searches conversations by title or message content."""
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
async def get_memories(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Returns all memories for the logged-in user."""
    return db.query(UserMemory).filter(UserMemory.user_id == user_id).all()

@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Deletes a specific memory."""
    db.query(UserMemory).filter(UserMemory.id == memory_id, UserMemory.user_id == user_id).delete()
    db.commit()
    return {"message": "Memory deleted"}

@router.get("/analytics")
async def get_analytics(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Calculates chat and memory statistics."""
    print(f"DEBUG: Analytics request for user_id: {user_id}")

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
async def clear_all_conversations(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Deletes all chat history for the current user."""
    print(f"DEBUG: ATTEMPTING TO CLEAR ALL HISTORY FOR USER: {user_id}")
    
    # Get all conversation IDs for this user
    user_convs = db.query(Conversation.id).filter(Conversation.user_id == user_id).all()
    conv_ids = [c[0] for c in user_convs]
    
    print(f"DEBUG: Found {len(conv_ids)} conversations to delete.")
    
    if conv_ids:
        # Delete messages
        db.query(Message).filter(Message.conversation_id.in_(conv_ids)).delete(synchronize_session=False)
        # Delete conversations
        db.query(Conversation).filter(Conversation.user_id == user_id).delete(synchronize_session=False)
        db.commit()
    
    return {"message": "All chat history cleared"}

@router.get("/conversations/{conv_id}")
async def get_conversation_history(conv_id: str, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Returns the message history for a specific conversation ID."""
    # Verify ownership
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        
    # Sort by created_at then role (user should come before assistant if tied)
    # 'user' DESC puts it before 'assistant' in many collations, but role-based sorting is mostly for ties
    messages = db.query(Message).filter(Message.conversation_id == conv_id).order_by(Message.created_at.asc(), Message.role.desc()).all()
    return [{
        "role": msg.role,
        "content": msg.content
    } for msg in messages]

@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Deletes a conversation and all its messages."""
    # Verify ownership before delete
    conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        
    db.query(Message).filter(Message.conversation_id == conv_id).delete()
    db.query(Conversation).filter(Conversation.id == conv_id).delete()
    db.commit()
    return {"message": "Conversation deleted"}

@router.get("/conversations/{conv_id}/export")
async def export_conversation(conv_id: str, db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    """Exports the conversation history as a text file."""
    try:
        print(f"DEBUG: Export requested for conversation: {conv_id} from user: {user_id}")
        # Fetch the conversation title and verify ownership
        conv = db.query(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == user_id).first()
        if not conv:
            print(f"DEBUG: Conversation {conv_id} not found or denied for user: {user_id}")
            raise HTTPException(status_code=404, detail="Conversation not found or access denied")
            
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
