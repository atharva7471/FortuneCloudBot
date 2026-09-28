from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import sys
import os

# Ensure we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag_chain import stream_qwen

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

def format_chat_history(messages: List[ChatMessage]):
    formatted = []
    # Only keep the last 4 messages to save context window and avoid slowing down the local LLM
    recent_messages = messages[-4:]
    for msg in recent_messages:
        role = "User" if msg.role == "user" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    chat_history_str = format_chat_history(request.history)
    
    def event_stream():
        for chunk in stream_qwen(request.message, chat_history_str):
            # Yield as Server-Sent Events (SSE) safely using JSON
            import json
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(event_stream(), media_type="text/event-stream")
