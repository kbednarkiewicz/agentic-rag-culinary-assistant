from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.agent import CulinaryAgent

app = FastAPI(title="Agentic RAG Culinary Assistant API",
    description="Conversational culinary assistant powered by LangGraph, ChromaDB, and Gemini.",
    version="1.0.0")

agent = CulinaryAgent()

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["Give me a recipe with potatoes"])
    thread_id: str = Field(default="1", examples=["session-123"])

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = agent.run_agent(query=request.query, thread_id=request.thread_id)
        return ChatResponse(response=answer, thread_id=request.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))