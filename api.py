import uvicorn
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict, Any, List

# MCP Client Imports
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from langchain_core.messages import HumanMessage

# Import your Master Agent Builder
from master import build_orchestrator_graph

# ----------------------------------------------------------------
# GLOBAL STATE
# ----------------------------------------------------------------
# 1. App State: Holds the "compiled graph" and "mcp session" globally
app_state = {}

# 2. Session Store: Holds the conversation history for each user (In-Memory Database)
# In production, you would replace this with Redis or PostgreSQL.
session_store: Dict[str, Dict[str, Any]] = {}

# ----------------------------------------------------------------
# LIFECYCLE MANAGER (The "Startup/Shutdown" Logic)
# ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs ONCE when the server starts.
    It connects to the MCP Server and builds the AI Graph.
    """
    mcp_url = "http://127.0.0.1:8000/sse"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    
    print(f"🔌 Connecting to MCP Server at {mcp_url}...")
    
    try:
        # Connect to the SSE stream
        async with sse_client(mcp_url, headers=headers) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                print("✅ Connected to MCP Server.")
                await session.initialize()
                
                # BUILD THE GRAPH with this active session
                # This passes the tool definitions to all worker agents
                print("🧠 Building AI Orchestrator...")
                agent_graph = await build_orchestrator_graph(session)
                
                # Store the compiled graph in global state so endpoints can use it
                app_state["graph"] = agent_graph
                
                # Yield control back to FastAPI to handle requests
                yield
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not connect to MCP: {e}")
        # We yield to let the server start (so you see errors), but it won't work well.
        yield
        
    print("🔌 Shutting down MCP connection...")

# Initialize App with the lifespan logic
app = FastAPI(lifespan=lifespan)


# ----------------------------------------------------------------
# API MODELS
# ----------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str  # Unique ID for the user (e.g., "user_123")
    message: str     # The text message user typed

class ChatResponse(BaseModel):
    response: str    # What the bot says back
    debug_stage: str # Current stage (e.g., "negotiation", "underwriting")


# ----------------------------------------------------------------
# CHAT ENDPOINT
# ----------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # 1. Get the Graph
    graph = app_state.get("graph")
    if not graph:
        raise HTTPException(status_code=503, detail="AI Brain not ready (MCP Connection failed)")

    # 2. Retrieve or Initialize User Session
    if request.session_id not in session_store:
        # Start fresh for new user
        session_store[request.session_id] = {
            "messages": [],              # Chat history
            "flow_stage": "start",       # Initial stage
            
            # Initialize empty data containers
            "negotiated_offer": {},
            "kyc_status": "pending",
            "underwriting_status": "pending",
            
            # Dedicated scratchpads for workers
            "sales_messages": [],
            "verification_messages": [],
            "underwriting_messages": [],
            "sanction_messages": []
        }
    
    # Load current state from memory
    current_state = session_store[request.session_id]
    
    # 3. Add User Input to State
    # We append the new message to the history
    current_state["messages"].append(HumanMessage(content=request.message))
    
    # 4. RUN THE BRAIN (Invoke the Graph)
    # recursion_limit=20 allows the Master -> Worker -> Master loop to happen multiple times
    try:
        final_state = await graph.ainvoke(current_state, config={"recursion_limit": 20})
    except Exception as e:
        print(f"Error executing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # 5. Save Updated State
    session_store[request.session_id] = final_state
    
    # 6. Extract Response for Frontend
    # The last message in "messages" is the AI's reply
    ai_response_text = "I'm having trouble processing that."
    if final_state["messages"]:
        last_msg = final_state["messages"][-1]
        ai_response_text = last_msg.content

    return ChatResponse(
        response=ai_response_text,
        debug_stage=final_state.get("flow_stage", "unknown")
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)