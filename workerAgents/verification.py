import asyncio
import json
from typing import TypedDict, Dict, Any, Annotated, List
from pydantic import Field, create_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI as genai
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from mcp.client.session import ClientSession

from sharedState.state import OrchestratorState

load_dotenv()

def mcp_to_langchain_tool(mcp_tool, session):
    
    fields = {}
    mcp_schema = mcp_tool.inputSchema or {}
    properties = mcp_schema.get("properties", {})
    required_fields = mcp_schema.get("required", [])

    for prop_name, prop_def in properties.items():
        py_type = str
        if prop_def.get("type") == "integer": py_type = int
        elif prop_def.get("type") == "boolean": py_type = bool
        elif prop_def.get("type") == "number": py_type = float
        
        if prop_name in required_fields:
            fields[prop_name] = (py_type, Field(description=prop_def.get("description", "")))
        else:
            fields[prop_name] = (py_type, Field(default=None, description=prop_def.get("description", "")))

    ToolInputSchema = create_model(f"{mcp_tool.name}Input", **fields)

    async def wrapped_tool(**kwargs):
        result = await session.call_tool(mcp_tool.name, arguments=kwargs)
        return result.content[0].text if result.content else "No content"

    return StructuredTool.from_function(
        func=None,
        coroutine=wrapped_tool,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema= ToolInputSchema
    )


async def build_verification_graph(session: ClientSession):

    llm = genai(model= "gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=2048)
    
    mcp_tools_list = await session.list_tools()
    tools = [mcp_to_langchain_tool(t,session) for t in mcp_tools_list.tools]
    
    # NODE 1 
    async def llm_node(state: OrchestratorState):
        llm_with_tools = llm.bind_tools(tools)

        inputMessage = f'''
                            You are a KYC Verification Agent in a state-based workflow. You are not a chat assistant. You are a function-calling engine. 
                            You MUST call the 'verify_kyc' tool immediately.

                            Input:
                            - Receive a state JSON.
                            - If `customer_id` is missing, immediately fail.
                            - Extract `customer_info.phone` if present, else use an empty string.

                            Tool:
                            - You must Call `verify_kyc(customer_id, phone)` exactly once.
                            - If the tool errors, store the error and mark status as failed.

                            Logic:
                            - verified  → phone_verified == true AND address_verified == true
                            - pending   → exactly one of them is true
                            - failed    → both false OR tool error

                            Output:
                            Return the updated state JSON with:
                            - `kyc_result`: tool response or error details
                            - `kyc_status`: "verified", "pending", or "failed"

                            Input:
                            customer_id is {state.get("customer_id")}
                            customer_phone is {state.get("customer_info").get("phone","")}
                            customer_address is {state.get("customer_info").get("city","")}
                            '''

        existing_history = state.get("verification_messages",[])
        response = await llm_with_tools.ainvoke(existing_history+ [HumanMessage(content=inputMessage)])

        return {"verification_messages" : [response]}
    
    # NODE 2
    tool_node = ToolNode(tools, messages_key="verification_messages")

    # NODE 3
    def final_status_node(state: OrchestratorState):
        messages = state["verification_messages"]
        last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        kyc_result = {}
        status = "failed"

        if last_tool_msg is not None:
            try:
                kyc_result = json.loads(last_tool_msg.content)
            except:
                kyc_result = {"raw": last_tool_msg.content}
            
    
            result = kyc_result.get("result")
            p_verified = result.get("phone_verified", False)
            a_verified = result.get("address_verified", False)
            

            if p_verified and a_verified:
                status = "verified"
            elif p_verified or a_verified:
                status = "pending"
            else:
                status = "failed"
        
        public_history = AIMessage(
            content=f"Verification Agent: Verified customer status is {status}",
            name="VerificationAgent"
        )
        
        return {"kyc_status": status, "kyc_result": kyc_result, "messages": [public_history]}
        
    def custom_tools_condition(state: OrchestratorState):
            messages = state.get("verification_messages", [])
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                return "tools"
            return END
    

    graph = StateGraph(OrchestratorState)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("final_logic", final_status_node)

    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges("llm_node", custom_tools_condition, {"tools": "tool_node", END : "final_logic" })
    graph.add_edge("tool_node", "llm_node")
    graph.add_edge("final_logic", END)
    return graph.compile()

