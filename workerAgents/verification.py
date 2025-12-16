import asyncio
import json
from typing import TypedDict, Dict, Any, Annotated, List
from pydantic import Field, create_model
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI as genai
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


load_dotenv()

#subset of the shard state relevant to verification agent
class VerificationAgentState(TypedDict, total=False):
    customer_id: str 
    customer_info: Dict[str, Any]
    kyc_result: Dict[str, Any]
    kyc_status: str
    messages: Annotated[List[BaseMessage], add_messages]

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

    llm = genai(model= "gemini-2.5-flash-lite",
            temperature=0.2,
            max_output_tokens=2048)
    
    mcp_tools_list = await session.list_tools()
    tools = [mcp_to_langchain_tool(t,session) for t in mcp_tools_list.tools]
    
    # NODE 1 
    async def llm_node(state: VerificationAgentState):
        llm_with_tools = llm.bind_tools(tools)

        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages" : [response]}
    
    # NODE 2
    tool_node = ToolNode(tools)

    # NODE 3
    def final_status_node(state: VerificationAgentState):
        messages = state["messages"]
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
        
        return {"kyc_status": status, "kyc_result": kyc_result}
        

    graph = StateGraph(VerificationAgentState)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("final_logic", final_status_node)

    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges("llm_node", tools_condition, {"tools": "tool_node", END : "final_logic" })
    graph.add_edge("tool_node", "llm_node")
    graph.add_edge("final_logic", END)
    return graph.compile()

async def main():

    server_url = "http://127.0.0.1:8000/sse"
    header = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    timeout = 30.0

    async with sse_client(server_url,headers=header, timeout=timeout) as streams:
        read_stream, write_stream = streams
        async with ClientSession(read_stream, write_stream) as session:
            print("Successfully connected to MCP server")

            await session.initialize()
        
            agent = await build_verification_graph(session)

            #HERE MAKE THE VERIFICATION AGENT FETCH DETAIL FROM ORCHESTRATOR SHARED STATE
            customer_id = "CUST005"
            customer_info = {"customer_id":"CUST005","name":"Nisha Patel","age":27,"city":"Ahmedabad","phone":"9810000005","email":"nisha@example.com","pre_approved_limit":250000,"salary_monthly":52000,"credit_score":710}
            
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
                            customer_id is {customer_id}
                            customer_phone is {customer_info.get("phone","")}
                            customer_address is {customer_info.get("city","")}
                            '''
            
            
            init: VerificationAgentState = {
                "customer_id": customer_id,
                "customer_info": customer_info,
                "messages" : [HumanMessage(content=inputMessage)]
            }


            final_state = await agent.ainvoke(init)

            # Create a clean dictionary for printing, excluding the raw 'messages' objects
            output = {
                "customer_id": final_state.get("customer_id"),
                "customer_info": final_state.get("customer_info"),
                "kyc_status": final_state.get("kyc_status"),
                "kyc_result": final_state.get("kyc_result")
            }
            # HERE RETURN THE OUTPUT AND NOT THE EXACT STATE
            print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
