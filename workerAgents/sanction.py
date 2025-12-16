import asyncio
import json
import os

from typing import TypedDict, Dict, Any, Optional, Annotated, List
from pydantic import Field, create_model

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI as genai
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode, tools_condition

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

load_dotenv()

DOWNLOAD_DIR = os.environ.get("SANCTION_DOWNLOAD_DIR", "./downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

#subset of the shard state relevant to sanction agent
class SanctionAgentState(TypedDict, total=False):
    customer_id: str
    underwriting_result: Dict[str, Any]
    underwriting_status: str
    negotiated_offer: Dict[str, Any]

    sanction_letter_resource: Optional[str]
    sanction_letter_path: Optional[str]
    sanction_letter_status: Optional[str]   # "generated" | "failed"

    messages: Annotated[List[BaseMessage],add_messages]


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


async def build_sanction_graph(session: ClientSession):

    llm = genai(model= "gemini-2.5-flash-lite",
            temperature=0.2,
            max_output_tokens=2048)
    
    mcp_tools_list = await session.list_tools()

    tools = [mcp_to_langchain_tool(t,session) for t in mcp_tools_list.tools]

    # NODE 1 -> LLM Node
    async def llm_node(state: SanctionAgentState):
        llm_with_tools = llm.bind_tools(tools)

        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages" : [response]}
    
    # NODE 2 -> TOOL NODE
    tool_node = ToolNode(tools)

    # NODE 3 
    def final_status_node(state: SanctionAgentState):
        messages = state["messages"]
        last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        sanction_result = {}
        status = "failed"

        if last_tool_msg is not None:

            try:
                sanction_result = json.loads(last_tool_msg.content)
            except:
                sanction_result = {"raw": last_tool_msg.content}
            
            result = sanction_result.get("result", {})
            
            if not isinstance(result, dict):
                 result = {}

            resource = result.get("sanction_letter_resource", None)
            path = result.get("sanction_letter_path", None)

            if resource is not None and path is not None:
                status = "generated"
            
            state["sanction_letter_resource"] = resource
            state["sanction_letter_path"] = path
            state["sanction_letter_status"] = status
        
        else:
             print("\n[DEBUG] No tool execution found in messages.\n")
        
        return state

    graph = StateGraph(SanctionAgentState)

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

            agent = await build_sanction_graph(session)

            #HERE MAKE THE SANCTION AGENT FETCH DETAIL FROM ORCHESTRATOR SHARED STATE
            customer_id = "CUST010"
            negotiated_offer = {
                    "customer_id": "CUST010",
                    "approved_amount": 250000,
                    "tenure_months": 36,
                    "interest_rate": 11.5,
                    "justification": "Strong credit, within pre-approved"
                }
            underwriting_result = {
                    "decision": "approve",
                    "emi": 8250.0,
                    "reason": "within_pre_approved_limit",
                    "salary_slip_resource": None
                }
            underwriting_status = "approve"

            inputMessage = f'''
                            You are an NBFC Sanction agent in a state-based workflow. You are not a chat assistant. You are a function-calling engine.
                            You have to utilise "generate_sanction_letter" tool immediately and only once.

                            Input:
                            - Receive many fields of data reuired to call both the tools, if some data fields are missing then use the default tool value.
                            - If `customer_id` is missing, immediately fail.

                            Tools:
                            - You must Call `generate_sanction_letter` tool exactly once.
                            - If the tool errors, store the error and mark status as failed.
                            
                            Logic:
                            - Firstly call the `generate_sanction_letter` tool only and only if  underwriting_status is "approve". 
                            - If underwriting_status is "reject" then mark the status as 'failed'
                            - If sanction_letter_resource and sanction_letter_path key values are present then mark the status as 'generated' or else as 'failed'
                            

                            Output:
                            Return the updated state JSON with:
                            - `sanction_letter_resource`: as returned by the tool
                            - `sanction_letter_path`: as returned by the tool
                            - `sanction_letter_status`: 'generated' or 'failed' based on whether sanction_letter_resource and sanction_letter_path are present

                            Input:
                            customer_id is {customer_id}
                            amount is {negotiated_offer["approved_amount"]}
                            tenure_months is {negotiated_offer["tenure_months"]}
                            interest_rate is {negotiated_offer["interest_rate"]}
                            underwriting_status is {underwriting_status}
                            '''


            init: SanctionAgentState = {
                "customer_id": customer_id,
                "negotiated_offer": negotiated_offer,
                "underwriting_result": underwriting_result,
                "underwriting_status": underwriting_status,
                "sanction_letter_resource": None,
                "sanction_letter_path": None,
                "sanction_letter_status": None,
                "messages" : [HumanMessage(content=inputMessage)]
            }

            final_state = await agent.ainvoke(init)

             # Create a clean dictionary for printing, excluding the raw 'messages' objects
            output = {
                "customer_id": customer_id,
                "negotiated_offer": negotiated_offer,
                "underwriting_result": underwriting_result,
                "underwriting_status": underwriting_status,
                "sanction_letter_resource": final_state["sanction_letter_resource"],
                "sanction_letter_path": final_state["sanction_letter_path"],
                "sanction_letter_status": final_state["sanction_letter_status"],
            }

            # HERE RETURN THE OUTPUT AND NOT THE EXACT STATE
            print(json.dumps(output, indent=2))
            
            


if __name__ == "__main__":
    asyncio.run(main())
