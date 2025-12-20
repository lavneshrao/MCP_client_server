import json
import os

from pydantic import Field, create_model

from dotenv import load_dotenv

from langchain_core.messages import  HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI as genai
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from mcp.client.session import ClientSession

from sharedState.state import OrchestratorState

load_dotenv()

DOWNLOAD_DIR = os.environ.get("SANCTION_DOWNLOAD_DIR", "./downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


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

    llm = genai(model= "gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=2048)
    
    mcp_tools_list = await session.list_tools()

    tools = [mcp_to_langchain_tool(t,session) for t in mcp_tools_list.tools]

    # NODE 1 -> LLM Node
    async def llm_node(state: OrchestratorState):
        llm_with_tools = llm.bind_tools(tools)

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
                            customer_id is {state.get("customer_id")}
                            amount is {state.get("negotiated_offer").get("approved_amount")}
                            tenure_months is {state.get("negotiated_offer").get("tenure_months")}
                            interest_rate is {state.get("negotiated_offer").get("interest_rate")}
                            underwriting_status is {state.get("underwriting_status")}
                            '''

        existing_history = state.get("sanction_messages",[])

        response = await llm_with_tools.ainvoke(existing_history + [HumanMessage(content=inputMessage)])
        return {"sanction_messages" : [response]}
    
    # NODE 2 -> TOOL NODE
    tool_node = ToolNode(tools, messages_key="sanction_messages")

    # NODE 3 
    def final_status_node(state: OrchestratorState):
        messages = state["sanction_messages"]
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
            
            updates={}
            updates["sanction_letter_resource"] = resource
            updates["sanction_letter_path"] = path
            updates["sanction_letter_status"] = status

            public_history = AIMessage(
            content=f'''Sanction Agent: Customer's sanction letter generation is {updates.get("sanction_letter_status")}''',
            name="SanctionAgent"
             )       
            updates["messages"] = [public_history]
        
        else:
             print("\n[DEBUG] No tool execution found in messages.\n")
        
        return updates

    graph = StateGraph(OrchestratorState)

    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("final_logic", final_status_node)

    def custom_tools_condition(state: OrchestratorState):
            messages = state.get("sanction_messages", [])
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                return "tools"
            return END

    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges("llm_node", custom_tools_condition, {"tools": "tool_node", END : "final_logic" })
    graph.add_edge("tool_node", "llm_node")
    graph.add_edge("final_logic", END)

    return graph.compile()


