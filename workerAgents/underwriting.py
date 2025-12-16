import asyncio
import json
import os
import base64

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
import httpx

load_dotenv()

#Subset of the shared state of orchestrator
class UnderwritingAgentState(TypedDict, total=False):
    customer_id: str
    customer_info: Dict[str, Any]
    negotiated_offer: Dict[str, Any]    

    credit_score: int
    salary_slip_resource: Optional[str]   # resource://... (after upload)
    underwriting_input: Dict[str, Any]
    underwriting_result: Dict[str, Any]
    underwriting_status: str              # "pending" | "approved" | "rejected" | "need_documents"

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

async def build_underwriting_graph(session: ClientSession):

    llm = genai(model= "gemini-2.5-flash-lite",
            temperature=0.2,
            max_output_tokens=2048)
    
    mcp_tools_list = await session.list_tools()

    tools = [mcp_to_langchain_tool(t,session) for t in mcp_tools_list.tools]
    

    # NODE 1 -> FETCHING CREDIT SCORE
    async def fetch_credit_score(state: UnderwritingAgentState):
        cid = str(state["customer_id"])
        try:
            result = await session.call_tool(
                "get_credit_score",
                arguments={"customer_id": cid}
            )
            credit_result = json.loads(result.content[0].text)
            if "result" in credit_result and isinstance(credit_result["result"], dict):
                new_result = credit_result.get("result",{})
                state["credit_score"] = new_result["credit_score"]
            else:
                 state["credit_score"] = credit_result

        except httpx.ConnectError:
            print("RROR: Connection Refused. (Likely the 0.0.0.0 bug)")
            state["credit_score"] = {"error": "Connection Refused"}
            
        except httpx.TimeoutException:
            print("ERROR: Timed Out. (Server is blocked or slow)")
            state["credit_score"] = {"error": "Timeout"}
            
        except Exception as e:
            print(f"ERROR: Generic Exception -> {type(e).__name__}: {e}")
            state["credit_score"] = {"error": str(e)}

        return state
    
    # NODE 2 -> LLM NODE
    async def llm_node(state: UnderwritingAgentState):
        llm_with_tools = llm.bind_tools(tools)

        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages" : [response]}
    
    # NODE 3 -> TOOL NODE
    tool_node = ToolNode(tools)

    # NODE 4 -> FINAL LOGIC NODE
    def final_status_node(state: UnderwritingAgentState):
        messages = state["messages"]
        last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        underwrite = {}

        if last_tool_msg is not None:
            try:
                underwrite = json.loads(last_tool_msg.content)
            except:
                underwrite = {"raw": last_tool_msg.content}

        result = underwrite.get("result")
        state["salary_slip_resource"] = result.get("salary_slip_resource")
        state["underwriting_status"] = result.get("decision")
        state["underwriting_input"] = {
            "customer_id": state["customer_id"],
            "requested_amount": state["negotiated_offer"]["approved_amount"],
            "tenure_months": state["negotiated_offer"]["tenure_months"],
            "annual_rate": state["customer_info"]["salary_monthly"],
        }
        state["underwriting_result"] = result

        return state


    graph = StateGraph(UnderwritingAgentState)

    graph.add_node("fetch_credit_score", fetch_credit_score)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("final_logic", final_status_node)

    graph.add_edge(START, "fetch_credit_score")
    graph.add_edge("fetch_credit_score", "llm_node")
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

            agent = await build_underwriting_graph(session)

            #HERE MAKE THE UNDERWRITING AGENT FETCH DETAIL FROM ORCHESTRATOR SHARED STATE
            customer_id = "CUST005"
            customer_info = {
                    "customer_id": "CUST005",
                    "name": "Nisha Patel",
                    "salary_monthly": 52000,
                    "pre_approved_limit": 250000,
                    "credit_score": 710
                }
            negotiated_offer = {
                    "customer_id": "CUST005",
                    "approved_amount": 200000,
                    "tenure_months": 36,
                    "interest_rate": 14.0,
                    "justification": "Within 2x limit and fits EMI criteria"
                }

            inputMessage = f'''
                            You are an NBFC Underwriting agent in a state-based workflow. You are not a chat assistant. You are a function-calling engine.
                            You have to utilise upload_salary_slip tool and underwrite_loan tool multiple times as per requirement.

                            Input:
                            - Receive many fields of data reuired to call both the tools, if some data fields are missing then use the default tool value.
                            - If `customer_id` is missing, immediately fail.

                            Tools:
                            - First you must Call `underwrite_loan` tool, based on the output decision field you need to make the next step.
                            - If the decision is reject or approved your job is over.
                            - If the decision is 'require_salary_slip' and reason is 'salary_slip_required' you need to call the 'upload_salary_slip' tool with required parameters.
                            
                            Logic:
                            - decision is approve  → go to the 'END' node. No further tool call required.
                            - decision is reject   → go to the 'END' node. No further tool call required.
                            - decision is require_salary_slip → then call 'upload_salary_slip' tool to make salary slip upload and then use the output of previous tool 
                              call to make a call again to `underwrite_loan` tool so that decision is either approve or reject.

                            Output:
                            Return the updated state JSON with:
                            - `decision`: 'approve' or 'reject' or 'upload_salary_slip'
                            - `reason`: as returned by the tool
                            - `emi`: as returned by the tool
                            - `salary_slip_resource`: as returned by the tool

                            Input:
                            customer_id is {customer_id}
                            requested_amount is {negotiated_offer["approved_amount"]}
                            tenure_months is {negotiated_offer["tenure_months"]}
                            annual_rate is {negotiated_offer["interest_rate"]}
                            salary_provided is {customer_info["salary_monthly"]}
                            '''

            init: UnderwritingAgentState = {
                "customer_id": customer_id,
                "customer_info": customer_info,
                "negotiated_offer": negotiated_offer,
                "messages" : [HumanMessage(content=inputMessage)]
            }

            final_state = await agent.ainvoke(init)

            # Create a clean dictionary for printing, excluding the raw 'messages' objects
            output = {
                "customer_id": final_state.get("customer_id"),
                "customer_info": final_state.get("customer_info"),
                "negotiated_offer": final_state.get("negotiated_offer"),
                "credit_score": final_state.get("credit_score"),
                "salary_slip_resource": final_state.get("salary_slip_resource"),
                "underwriting_input": final_state.get("underwriting_input"),
                "underwriting_result": final_state.get("underwriting_result"),
                "underwriting_status": final_state.get("underwriting_status")
            }

            # HERE RETURN THE OUTPUT AND NOT THE EXACT STATE
            print(json.dumps(output, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
