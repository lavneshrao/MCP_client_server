import asyncio
import json
import os
import base64

from typing import TypedDict, Dict, Any, Optional, Annotated, List
from pydantic import Field, create_model

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI as genai
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from mcp.client.session import ClientSession
import httpx

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

async def build_underwriting_graph(session: ClientSession):

    llm = genai(model= "gemini-2.5-flash-lite",
            temperature=0.2,
            max_output_tokens=2048)
    
    mcp_tools_list = await session.list_tools()

    tools = [mcp_to_langchain_tool(t,session) for t in mcp_tools_list.tools]
    

    # NODE 1 -> FETCHING CREDIT SCORE
    async def fetch_credit_score(state: OrchestratorState):
        cid = str(state["customer_id"])
        try:
            result = await session.call_tool(
                "get_credit_score",
                arguments={"customer_id": cid}
            )
            updates = {}
            credit_result = json.loads(result.content[0].text)
            if "result" in credit_result and isinstance(credit_result["result"], dict):
                new_result = credit_result.get("result",{})
                updates["credit_score"] = new_result["credit_score"]
            else:
                updates["credit_score"] = credit_result

        except httpx.ConnectError:
            print("RROR: Connection Refused. (Likely the 0.0.0.0 bug)")
            updates["credit_score"] = {"error": "Connection Refused"}
            
        except httpx.TimeoutException:
            print("ERROR: Timed Out. (Server is blocked or slow)")
            updates["credit_score"] = {"error": "Timeout"}
            
        except Exception as e:
            print(f"ERROR: Generic Exception -> {type(e).__name__}: {e}")
            updates["credit_score"] = {"error": str(e)}

        return updates
    
    # NODE 2 -> LLM NODE
    async def llm_node(state: OrchestratorState):
        llm_with_tools = llm.bind_tools(tools)

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
                            customer_id is {state.get("customer_id")}
                            requested_amount is {state.get("negotiated_offer").get("approved_amount")}
                            tenure_months is {state.get("negotiated_offer").get("tenure_months")}
                            annual_rate is {state.get("negotiated_offer").get("interest_rate")}
                            salary_provided is {state.get("customer_info").get("salary_monthly")}
                            '''

        existing_history = state.get("underwriting_messages",[])

        response = await llm_with_tools.ainvoke(existing_history + [HumanMessage(content=inputMessage)])

        return {"underwriting_messages" : [response]}
    
    # NODE 3 -> TOOL NODE
    tool_node = ToolNode(tools, messages_key="underwriting_messages")

    # NODE 4 -> FINAL LOGIC NODE
    def final_status_node(state: OrchestratorState):
        messages = state["underwriting_messages"]
        last_tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        underwrite = {}

        if last_tool_msg is not None:
            try:
                underwrite = json.loads(last_tool_msg.content)
            except:
                underwrite = {"raw": last_tool_msg.content}

        result = underwrite.get("result")

        updates = {}

        updates["salary_slip_resource"] = result.get("salary_slip_resource")
        updates["underwriting_status"] = result.get("decision")
        updates["underwriting_input"] = {
            "customer_id": state["customer_id"],
            "requested_amount": state["negotiated_offer"]["approved_amount"],
            "tenure_months": state["negotiated_offer"]["tenure_months"],
            "annual_rate": state["customer_info"]["salary_monthly"],
        }
        updates["underwriting_result"] = result

        public_history = AIMessage(
            content=f'Underwriting Agent: Underwriting of customer offer is {updates.get("underwriting_status")}',
            name="UnderWritingAgent"
        )

        updates["messages"] = [public_history]

        return updates


    graph = StateGraph(OrchestratorState)

    graph.add_node("fetch_credit_score", fetch_credit_score)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("final_logic", final_status_node)

    def custom_tools_condition(state: OrchestratorState):
            messages = state.get("underwriting_messages", [])
            if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                return "tools"
            return END

    graph.add_edge(START, "fetch_credit_score")
    graph.add_edge("fetch_credit_score", "llm_node")
    graph.add_conditional_edges("llm_node", custom_tools_condition, {"tools": "tool_node", END : "final_logic" })
    graph.add_edge("tool_node", "llm_node")
    graph.add_edge("final_logic", END)

    return graph.compile()

