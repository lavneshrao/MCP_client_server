import asyncio
import json
import os
import base64
from typing import TypedDict, Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from mcp.client import MCPClient
from mcp.client.session import ClientSession
from mcp.transport.http import HTTPTransport

load_dotenv()

#Subset of the shared state of orchestrator
class OrchestrationState(TypedDict, total=False):
    customer_id: str
    customer_info: Dict[str, Any]
    negotiated_offer: Dict[str, Any]      
    credit_score: Optional[int]
    salary_slip_local_path: Optional[str] # optional local path 
    salary_slip_resource: Optional[str]   # resource://... (after upload)
    underwriting_input: Dict[str, Any]
    underwriting_result: Dict[str, Any]
    underwriting_status: str              # "pending" | "approved" | "rejected" | "need_documents"


async def call_mcp_tool(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    transport = HTTPTransport(url="http://localhost:8000/mcp", timeout=60)
    async with MCPClient(transport=transport) as client:
        session: ClientSession = await client.start()
        return await session.call_tool(name=tool, arguments=args)


async def fetch_credit_score(state: OrchestrationState) -> OrchestrationState:
    cid = state.get("customer_id")
    if not cid:
        state["underwriting_status"] = "rejected"
        state["underwriting_result"] = {"error": "customer_id_missing"}
        return state

    try:
        res = await call_mcp_tool("get_credit_score", {"customer_id": cid})
    except Exception as e:
        state["underwriting_status"] = "rejected"
        state["underwriting_result"] = {"error": str(e)}
        return state

    payload = res.get("result", {})
    state["credit_score"] = payload.get("credit_score")
    return state


async def maybe_upload_salary_slip(state: OrchestrationState) -> OrchestrationState:
    cid = state.get("customer_id")
    if not cid:
        return state

    local_path = state.get("salary_slip_local_path")
    if state.get("salary_slip_resource"):
        return state

    if not local_path:
        return state

    if not os.path.exists(local_path):
        state["underwriting_status"] = "need_documents"
        state["underwriting_result"] = {"error": f"salary slip not found at {local_path}"}
        return state

    try:
        with open(local_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
    except Exception as e:
        state["underwriting_status"] = "need_documents"
        state["underwriting_result"] = {"error": f"read_error: {e}"}
        return state

    filename = os.path.basename(local_path)

    try:
        res = await call_mcp_tool(
            "upload_salary_slip",
            {"customer_id": cid, "filename": filename, "content_base64": b64},
        )
    except Exception as e:
        state["underwriting_status"] = "need_documents"
        state["underwriting_result"] = {"error": str(e)}
        return state

    result_payload = res.get("result", {})
    state["salary_slip_resource"] = result_payload.get("resource")
    return state


async def run_underwriting_node(state: OrchestrationState) -> OrchestrationState:
    cid = state.get("customer_id")
    negotiated = state.get("negotiated_offer", {})
    customer_info = state.get("customer_info", {})

    if not cid or not negotiated:
        state["underwriting_status"] = "rejected"
        state["underwriting_result"] = {"error": "missing customer_id_or_negotiated_offer"}
        return state

    requested_amount = int(negotiated.get("approved_amount", negotiated.get("requested_amount", 0)))
    tenure = int(negotiated.get("tenure_months", 36))
    interest_rate = float(negotiated.get("interest_rate", 12.0))

    # Prefer explicit salary_provided in negotiated_offer > customer_info.salary_monthly
    salary_provided = negotiated.get("salary_provided")
    if salary_provided is None:
        salary_provided = customer_info.get("salary_monthly")

    salary_slip_resource = state.get("salary_slip_resource")

    underwriting_payload = {
        "customer_id": cid,
        "requested_amount": requested_amount,
        "tenure_months": tenure,
        "annual_rate": interest_rate,
        "salary_provided": salary_provided,
        "salary_slip_resource": salary_slip_resource,
    }

    state["underwriting_input"] = underwriting_payload

    try:
        res = await call_mcp_tool("underwrite_loan", underwriting_payload)
    except Exception as e:
        state["underwriting_status"] = "rejected"
        state["underwriting_result"] = {"error": str(e)}
        return state

    result_payload = res.get("result", res)
    state["underwriting_result"] = result_payload

    decision = None
    if isinstance(result_payload, dict):
        decision = result_payload.get("decision") or result_payload.get("status")

    if decision == "approve" or decision == "approved":
        state["underwriting_status"] = "approved"
    elif decision in ("require_salary_slip", "need_documents"):
        state["underwriting_status"] = "need_documents"
    elif decision == "reject" or decision == "rejected":
        state["underwriting_status"] = "rejected"
    else:
        # fallback: if EMI present and reason is approval-like
        if result_payload.get("emi") is not None and result_payload.get("reason"):
            reason = str(result_payload.get("reason", "")).lower()
            if "approve" in reason:
                state["underwriting_status"] = "approved"
            elif "require" in reason or "salary" in reason:
                state["underwriting_status"] = "need_documents"
            else:
                state["underwriting_status"] = "rejected"
        else:
            state["underwriting_status"] = "rejected"

    return state


def build_underwriting_graph():
    graph = StateGraph(OrchestrationState)

    graph.add_node("fetch_credit_score", fetch_credit_score)
    graph.add_node("maybe_upload_salary_slip", maybe_upload_salary_slip)
    graph.add_node("run_underwriting", run_underwriting_node)

    graph.add_edge(START, "fetch_credit_score")
    graph.add_edge("fetch_credit_score", "maybe_upload_salary_slip")
    graph.add_edge("maybe_upload_salary_slip", "run_underwriting")
    graph.add_edge("run_underwriting", END)

    return graph.compile()

async def run_example():
    app = build_underwriting_graph()

    #HERE MAKE THE UNDERWRITING AGENT FETCH DETAIL FROM ORCHESTRATOR SHARED STATE
    init_state: OrchestrationState = {
        "customer_id": "CUST005",
        "customer_info": {
            "customer_id": "CUST005",
            "name": "Nisha Patel",
            "salary_monthly": 52000,
            "pre_approved_limit": 250000,
            "credit_score": 710
        },
        "negotiated_offer": {
            "customer_id": "CUST005",
            "approved_amount": 200000,
            "tenure_months": 36,
            "interest_rate": 14.0,
            "justification": "Within 2x limit and fits EMI criteria"
        },
    }

    await app.ainvoke(init_state)

if __name__ == "__main__":
    asyncio.run(run_example())
