import asyncio
import json
import os
from typing import TypedDict, Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from mcp.client import MCPClient
from mcp.client.session import ClientSession
from mcp.transport.http import HTTPTransport

load_dotenv()

DOWNLOAD_DIR = os.environ.get("SANCTION_DOWNLOAD_DIR", "./downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

#subset of the shard state relevant to sanction agent
class OrchestrationState(TypedDict, total=False):
    customer_id: str
    underwriting_result: Dict[str, Any]
    negotiated_offer: Dict[str, Any]

    sanction_letter_resource: Optional[str]
    sanction_letter_path: Optional[str]
    sanction_status: Optional[str]  # "pending" | "generated" | "failed"



async def with_mcp_session(fn):
    """
    Helper to create an MCP session and run a coroutine that receives the session.
    Usage:
        await with_mcp_session(lambda session: session.call_tool(...))
    """
    transport = HTTPTransport(url="http://localhost:8000/mcp", timeout=60)
    async with MCPClient(transport=transport) as client:
        session: ClientSession = await client.start()
        return await fn(session)



async def generate_sanction_letter_node(state: OrchestrationState) -> OrchestrationState:
    """
    Preconditions:
      - state['customer_id'] present
      - state['underwriting_result'] shows approval OR negotiated_offer present
    Behavior:
      - Call MCP tool generate_sanction_letter with customer_id and amount
      - Expect response: {"status":"ok","result":{"resource":"resource://...","path":"..."}}
      - Fetch the resource bytes via session.read_resource(resource_uri) and save locally
      - Update state with sanction_letter_resource, sanction_letter_path, sanction_status
    """

    cid = state.get("customer_id")
    if not cid:
        state["sanction_status"] = "failed"
        state["sanction_letter_path"] = None
        state["sanction_letter_resource"] = None
        return state

    approved_amount = None
    uw = state.get("underwriting_result", {})
    negotiated = state.get("negotiated_offer", {})

    # Many underwriting_result shapes; try common keys
    if isinstance(uw, dict):
        # if underwriting tool returned 'approved' decision and possibly 'emi' but not amount,
        # prefer negotiated_offer if present. Otherwise try 'approved_amount' or 'requested' fields.
        if uw.get("decision") in ("approve", "approved"):
            approved_amount = negotiated.get("approved_amount") or uw.get("requested") or uw.get("approved_amount")
    # fallback to negotiated offer
    if approved_amount is None and negotiated:
        approved_amount = negotiated.get("approved_amount") or negotiated.get("requested_amount")

    if approved_amount is None:
        # cannot generate sanction without amount
        state["sanction_status"] = "failed"
        state["sanction_letter_resource"] = None
        state["sanction_letter_path"] = None
        return state

    tenure_months = negotiated.get("tenure_months") or uw.get("tenure_months") or 36
    interest_rate = negotiated.get("interest_rate") or uw.get("interest_rate") or 12.0

    payload = {
        "customer_id": cid,
        "amount": int(approved_amount),
        "tenure_months": int(tenure_months),
        "interest_rate": float(interest_rate),
    }

    # Call generate_sanction_letter tool and then fetch resource
    try:
        async def call_and_fetch(session: ClientSession):
            res = await session.call_tool(name="generate_sanction_letter", arguments=payload)
            result_payload = res.get("result", res)
            resource_uri = result_payload.get("resource")
            # If resource not provided, treat as failure
            if not resource_uri:
                raise RuntimeError(f"no resource returned from generate_sanction_letter: {result_payload}")

            # read resource bytes
            content_bytes = await session.read_resource(resource_uri)
            # save locally
            filename = resource_uri.replace("resource://", "")
            local_path = os.path.join(DOWNLOAD_DIR, filename)

            with open(local_path, "wb") as f:
                f.write(content_bytes)

            return resource_uri, local_path, result_payload

        resource_uri, local_path, result_payload = await with_mcp_session(call_and_fetch)

    except Exception as e:
        state["sanction_status"] = "failed"
        state["sanction_letter_resource"] = None
        state["sanction_letter_path"] = None
        state.setdefault("sanction_error", str(e))
        return state

    # Success — update state
    state["sanction_letter_resource"] = resource_uri
    state["sanction_letter_path"] = local_path
    state["sanction_status"] = "generated"
    # Optionally store raw tool result
    state["sanction_tool_result"] = result_payload

    return state



def build_sanction_graph():
    graph = StateGraph(OrchestrationState)
    graph.add_node("generate_sanction_letter", generate_sanction_letter_node)
    graph.add_edge(START, "generate_sanction_letter")
    graph.add_edge("generate_sanction_letter", END)
    return graph.compile()


async def run_example():
    app = build_sanction_graph()

    #HERE MAKE THE SANCTION AGENT FETCH DETAIL FROM ORCHESTRATOR SHARED STATE

    init_state: OrchestrationState = {
        "customer_id": "CUST010",
        "negotiated_offer": {
            "customer_id": "CUST010",
            "approved_amount": 250000,
            "tenure_months": 36,
            "interest_rate": 11.5,
            "justification": "Strong credit, within pre-approved"
        },
        "underwriting_result": {
            "decision": "approve",
            "emi": 8250.0,
            "reason": "within_pre_approved_limit"
        }
    }

    await app.ainvoke(init_state)
    


if __name__ == "__main__":
    asyncio.run(run_example())
