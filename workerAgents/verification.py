import asyncio
import json
from typing import TypedDict, Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from mcp.client import MCPClient
from mcp.client.session import ClientSession
from mcp.transport.http import HTTPTransport

load_dotenv()

#subset of the shard state relevant to verification agent
class OrchestrationState(TypedDict, total=False):
    customer_id: str
    customer_info: Dict[str, Any]
    kyc_result: Dict[str, Any]
    kyc_status: str  # "pending" | "verified" | "failed"


async def call_mcp_tool(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    transport = HTTPTransport(url="http://localhost:8000/mcp", timeout=30)
    async with MCPClient(transport=transport) as client:
        session: ClientSession = await client.start()
        return await session.call_tool(name=tool, arguments=args)


async def verify_kyc_node(state: OrchestrationState) -> OrchestrationState:
    """
    Reads state['customer_id'] and optionally state['customer_info']['phone'].
    Calls MCP tool verify_kyc(customer_id, phone) and writes:
      - state['kyc_result'] = result.get('result', {...})
      - state['kyc_status'] = "verified" / "failed" / "pending"
    """

    cid = state.get("customer_id")
    if not cid:
        state["kyc_status"] = "failed"
        state["kyc_result"] = {"error": "customer_id_missing"}
        return state

    phone = None
    if "customer_info" in state and isinstance(state["customer_info"], dict):
        phone = state["customer_info"].get("phone")

    args = {"customer_id": cid, "phone": phone} if phone is not None else {"customer_id": cid, "phone": ""}

    try:
        res = await call_mcp_tool("verify_kyc", args)
    except Exception as e:
        state["kyc_status"] = "failed"
        state["kyc_result"] = {"error": str(e)}
        return state

    result_payload = res.get("result", res)
    state["kyc_result"] = result_payload

    phone_verified = bool(result_payload.get("phone_verified")) if isinstance(result_payload, dict) else False
    address_verified = bool(result_payload.get("address_verified")) if isinstance(result_payload, dict) else False

    if phone_verified and address_verified:
        state["kyc_status"] = "verified"
    elif phone_verified or address_verified:
        state["kyc_status"] = "pending"
    else:
        state["kyc_status"] = "failed"

    return state


def build_verification_graph():
    graph = StateGraph(OrchestrationState)
    graph.add_node("verify_kyc", verify_kyc_node)
    graph.add_edge(START, "verify_kyc")
    graph.add_edge("verify_kyc", END)
    return graph.compile()

async def run_example():
    app = build_verification_graph()

    #HERE MAKE THE VERIFICATION AGENT FETCH DETAIL FROM ORCHESTRATOR SHARED STATE
    init_state: OrchestrationState = {
        "customer_id": "CUST005",
    }

    await app.ainvoke(init_state)


if __name__ == "__main__":
    asyncio.run(run_example())
