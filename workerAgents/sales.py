import asyncio
import json
from typing import TypedDict, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from mcp.client import MCPClient
from mcp.client.session import ClientSession
from mcp.transport.http import HTTPTransport


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    max_output_tokens=2048,
)


class NegotiatedOffer(BaseModel):
    customer_id: str
    approved_amount: int = Field(description="Amount upto which NBFC is allowed to give up personal loan")
    tenure_months: int = Field(description="Period in months for which loan is to be taken")
    interest_rate: float = Field(description="Interest rate upto which loan can be approved")
    justification: str = Field(description="Justification for the approval or rejection of deal")


class SalesAgentState(TypedDict, total=False):
    customer_id: str
    requested_amount: int
    preferred_tenure_months: int
    max_interest_rate: float

    customer_info: Dict[str, Any]
    negotiated_offer: Dict[str, Any]



async def call_mcp_tool(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    transport = HTTPTransport(url="http://localhost:8000/mcp", timeout=30)

    async with MCPClient(transport=transport) as client:
        session: ClientSession = await client.start()
        return await session.call_tool(name=tool, arguments=args)



async def fetch_customer_info(state: SalesAgentState) -> SalesAgentState:
    cid = state["customer_id"]

    result = await call_mcp_tool("get_customer_info", {"customer_id": cid})
    state["customer_info"] = result.get("result", {})

    return state


# NODE 2 → NEGOTIATE TERMS (Gemini-2.5-Pro)
def negotiate_terms(state: SalesAgentState) -> SalesAgentState:
    customer = state["customer_info"]
    requested = state["requested_amount"]
    tenure = state["preferred_tenure_months"]
    max_rate = state["max_interest_rate"]

    structured_llm = llm.with_structured_output(NegotiatedOffer)

    system_prompt = """
You are an NBFC Sales Agent.
Negotiate loan terms based on:
- customer salary
- pre-approved limit
- credit score
- requested amount
- preferred tenure
- max acceptable interest rate

Rules:
- Never exceed max_interest_rate.
- Stay within 2x pre-approved limit.
- If credit score < 720, be conservative.
- Provide clear justification.
Return only JSON following the NegotiatedOffer schema.
"""

    context = {
        "customer_info": customer,
        "requested_amount": requested,
        "preferred_tenure_months": tenure,
        "max_interest_rate": max_rate,
    }

    result: NegotiatedOffer = structured_llm.invoke(
        f"{system_prompt}\n\nContext:\n{json.dumps(context, indent=2)}"
    )

    state["negotiated_offer"] = result.model_dump()
    return state



def build_sales_graph():
    graph = StateGraph(SalesAgentState)

    graph.add_node("fetch_customer_info", fetch_customer_info)
    graph.add_node("negotiate_terms", negotiate_terms)

    graph.add_edge(START, "fetch_customer_info")
    graph.add_edge("fetch_customer_info", "negotiate_terms")
    graph.add_edge("negotiate_terms", END)

    return graph.compile()



async def run_example():
    app = build_sales_graph()

    #HERE MAKE THE SALES AGENT FETCH DETAIL FROM ORCHESTRATOR SHARED STATE
    init: SalesAgentState = {
        "customer_id": "CUST005",
        "requested_amount": 350000,
        "preferred_tenure_months": 48,
        "max_interest_rate": 16.0
    }

    await app.ainvoke(init)


if __name__ == "__main__":
    asyncio.run(run_example())
