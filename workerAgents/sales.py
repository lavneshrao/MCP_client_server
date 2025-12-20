import asyncio
import json
import httpx

from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from mcp.client.session import ClientSession

from sharedState.state import OrchestratorState

load_dotenv()

class NegotiatedOffer(BaseModel):
    customer_id: str = Field(description="Unique identifier for the customer")
    approved_amount: int = Field(description="Amount upto which NBFC is allowed to give up personal loan")
    tenure_months: int = Field(description="Period in months for which loan is to be taken")
    interest_rate: float = Field(description="Interest rate upto which loan can be approved")
    justification: str = Field(description="Justification for the approval or rejection of deal")


async def build_sales_graph(session:ClientSession):

    llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            max_output_tokens=2048,
        ).with_structured_output(NegotiatedOffer)

    # NODE 1 → FETCH CUSTOMER INFO
    async def fetch_customer_info(state: OrchestratorState) -> OrchestratorState:
        cid = str(state["customer_id"])
        try:
            result = await session.call_tool(
                "get_customer_info",
                arguments={"customer_id": cid}
            )
            updates = {}
            customer_data = json.loads(result.content[0].text)
            if "result" in customer_data and isinstance(customer_data["result"], dict):
                updates["customer_info"] = customer_data["result"]
            else:
                 updates["customer_info"] = customer_data

        except httpx.ConnectError:
            print("RROR: Connection Refused. (Likely the 0.0.0.0 bug)")
            updates["customer_info"] = {"error": "Connection Refused"}
            
        except httpx.TimeoutException:
            print("ERROR: Timed Out. (Server is blocked or slow)")
            updates["customer_info"] = {"error": "Timeout"}
            
        except Exception as e:
            print(f"ERROR: Generic Exception -> {type(e).__name__}: {e}")
            updates["customer_info"] = {"error": str(e)}

        return updates

    # NODE 2 → LLM_NODE 
    async def llm_node(state: OrchestratorState):
        customer = state["customer_info"]
        requested = state["requested_amount"]
        tenure = state["preferred_tenure_months"]
        max_rate = state.get("max_interest_rate", 25.0)

        context = {
            "customer_info": customer,
            "requested_amount": requested,
            "preferred_tenure_months": tenure,
            "max_interest_rate": max_rate,
        }

        inputMessage = f"""
        You are an NBFC Sales Agent. 
        negotiate the offer based on the given rules by utilising the given context.

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

        "context": {json.dumps(context, indent=2)}
        """

        existing_history = state.get("sales_messages", [])

        result = await llm.ainvoke(existing_history + [HumanMessage(content= inputMessage)])

        negotiated_offer = result.model_dump()
        public_history = AIMessage(
            content=f"Sales Agent: Generated offer of {negotiated_offer['approved_amount']} at {negotiated_offer['interest_rate']}% interest.",
            name="SalesAgent"
        )
        private_history = [HumanMessage(content= inputMessage), AIMessage(content=str(negotiated_offer))]

        return {"negotiated_offer" : negotiated_offer,
                "messages" : [public_history],
                "sales_messages" : private_history}
    

    graph = StateGraph(OrchestratorState)

    graph.add_node("fetch_customer_info", fetch_customer_info)
    graph.add_node("llm_node", llm_node)
    

    graph.add_edge(START, "fetch_customer_info")
    graph.add_edge("fetch_customer_info", "llm_node")
    graph.add_edge("llm_node", END)

    return graph.compile()

