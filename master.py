from typing import Literal, Optional
from pydantic import BaseModel,Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI as genai
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import AIMessage, HumanMessage

from sharedState.state import OrchestratorState

from workerAgents.sales import build_sales_graph
from workerAgents.verification import build_verification_graph
from workerAgents.underwriting import build_underwriting_graph
from workerAgents.sanction import build_sanction_graph

load_dotenv()

class Router(BaseModel):
    response_to_user: str = Field(description="The natural language response to the user")
    next_worker: Literal["sales", "verification", "underwriting", "sanction", "none"] = Field(description="The next worker to call, or 'none' if waiting for user")
    
    # Extraction Fields
    update_customer_id: Optional[str] = Field(description="Extracted Customer ID if present", default=None)
    update_requested_amount: Optional[int] = Field(description="Extracted loan amount if present", default=None)
    update_preferred_tenure: Optional[int] = Field(description="Extracted tenure in months if present", default=None)
    update_max_interest: Optional[float] = Field(description="New max interest rate if user is negotiating", default=None)


async def build_orchestrator_graph(session):

    llm = genai(
        model="gemini-2.0-flash",
        temperature=0.2,
        max_output_tokens= 2048)
    

    sales_graph = await build_sales_graph(session)
    verification_graph = await build_verification_graph(session)
    underwriting_graph = await build_underwriting_graph(session)
    sanction_graph = await build_sanction_graph(session)

    async def master_llm_node(state: OrchestratorState):
        messages = state.get("messages", [])
        flow_stage = state.get("flow_stage", "start")
        negotiated_offer = state.get("negotiated_offer", {})

        has_amount = state.get("requested_amount") is not None

        system_prompt = f"""
        You are the Senior Loan Manager (Master Agent) at an NBFC. 
        Your goal: Guide the user from ID collection -> Negotiation -> Verification -> Underwriting -> Sanction.
        
        Current Flow Stage: {flow_stage}
        
        DATA CONTEXT:
        - Customer ID: {state.get('customer_id')}
        - Requested Amount: {state.get('requested_amount')}
        - Negotiated Offer: {negotiated_offer}
        - KYC Status: {state.get('kyc_status')}
        - Underwriting Status: {state.get('underwriting_status')}
        
        RULES:
        1. **START**: 
           - If no Customer ID, ask for it. 
           - If user provides ID (e.g. "CUST001"), extract it.
           
        2. **NEGOTIATION PREP**:
           - Once you have ID, check if 'Requested Amount' is present.
           - If MISSING, ask the user: "How much loan amount do you need and for what tenure?"
           - DO NOT call 'sales' until you have the Amount.

        3. **NEGOTIATION (Call Sales)**: 
           - If you have ID AND Amount, set next_worker='sales'.
           - If offer exists and user REJECTS it, ask for new terms, update state, and set next_worker='sales'.
           - If user ACCEPTS, set next_worker='verification'.

        4. **VERIFICATION**: 
           - If status is 'pending' or 'failed', set next_worker='verification'.
           - If 'verified', inform user and set next_worker='underwriting'.

        5. **UNDERWRITING**:
           - If status 'need_documents', ask user to upload salary slip.
           - If 'approve', set next_worker='sanction'.
           - If 'reject', end conversation.

        6. **SANCTION**:
           - If letter generated, show link.
        
        OUTPUT FORMAT:
        Decide the 'next_worker' and the 'response_to_user'.
        """

        chatbot = llm.with_structured_output(Router)

        response = await chatbot.ainvoke([HumanMessage(content=system_prompt)]+messages)

        updates = {
            "messages": [AIMessage(content=response.response_to_user)],
            "next_worker": response.next_worker
        }

        if response.update_customer_id:
            updates["customer_id"] = response.update_customer_id
            updates["flow_stage"] = "negotiation"
        
        if response.update_requested_amount:
            updates["requested_amount"] = response.update_requested_amount
        
        if response.update_preferred_tenure:
            updates["preferred_tenure_months"] = response.update_preferred_tenure

        if response.update_max_interest:
            updates["max_interest_rate"] = response.update_max_interest
            updates["negotiated_offer"] = {}

        if response.next_worker == "sales": updates["flow_stage"] = "negotiation"
        if response.next_worker == "verification": updates["flow_stage"] = "verification"
        if response.next_worker == "underwriting": updates["flow_stage"] = "underwriting"
        if response.next_worker == "sanction": updates["flow_stage"] = "sanction"

        return updates
    
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("master", master_llm_node)
    workflow.add_node("sales", sales_graph)
    workflow.add_node("verification", verification_graph)
    workflow.add_node("underwriting", underwriting_graph)
    workflow.add_node("sanction", sanction_graph)

    workflow.add_edge(START, "master")

    def route_next(state: OrchestratorState):
        next_worker = state.get("next_worker", "none")
        if next_worker == "sales": return "sales"
        if next_worker == "verification": return "verification"
        if next_worker == "underwriting": return "underwriting"
        if next_worker == "sanction": return "sanction"
        return END
    
    workflow.add_conditional_edges("master", route_next)
    workflow.add_edge("sales","master")
    workflow.add_edge("verification","master")
    workflow.add_edge("underwriting","master")
    workflow.add_edge("sanction","master")

    return workflow.compile()


