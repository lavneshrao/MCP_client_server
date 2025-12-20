from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class OrchestratorState(TypedDict, total=False):
    customer_id: str

    flow_stage: str  # start, negotiation, verification, underwriting, sanction, complete
    next_worker: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    
    customer_info: Dict[str, Any]
    user_feedback: str
    
    requested_amount: Optional[int]
    preferred_tenure_months: Optional[int]
    max_interest_rate: Optional[float]
    negotiated_offer: Dict[str, Any]
    
    kyc_result: Dict[str, Any]
    kyc_status: str
    
    credit_score: Optional[int]
    salary_slip_resource: Optional[str]
    underwriting_input: Dict[str, Any]
    underwriting_result: Dict[str, Any]
    underwriting_status: str
    
    sanction_letter_resource: Optional[str]
    sanction_letter_path: Optional[str]
    sanction_letter_status: Optional[str]

    sales_messages: Annotated[List[BaseMessage], add_messages]
    verification_messages: Annotated[List[BaseMessage], add_messages]
    underwriting_messages: Annotated[List[BaseMessage], add_messages]
    sanction_messages: Annotated[List[BaseMessage], add_messages]