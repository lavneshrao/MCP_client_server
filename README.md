1. Structure: one master agent, orchestrator, four worker agents
2. Keep the structure of the state same for all the agents and store this state in the orchestrator such that all the agents can perform read and write operations on it. This state must cover all the fields.
3. Firstly the master agent start the conversation and fill the customer_id field in the state along with performing regular chatbot operations.
4. Sales Agent: This customer_id is then taken up by the sales agent and using the get_customer_info tool in the mcp server it fetches the and write the negotiation details required in the orchestration state.
5. Master agent then seeks approval of the user and update the same in the status field of the state. if status is not convinced then orchestrator goes back to sales agent to update the negotiation terms.
6. Once the status field in orchestration state is confirmed by the user, the orchestrator moves to verification agent.
7. Verification Agent: The verification agent uses the verify_kyc tool from the mcp server to verify the user and update the same on the orchestration shared state.
8. Once verified the process further moves on to the underwriting agent.
9. Underwriting Agent: The underwriting agent uses tools such as get_credit_score and upload_salary_slip to get the details and perform the actions as mentioned in the tool functions. 
10. As per the results and acceptability of the offer the master agent seeks final acceptance of the loan from the customer/user on getting the confirmation the Sanction Letter Generator agent start working.
11. Sanction Letter Generator: After getting the complete approval it generates a pdf by using the generate_sanction_letter mcp tool.
12. At last the master agent congratulates and give the sanction letter to the user

SHARED STATE:

from typing import TypedDict, Dict, Any, Optional, List


class OrchestrationState(TypedDict, total=False):
    # -------------------------------------------------------
    # 1. MASTER AGENT / USER INPUT
    # -------------------------------------------------------
    customer_id: str                       # Filled by master agent when user provides ID
    user_query: str                        # Raw user input for conversation continuity
    status: str                            # negotiation_status: "pending" | "approved" | "rejected"

    # -------------------------------------------------------
    # 2. CUSTOMER DATA (from MCP: get_customer_info)
    # -------------------------------------------------------
    customer_info: Dict[str, Any]          # Full customer dict returned by MCP server

    # -------------------------------------------------------
    # 3. NEGOTIATION (Sales Agent)
    # -------------------------------------------------------
    requested_amount: Optional[int]        # Customer-requested amount
    preferred_tenure_months: Optional[int] # Customer preferred tenure
    max_interest_rate: Optional[float]     # User threshold

    negotiated_offer: Dict[str, Any]       # {
                                           #   "customer_id": str,
                                           #   "approved_amount": int,
                                           #   "tenure_months": int,
                                           #   "interest_rate": float,
                                           #   "justification": str
                                           # }

    negotiation_round: int                 # increment when user rejects offer → agent renegotiates

    # -------------------------------------------------------
    # 4. KYC VERIFICATION (Verification Agent)
    # -------------------------------------------------------
    kyc_result: Dict[str, Any]             # {
                                           #   "phone_verified": bool,
                                           #   "address_verified": bool
                                           # }
    kyc_status: str                        # "pending" | "verified" | "failed"

    # -------------------------------------------------------
    # 5. UNDERWRITING (Underwriting Agent)
    # -------------------------------------------------------
    credit_score: Optional[int]            # fetched from get_credit_score
    salary_slip_resource: Optional[str]    # resource://<file> link after upload
    underwriting_input: Dict[str, Any]     # the exact MCP input payload
    underwriting_result: Dict[str, Any]    # MCP underwriting output
    underwriting_status: str               # "pending" | "approved" | "rejected" | "need_documents"

    # -------------------------------------------------------
    # 6. SANCTION LETTER (Sanction Letter Generator Agent)
    # -------------------------------------------------------
    sanction_letter_resource: Optional[str]  # resource://<pdf>
    sanction_letter_path: Optional[str]      # local server path for debugging

    # -------------------------------------------------------
    # 7. SYSTEM FLOW CONTROL (used by orchestrator)
    # -------------------------------------------------------
    flow_stage: str                        # one of:
                                           # "start"
                                           # "collect_customer_id"
                                           # "negotiation"
                                           # "await_user_confirmation"
                                           # "verification"
                                           # "underwriting"
                                           # "sanction_generation"
                                           # "complete"

    messages: List[Dict[str, Any]]         # optional conversation history for master agent

    # -------------------------------------------------------
    # 8. LOGGING / AUDIT
    # -------------------------------------------------------
    log_events: List[Dict[str, Any]]       # track events locally before sending to MCP log_event

   
