# server.py  – NBFC MCP server using FastMCP (Python MCP SDK)

import os
import json
import uuid
import base64
from typing import Any, Dict, Optional

from datetime import datetime

from mcp.server.fastmcp import FastMCP, ToolError

# Config
STORAGE_DIR = os.environ.get("MCP_STORAGE_DIR", "./storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Create MCP server instance
mcp = FastMCP("NBFC MCP Server", json_response=True)

# --- Mock data: 10 synthetic customers (same as before)
CUSTOMERS: Dict[str, Dict[str, Any]] = {
    "CUST001": {"customer_id":"CUST001","name":"Asha Verma","age":32,"city":"Pune","phone":"9810000001","email":"asha@example.com","pre_approved_limit":300000,"salary_monthly":60000,"credit_score":745},
    "CUST002": {"customer_id":"CUST002","name":"Rahul Sharma","age":29,"city":"Delhi","phone":"9810000002","email":"rahul@example.com","pre_approved_limit":200000,"salary_monthly":45000,"credit_score":712},
    "CUST003": {"customer_id":"CUST003","name":"Sneha Iyer","age":35,"city":"Bengaluru","phone":"9810000003","email":"sneha@example.com","pre_approved_limit":400000,"salary_monthly":85000,"credit_score":780},
    "CUST004": {"customer_id":"CUST004","name":"Vikram Singh","age":40,"city":"Lucknow","phone":"9810000004","email":"vikram@example.com","pre_approved_limit":150000,"salary_monthly":30000,"credit_score":690},
    "CUST005": {"customer_id":"CUST005","name":"Nisha Patel","age":27,"city":"Ahmedabad","phone":"9810000005","email":"nisha@example.com","pre_approved_limit":250000,"salary_monthly":52000,"credit_score":710},
    "CUST006": {"customer_id":"CUST006","name":"Arjun Rao","age":31,"city":"Hyderabad","phone":"9810000006","email":"arjun@example.com","pre_approved_limit":350000,"salary_monthly":70000,"credit_score":760},
    "CUST007": {"customer_id":"CUST007","name":"Meera Desai","age":30,"city":"Surat","phone":"9810000007","email":"meera@example.com","pre_approved_limit":180000,"salary_monthly":40000,"credit_score":695},
    "CUST008": {"customer_id":"CUST008","name":"Karan Mehta","age":33,"city":"Mumbai","phone":"9810000008","email":"karan@example.com","pre_approved_limit":320000,"salary_monthly":65000,"credit_score":735},
    "CUST009": {"customer_id":"CUST009","name":"Priya Nair","age":28,"city":"Kochi","phone":"9810000009","email":"priya@example.com","pre_approved_limit":280000,"salary_monthly":48000,"credit_score":725},
    "CUST010": {"customer_id":"CUST010","name":"Sourav Ghosh","age":36,"city":"Kolkata","phone":"9810000010","email":"sourav@example.com","pre_approved_limit":500000,"salary_monthly":90000,"credit_score":790},
}

# --- Helper: EMI calculation (same math as before)
def compute_emi(P: float, annual_rate: float, n_months: int) -> float:
    r = annual_rate / 12.0 / 100.0
    if r == 0:
        return P / n_months
    num = P * r * (1 + r) ** n_months
    den = (1 + r) ** n_months - 1
    return num / den

# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
def get_customer_info(customer_id: str) -> Dict[str, Any]:
    """Fetch customer basic info."""
    cust = CUSTOMERS.get(customer_id)
    if not cust:
        raise ToolError(f"customer not found: {customer_id}")
    return {"status": "ok", "result": cust}


@mcp.tool()
def verify_kyc(customer_id: str, phone: str) -> Dict[str, Any]:
    """Verify phone (and mock-address) for a customer."""
    cust = CUSTOMERS.get(customer_id)
    if not cust:
        raise ToolError(f"customer not found: {customer_id}")

    phone_verified = cust.get("phone") == phone
    # Address verification is mocked as always True, same as original
    return {
        "status": "ok",
        "result": {
            "phone_verified": phone_verified,
            "address_verified": True,
        },
    }


@mcp.tool()
def get_credit_score(customer_id: str) -> Dict[str, Any]:
    """Return credit score for customer."""
    cust = CUSTOMERS.get(customer_id)
    if not cust:
        raise ToolError(f"customer not found: {customer_id}")
    return {
        "status": "ok",
        "result": {"credit_score": cust.get("credit_score")},
    }


@mcp.tool()
def underwrite_loan(
    customer_id: str,
    requested_amount: int,
    tenure_months: int = 36,
    annual_rate: float = 12.0,
    salary_provided: Optional[int] = None,
    salary_slip_resource: Optional[str] = None,
) -> Dict[str, Any]:
    """Underwriting decision using stated rules."""
    cust = CUSTOMERS.get(customer_id)
    if not cust:
        raise ToolError(f"customer not found: {customer_id}")

    score = cust.get("credit_score", 0)
    pre_limit = cust.get("pre_approved_limit", 0)
    requested = requested_amount
    tenure = tenure_months

    if score < 700:
        return {
            "status": "ok",
            "result": {
                "decision": "reject",
                "reason": "credit_score_below_700",
                "credit_score": score,
            },
        }

    if requested <= pre_limit:
        emi = compute_emi(requested, annual_rate, tenure)
        return {
            "status": "ok",
            "result": {
                "decision": "approve",
                "emi": emi,
                "reason": "within_pre_approved_limit",
            },
        }

    if requested <= 2 * pre_limit:
        if not salary_slip_resource and salary_provided is None:
            return {
                "status": "ok",
                "result": {
                    "decision": "require_salary_slip",
                    "reason": "salary_slip_required",
                },
            }

        salary = (
            salary_provided
            if salary_provided is not None
            else cust.get("salary_monthly", 0)
        )
        emi = compute_emi(requested, annual_rate, tenure)

        if emi <= 0.5 * salary:
            return {
                "status": "ok",
                "result": {
                    "decision": "approve",
                    "emi": emi,
                    "reason": "emi_within_50pct_salary",
                },
            }
        else:
            return {
                "status": "ok",
                "result": {
                    "decision": "reject",
                    "reason": "emi_exceeds_50pct_salary",
                    "emi": emi,
                    "salary_monthly": salary,
                },
            }

    return {
        "status": "ok",
        "result": {
            "decision": "reject",
            "reason": "amount_exceeds_2x_pre_approved",
            "pre_limit": pre_limit,
            "requested": requested,
        },
    }


@mcp.tool()
def upload_salary_slip(
    customer_id: str,
    filename: str,
    content_base64: str,
) -> Dict[str, Any]:
    """
    Upload a salary slip for a customer.

    Note: in FastAPI this was multipart/form-data.
    In MCP we pass base64-encoded file content instead.
    """
    if customer_id not in CUSTOMERS:
        raise ToolError(f"customer not found: {customer_id}")

    ext = os.path.splitext(filename)[1] or ".pdf"
    stored_name = f"salary_{customer_id}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(STORAGE_DIR, stored_name)

    try:
        raw = base64.b64decode(content_base64)
    except Exception as e:
        raise ToolError(f"invalid base64 content: {e}")  # visible to client

    with open(path, "wb") as f:
        f.write(raw)

    resource_url = f"resource://{stored_name}"
    return {
        "status": "ok",
        "result": {
            "resource": resource_url,
            "path": path,
        },
    }


# PDF generation needs reportlab, same as your FastAPI code
from reportlab.pdfgen import canvas  # type: ignore


@mcp.tool()
def generate_sanction_letter(
    customer_id: str,
    amount: int,
    tenure_months: int = 36,
    interest_rate: float = 12.0,
) -> Dict[str, Any]:
    """Generate a sanction letter PDF and return a resource URL."""
    cust = CUSTOMERS.get(customer_id)
    if not cust:
        raise ToolError(f"customer not found: {customer_id}")

    filename = f"sanction_{customer_id}_{uuid.uuid4().hex}.pdf"
    path = os.path.join(STORAGE_DIR, filename)

    c = canvas.Canvas(path)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Sanction Letter")
    c.setFont("Helvetica", 11)
    c.drawString(50, 770, f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}")
    c.drawString(50, 750, f"Customer: {cust.get('name')} (ID: {customer_id})")
    c.drawString(50, 730, f"Approved Amount: INR {amount}")
    c.drawString(50, 710, f"Tenure: {tenure_months} months")
    c.drawString(50, 690, f"Interest Rate (annual): {interest_rate}%")
    c.drawString(
        50,
        660,
        "This is a demo sanction letter generated by MCP server.",
    )
    c.save()

    resource_url = f"resource://{filename}"
    return {
        "status": "ok",
        "result": {
            "resource": resource_url,
            "path": path,
        },
    }


@mcp.tool()
def log_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Append an audit event to a log file."""
    log_path = os.path.join(STORAGE_DIR, "mcp_audit.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"ts": datetime.utcnow().isoformat(), "event": event},
                ensure_ascii=False,
            )
            + "\n"
        )
    return {"status": "ok"}


@mcp.tool()
def health() -> Dict[str, Any]:
    """Simple health check."""
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# ---------------------------------------------------------------------------
# RESOURCES
# ---------------------------------------------------------------------------

@mcp.resource("resource://{filename}")
def fetch_resource(filename: str) -> bytes:
    """
    Return the raw bytes of a stored resource (PDF, salary slip, etc.).
    """
    path = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(path):
        raise ToolError(f"resource not found: {filename}")

    with open(path, "rb") as f:
        return f.read()

# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # HTTP transport is convenient for local testing & MCP Inspector
    # You can also use the default stdio transport if you prefer.
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, path="/mcp")
