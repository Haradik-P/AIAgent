from fastapi import FastAPI, HTTPException
import json
from database import run_query
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware
from chat import (
    LeadInput,
    EmailInput,
    ASSIGNEES,
    get_lead_from_llm,
    send_email,
    save_lead_to_crm,
    format_lead_assignment_email,
)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your client's origin for production e.g. ["https://yourwebsite.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, including OPTIONS
    allow_headers=["*"],  # Allow all headers
)
logging.basicConfig(level=logging.INFO)


@app.post("/extract-lead")
async def extract_lead_and_process(lead_input: LeadInput):
    try:
        lead_data = get_lead_from_llm(lead_input.user_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lead extraction failed: {str(e)}")

    missing_fields = [f for f in ["Name", "Email", "Phone", "Org"] if not lead_data.get(f)]
    if missing_fields:
        return {
            "error": "Missing required fields in lead data",
            "missing_fields": missing_fields,
            "lead_data": lead_data,
        }

    assignee = next((a for a in ASSIGNEES if a["id"] == lead_input.assigned_to_id), None)
    if not assignee:
        raise HTTPException(status_code=400, detail="Invalid assignee ID provided")

    assigned_email = assignee["email"]
    logging.info(f"Sending email to assigned person: {assigned_email}")

    # Use formatted email body for better readability
    body_text = format_lead_assignment_email(assignee["name"], lead_data)
    subject = f"New Lead Assigned: {lead_data.get('Name', 'Unnamed Lead')}"

    email_result = send_email(assigned_email, subject, body_text)
    if "error" in email_result:
        raise HTTPException(status_code=400, detail=f"Email sending failed: {email_result['error']}")

    lead_data["AssignedTo"] = assignee["name"]
    save_result = save_lead_to_crm(lead_data)
    if "error" in save_result:
        raise HTTPException(status_code=400, detail=f"Saving lead failed: {save_result['error']}")

    return {
        "lead_data": lead_data,
        "email_message_id": email_result.get("id"),
        "crm_response": save_result,
    }

# ✅ Pydantic model for request
class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_database(request: QueryRequest):
    try:
        result = run_query(request.query)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

