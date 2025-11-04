from pydantic import BaseModel
import os
import json
import re
import pickle
import base64
import requests
import certifi
import logging
from dotenv import load_dotenv
from email.mime.text import MIMEText
from urllib.parse import urljoin
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
logging.basicConfig(level=logging.INFO)


class LeadInput(BaseModel):
    user_text: str
    assigned_to_id: str  # Assignee selected by user


class EmailInput(BaseModel):
    to_email: str
    subject: str
    body_text: str


ASSIGNEES = [
    {"id": "7294", "name": "Kundan", "email": "bitise8899@gmail.com"},
    {"id": "7319", "name": "Nikhil", "email": "ravi.patel@example.com"},
    {"id": "7295", "name": "Nisha Verma", "email": "nisha.verma@example.com"},
]


GMAIL_CREDENTIALS_FILE = "client_secret.json"
TOKEN_PICKLE = "token_gmail.pickle"
SCOPES = [""] # use your Scopes



CRM_API_URL = os.getenv("CRM_API_URL", ) # use your CRm API
CRM_API_KEY = os.getenv("CRM_API_KEY")


def update_certifi_certificates():
    try:
        return certifi.where()
    except Exception:
        return None


def gmail_authenticate():
    creds = None
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GMAIL_CREDENTIALS_FILE):
                raise FileNotFoundError(f"{GMAIL_CREDENTIALS_FILE} not found.")
            flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PICKLE, "wb") as token:
            pickle.dump(creds, token)
    return build("gmail", "v1", credentials=creds)


def create_message(to_email: str, subject: str, body_text: str):
    message = MIMEText(body_text)
    message["to"] = to_email
    message["from"] = "me"
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {"raw": raw}


def send_email(to_email: str, subject: str, body_text: str):
    try:
        service = gmail_authenticate()
        message = create_message(to_email, subject, body_text)
        return service.users().messages().send(userId="me", body=message).execute()
    except Exception as e:
        return {"error": str(e)}


def save_lead_to_crm(lead: dict) -> dict:
    payload = {
        "org": lead.get("Org", ""),
        "city": lead.get("City", ""),
        "email": lead.get("Email", ""),
        "phone": lead.get("Phone", ""),
        "state": lead.get("State", ""),
        "gst_no": lead.get("GSTNo", ""),
        "address": lead.get("Address", ""),
        "country": lead.get("Country", "India"),
        "pincode": lead.get("Pincode", ""),
        "assigned_to": lead.get("AssignedTo", "crmsuperadmin"),
        "description": lead.get("Summary", ""),
        "designation": lead.get("Designation", ""),
        "lead_source": lead.get("Source", "Trade Show"),
        "lead_status": lead.get("Status", "New"),
        "contact_name": lead.get("Name", ""),
        "industry_type": lead.get("Industry", "Other"),
        "gst_state_code": lead.get("GSTStateCode", "")
    }
    ca_bundle_path = update_certifi_certificates()
    session = requests.Session()
    headers = {"Content-Type": "application/json", "Referer": CRM_API_URL}
    if CRM_API_KEY:
        headers["api-key"] = f"{CRM_API_KEY}"
    try:
        try:
            session.get(urljoin(CRM_API_URL, "/sanctum/csrf-cookie"), verify=ca_bundle_path, timeout=10)
        except requests.exceptions.RequestException:
            pass

        xsrf_token = next((c.value for c in session.cookies if c.name.lower() in ("xsrf-token", "xsrftoken")), None)
        if xsrf_token:
            headers["X-XSRF-TOKEN"] = xsrf_token

        resp = session.post(CRM_API_URL, headers=headers, json=payload, verify=ca_bundle_path, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_lead_from_llm(user_input: str) -> dict:
    prompt = PromptTemplate(
        template="""
You are a lead extractor. Extract the following fields from the user input and RETURN ONLY a JSON object (no explanation).
Keys: "Name", "Org", "Email", "Phone", "Source", "Status":"Open", "Summary".
If a field is missing, set its value to an empty string, Only validate Email should be taken, if there is invalidate email ask user to enter valid mail.
Extract from this input:
{user_input}
""",
        input_variables=["user_input"],
    )
    formatted_prompt = prompt.format(user_input=user_input)

    model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)

    llm = ChatHuggingFace(llm=model)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    text = response.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"LLM did not return parsable JSON:\n{text}")


def format_lead_assignment_email(assignee_name: str, lead_data: dict) -> str:
    return f"""
Dear {assignee_name},

A new lead has been assigned to you in CRM.
Details:
- Name: {lead_data.get('Name', '')}
- Organization: {lead_data.get('Org', '')}
- Email: {lead_data.get('Email', '')}
- Phone: {lead_data.get('Phone', '')}
- Lead Source: {lead_data.get('Source', '')}
- Status: {lead_data.get('Status', '')}
- Summary: {lead_data.get('Summary', '')}

Please log into the CRM dashboard to review and manage this lead.

Regards,
CRM Team
"""


# Example usage when sending an email after lead extraction and assignment
def process_and_notify(user_text: str, assigned_to_id: str):
    # Extract lead data from LLM
    lead_data = get_lead_from_llm(user_text)
    
    # Save lead to CRM
    lead_data["AssignedTo"] = assigned_to_id
    save_response = save_lead_to_crm(lead_data)

    # Find assignee details
    assignee = next((a for a in ASSIGNEES if a["id"] == assigned_to_id), None)
    if not assignee:
        logging.error(f"No assignee found with id {assigned_to_id}")
        return {"error": "Assignee not found"}

    # Prepare email
    subject = f"New Lead Assigned: {lead_data.get('Name', 'Unnamed Lead')}"
    body_text = format_lead_assignment_email(assignee["name"], lead_data)

    # Send email notification
    email_result = send_email(assignee["email"], subject, body_text)
    return {"crm_response": save_response, "email_response": email_result}
