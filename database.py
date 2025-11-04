from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
import json
import os

# -------------------- Load environment --------------------
load_dotenv()

# -------------------- Initialize Model and DB --------------------
model = ChatOpenAI()

db_uri = (
    f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}"
)
db = SQLDatabase.from_uri(db_uri)

# Create the SQL Agent
agent = create_sql_agent(llm=model, db=db, verbose=False,handle_parsing_errors=True)

# -------------------- Run Query Using Agent Only --------------------
def run_query(user_input: str):
    """
    Runs the user query directly through LangChain SQL Agent
    and returns plain, human-readable text.
    """
    try:
        agent_result = agent.invoke({"input": f"Answer using DB: {user_input}"})
        # ---- Extract clean text ----
        if isinstance(agent_result, dict):
            if "data" in agent_result and isinstance(agent_result["data"], str):
                output_text = agent_result["data"]
            elif "output" in agent_result and isinstance(agent_result["output"], str):
                output_text = agent_result["output"]
            else:
                output_text = json.dumps(agent_result, indent=2)
        else:
            output_text = str(agent_result)

        # ---- Clean unwanted wrappers ----
        output_text = (
            output_text.replace("{", "")
            .replace("}", "")
            .replace('"status": "success",', "")
            .replace('"data":', "")
            .strip()
        )

        # Remove leading/trailing quotes or commas
        output_text = output_text.strip('"').strip().strip(",")

        return output_text or "No data found."

    except Exception as e:
        return f"⚠️ Agent query failed: {e}"
