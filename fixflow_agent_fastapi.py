from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
import os
import re
import json
import logging

# OpenAI Agents SDK imports (adjust names per installed SDK)
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    SQLiteSession,
    set_tracing_disabled,
    input_guardrail,
    RunContextWrapper,
    GuardrailFunctionOutput,
    TResponseInputItem,
    InputGuardrailTripwireTriggered
)

load_dotenv()
set_tracing_disabled(disabled=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY (or OPENAI_API_KEY) not set in environment")

DB_PATH = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")

# small logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fixflow-agent")

# configure external client & model
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

class AgentInputGuardrailOUtput(BaseModel):
    unsafe: bool
    reason: str


bank_input_guard_agent = Agent(
    name = "Bank InputGuardrail Agent",
    instructions= (
    """Check if the user input is a valid Habib Bank complaint.
        Also detect profanity or sensitive personal info."""
    ),
    model=model,
    output_type=AgentInputGuardrailOUtput,
)

@input_guardrail
async def bank_input_guard(ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(bank_input_guard_agent, input, context=ctx.context)

    # Directly pass AI output to GuardrailFunctionOutput
    return GuardrailFunctionOutput(
        output_info=result.final_output,  # full AI output
        tripwire_triggered=result.final_output.unsafe,  # AI decides unsafe
    )

# Agent instruction: (tweak for your UX)
BankFixFlowAgent = Agent(
    name="Bank FixFlow Agent",
    instructions=(
    "You are a specialized complaint-handling agent for Habib Bank, responsible for support and intake of customer issues. When user describes a problem, first try to provide a clear, actionable fix if it can be solved by steps (app login, PIN reset, blocking card, OTP issues, branch info). "
        "If you can give a reliable fix, do so and then ask: 'Do you want me to file an official complaint for this issue?' — do NOT produce a JSON complaint yet. "
        "If you cannot reliably solve the issue (fraud, ATM cash not dispensed, unauthorized deduction, loan dispute, staff misbehavior, unresolved backend issues), start complaint intake: ask only relevant questions, never guess values, and leave unknown fields null. "
        "Only when the user explicitly confirms 'submit' should you output EXACTLY ONE ```json``` block containing the complaint with fields: "
        "{'issue':'', 'branch_or_atm':null, 'date_time':null, 'amount':null, 'description':null, 'photos':[]}. auto deduct category and priority based on issue."
    ),
    model=model,
    input_guardrails=[bank_input_guard],
)

hospital_input_guard_agent = Agent(
    name = "Hospital InputGuardrail Agent",
    instructions= (
    """Check if the user input is a valid Indus Hospital complaint.
        Also detect profanity or sensitive personal info."""
    ),
    model=model,
    output_type=AgentInputGuardrailOUtput,
)

@input_guardrail
async def hospital_input_guard(ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(hospital_input_guard_agent, input, context=ctx.context)

    # Directly pass AI output to GuardrailFunctionOutput
    return GuardrailFunctionOutput(
        output_info=result.final_output,  # full AI output
        tripwire_triggered=result.final_output.unsafe,  # AI decides unsafe
    )


HospitalFixFlowAgent = Agent(
    name="HospitalFixFlowAgent",
    instructions=(
        "You are a hospital complaint Agent. "
        "If issue is simple (appointment, billing, reports, app login) → give fix steps. "
        "If serious (staff, treatment, negligence) → collect complaint info step by step: "
        "patient name, department, reason, issue, date/time, description, category, photos. "
        "Never guess, leave unknown as null. "
        "Only when user says 'submit', output one ```json``` with these fields."
    ),
    model=model,
    input_guardrails=[hospital_input_guard]
)

AGENT_MAP = {
    "bank": BankFixFlowAgent,
    "hospital": HospitalFixFlowAgent,
}

# MainRouterAgent = Agent(
#     name="MainAgent",
#     instructions=(
#         """You are an orchestrator. Based on the 'modal_name' provided from the frontend,
#     you will hand off the query directly to the matching specialized agent."""
#     ),
#     handoffs=[BankFixFlowAgent, HospitalFixFlowAgent],
#     model=model,
# )

# helper functions
def parse_json_code_fence(text: str) -> Optional[Dict[str, Any]]:
    """Extract a single JSON object from a ```json { ... } ``` code fence in text."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception as e:
        logger.warning("Failed to parse JSON code fence: %s", e)
        return None



# FastAPI app
app = FastAPI(title="FixFlow Agent API")

class MessageIn(BaseModel):
    session_id: str
    modal_name:str
    message: str

class AgentResponse(BaseModel):
    reply: str
    is_final: bool
    final_complaint: Optional[Dict[str, Any]] = None
    transcript: Optional[List[Dict[str, Any]]] = None

@app.post("/agent/message", response_model=AgentResponse)
async def agent_message(payload: MessageIn):
    """Send a user message to the agent. Returns agent reply and session transcript."""
    session_db_path = DB_PATH
    session_id = payload.session_id

    # create a per-session SQLiteSession (persistent file)
    session = SQLiteSession(session_id, session_db_path)

    # Pick agent from modal_name
    target_agent = AGENT_MAP.get(payload.modal_name.lower())
    if not target_agent:
        raise HTTPException(status_code=400, detail=f"Service not found. Unknown modal_name: {payload.modal_name}. Available services: bank, hospital.")
    
    # normal intake flow
    logger.info("Running FixFlowAgent for session %s", session_id, target_agent.name)
    try:
        result = await Runner.run(target_agent, input=payload.message, session=session)
    
    except InputGuardrailTripwireTriggered:
    # map agent to friendly label
        friendly_labels = {
            "bank": "Habib Bank",
            "hospital": "Indus Hospital",
            # future agents
            "school": "Beaconhouse School",
            "telecom": "Jazz Telecom",
            "airline": "PIA Airlines"
        }

        agent_label = friendly_labels.get(payload.modal_name.lower(), payload.modal_name.capitalize())

        reply = (
            f"I am here to assist with {agent_label} complaints only. "
            f"If your issue is related to {agent_label}, I can file the complaint for you. "
            )
        
        return AgentResponse(
        reply=reply,
        is_final=False,
        final_complaint=None,
        transcript=[]
    )

    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(e))

    output_text = result.final_output or ""

    # attempt to parse final JSON if agent emitted it
    final_json = parse_json_code_fence(output_text)
    is_final = final_json is not None

    # get transcript (list of stored messages)
    try:
        transcript = await session.get_items()
    except Exception:
        transcript = None

    return AgentResponse(
        reply=output_text,
        is_final=is_final,
        final_complaint=final_json,
        transcript=transcript,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fixflow_agent_fastapi:app", host="0.0.0.0", port=8000, reload=True)
