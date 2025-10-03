import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
import aiohttp
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    input_guardrail,
    output_guardrail,
    function_tool,
    GuardrailFunctionOutput,
    TResponseInputItem,
    RunContextWrapper,
    InputGuardrailTripwireTriggered,
)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


if not GEMINI_API_KEY and not SERPAPI_KEY:
    raise ValueError("Missing GEMINI_API_KEY")

# Configure client for Gemini/Vertex AI
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- Guardrail Schemas ---
class EcommerceInputCheck(BaseModel):
    reason: str
    is_ecommerce: bool

class MessageOutput(BaseModel):
    response: str

class MaxTokensCheck(BaseModel):
    is_max: bool 
    reason: str

# --- Filter Agent Definition ---
input_guardrail_agent = Agent(
    name="Input Guardrail Check",
    instructions="""
You are a quick filter that ONLY allows ecommerce or business operations queries.
Accept any query about:
  • Dropshipping (planning, starting, scaling)
  • Product sourcing or supplier selection
  • Pricing, profit margins, cost analyses
  • Store setup or migrations (Shopify, WooCommerce, etc.)
  • Marketing (ads on Facebook, Instagram, TikTok)
  • Optimization (conversion rates, A/B testing)
  • Fulfillment, shipping, logistics
  • Payment integrations
  • Market research, product validation

Return JSON {"is_ecommerce": bool, "reason": "brief justification"}.
""",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    ),
    output_type=EcommerceInputCheck,
)

@input_guardrail
async def input_domain_filter(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)
    # Tripwire only when NOT ecommerce
    trip = not result.final_output.is_ecommerce
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=trip,
    )

@output_guardrail
async def output_token_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    response_data: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    text = response_data if isinstance(response_data, str) else ""
    word_count = len(text.split())
    estimated_tokens = int(word_count * 1.33)

    if estimated_tokens > 1000:
        print(f"⚠️ Too many tokens! Approx: {estimated_tokens} tokens")
        # Return an *instance* of your schema, not the class itself
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info=MaxTokensCheck(
                is_max=True,
                reason=f"Estimated {estimated_tokens} tokens exceeds 1000-token guardrail"
            )
        )

    return GuardrailFunctionOutput(tripwire_triggered=False, output_info=None)


# @rate_limited(3, 5)  # 3 calls every 5 seconds
@function_tool
async def fetch_google_trends(topic: str) -> dict:
    """Get real Google Trends data for product validation"""
    params = {
        "api_key": SERPAPI_KEY,
        "engine": "google_trends",
        "q": topic,
        "data_type": "TIMESERIES",
        "tz": "180"  # UTC+3
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get("https://serpapi.com/search", params=params) as response:
            data = await response.json()
            
            # Process trend data
            interest = []
            if 'interest_over_time' in data:
                for point in data['interest_over_time']:
                    interest.append(point['value'])
                    
            return {
                "topic": topic,
                "interest_over_time": interest,
                "peak_interest": max(interest) if interest else 0,
                "trending": any(value > 70 for value in interest[-3:])
            }



# --- Fallback Agent for Irrelevant Queries ---
fallback_agent = Agent(
    name="Fallback Help Agent",
    instructions="""
You are a helpful assistant for StorePilot AI. When a user asks anything outside of ecommerce or business operations, explain the purpose of this platform (which is to guide and automate end-to-end ecommerce store setup and optimization) and then answer the user's original query as best as you can.
Respond with a friendly explanation and the requested information.
""",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    ),
    output_type=MessageOutput,
)

# --- Orchestrator Agent ---
orchestrator = Agent(
    name="EcommerceOrchestrator",
    instructions="""
Manage end-to-end ecommerce tasks: platform choice, market research, store setup, product validation, and marketing. 
When the user asks if a product or topic is trending, always call the `fetch_google_trends` tool first. If that tool fails, say so briefly and ask for a different keyword or offer another check. 
If the query is not about ecommerce, route the user to the fallback agent and give a short reason. 
Keep answers short (about 1000 tokens or less).
""",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    ),
    tools=[fetch_google_trends],
    input_guardrails=[input_domain_filter],
    output_guardrails=[output_token_guardrail],
    output_type=MessageOutput,
)



async def main():
    try:
        user_input = input("\nUser: ")
        final = await Runner.run(
            starting_agent=orchestrator,
            input=user_input,

        )
        print(f"\n🤖 StorePilot AI response: {final.final_output.response}")
    except InputGuardrailTripwireTriggered:
        # Handle irrelevant queries with fallback agent
        fallback = await Runner.run(fallback_agent, user_input)
        print(f"\n🤖 StorePilot Help: {fallback.final_output.response}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
