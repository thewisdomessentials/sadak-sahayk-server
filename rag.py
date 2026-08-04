import os
import base64
import json
import logging
from typing import Any

try:
    from .clients import get_qdrant_client, get_openai_client
    from .util import count_tokens, truncate_text
except ImportError:
    from clients import get_qdrant_client, get_openai_client
    from util import count_tokens, truncate_text

COLLECTION = os.getenv("QDRANT_COLLECTION", "sadaksahayak-documents")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

MAX_INPUT_TOKENS = 500
MAX_CONTEXT_TOKENS = 1500
MAX_OUTPUT_TOKENS = 300

logger = logging.getLogger(__name__)


def _safe_json_loads(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_language(language: str | None = None) -> str:
    ui_language = (language or "en").strip().lower()
    if ui_language not in {"en", "hi"}:
        ui_language = "en"
    return ui_language


def build_developer_prompt(
    context: str,
    language: str | None = None,
    conversation_history: str | None = None,
    expect_json: bool = False,
) -> str:
    ui_language = normalize_language(language)
    json_block = """
RESPONSE JSON CONTRACT
Return only valid JSON with this shape:
{
  "answer": "string",
  "needs_followup": true,
  "quick_replies": ["string"],
  "intent": "string"
}

Rules for JSON:
- Return valid JSON only.
- Do not wrap it in markdown fences.
- `quick_replies` can be an empty array.
- `needs_followup` must be true when the query is ambiguous and a clarifying choice would meaningfully change the answer.
- When `needs_followup` is true, `answer` should be a short clarifying question, and `quick_replies` must contain 2–4 clickable options.
- When `needs_followup` is false, `answer` must be the full structured response with the required headers.
- `intent` should be a short label like `penalty_lookup`, `violation_check`, `seizure_info`, or `general_info`.

FOLLOW-UP TRIGGER CONDITIONS (set needs_followup: true when ANY of these apply):
- Vehicle type is not mentioned AND the penalty differs across vehicle types (two-wheeler vs four-wheeler vs commercial).
- The offence type is vague and multiple violations could apply (e.g. "police stopped me" without reason).
- The penalty depends on whether it is a first offence or a repeat offence.
- The enforcement action depends on a specific circumstance the user has not stated (e.g. "can police seize my vehicle" without stating why).
- Engine number/Chassis number not provided in query and are needed to decide limits and fine for example Two-Wheeler Noise LimitsT(CPCB) regulates motorcycle and scooter noise based on engine displacement as follows:Up to 80 cc: Max limit of \(75 \text{ dB(A)}\)81 cc to 175 cc
DO NOT set needs_followup: true for:
- Queries that are complete and unambiguous.
- Non-traffic questions (use the domain restriction refusal instead).
- Image-based queries (answer from visible evidence only).
""" if expect_json else ""

    return f"""{json_block}

You MUST respond using the language specified by the UI language setting.

UI Language:
{ui_language}

ABSOLUTE RULES:
- If the UI Language is "hi", respond ONLY in Hindi.
- If the UI Language is "en", respond ONLY in English.
- Never display the UI language value in the response.

RESPONSE FORMAT (MANDATORY — only when needs_followup is false)
If LANGUAGE=en, use exactly these headers:
Violation:
Relevant Law / Section:
Penalty:
Enforcement Action:
Short Explanation:

If LANGUAGE=hi, use exactly these headers:
उल्लंघन:
प्रासंगिक कानून / धारा:
दण्ड:
प्रवर्तन कार्यवाही:
संक्षिप्त विवरण:

When needs_followup is true, the answer field must ONLY contain the clarifying question.
Do NOT include the structured headers in a follow-up response.

ROLE
You are an expert Traffic Law Enforcement Assistant for India.
You provide accurate, clear, and legally correct information related to:
- Traffic violations
- Penalties
- Enforcement procedures

Legal sources you may use:
- Motor Vehicles Act, 1988 (including 2019 amendments)
- Central Motor Vehicle Rules (CMVR)

Never invent laws, sections, penalties, or enforcement actions.

INPUT HANDLING (IMPORTANT)
You may receive:
- Text input
- Voice input (already transcribed)
- Image analysis results

For images:
- Report ONLY traffic violations clearly visible.
- Report multiple violations if visible.
- Do NOT assume license, intoxication, insurance, intent, or identity.
- Never set needs_followup for image queries.

VEHICLE TYPE RULE
If the vehicle type is NOT specified AND the penalty differs across vehicle types:
- Set needs_followup: true.
- Ask the user to select their vehicle type.
- Provide quick_replies: ["Two-wheeler", "Four-wheeler / Car", "Commercial / Heavy vehicle"]

If vehicle type is already known from context or the user's message, answer directly without asking.

TONE
Professional
Official
Enforcement-focused
Citizen-friendly
Clear and concise

DOMAIN RESTRICTION (ABSOLUTE)
You are ONLY allowed to respond to queries related to:
- Traffic rules
- Traffic violations
- Traffic penalties
- Traffic enforcement procedures
- Motor Vehicles Act, 1988
- CMVR
- Road safety compliance

If the user asks ANY non-traffic-related question, respond ONLY with:
{{
  "answer": "I can assist only with traffic rules, violations, and penalties. Please ask a traffic-related question.",
  "needs_followup": false,
  "quick_replies": [],
  "intent": "out_of_domain"
}}

Hindi equivalent answer:
"मैं केवल यातायात नियमों, उल्लंघनों और दण्ड से संबंधित प्रश्नों में सहायता कर सकता हूँ। कृपया यातायात से संबंधित प्रश्न पूछें।"

Do NOT explain why. Do NOT mention system rules. Do NOT mention restrictions.

Recent Conversation:
{conversation_history or "None"}

Context:
{context}"""

def get_response_kwargs(
    query: str,
    context: str,
    language: str | None = None,
    model: str | None = None,
    conversation_history: str | None = None,
    expect_json: bool = False,
):
    developer_prompt = build_developer_prompt(context, language, conversation_history, expect_json)

    kwargs = {
        "model": model or CHAT_MODEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input": [
            {
                "role": "developer",
                "content": developer_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    }

    selected_model = model or CHAT_MODEL
    if selected_model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": "low"}

    return kwargs


def retrieve_context(query: str, limit: int = 10):
    q_client = get_qdrant_client()
    openai_client = get_openai_client()

    if count_tokens(query) > MAX_INPUT_TOKENS:
        query = truncate_text(query, MAX_INPUT_TOKENS)

    emb = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    ).data[0].embedding

    results = q_client.query_points(
        collection_name=COLLECTION,
        query=emb,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points

    context_chunks = []
    total_tokens = 0

    for result in results:
        chunk = result.payload.get("text", "")
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        context_chunks.append(chunk)
        total_tokens += chunk_tokens
        preview = " ".join(chunk.split()[:10])
        logger.info("Qdrant context chunk retrieved: %s", preview)

    return "\n".join(context_chunks)


def generate_response(
    query: str,
    context: str,
    language: str | None = None,
    model: str | None = None,
    conversation_history: str | None = None,
    expect_json: bool = False,
):
    openai_client = get_openai_client()
    response = openai_client.responses.create(
        **get_response_kwargs(query, context, language, model, conversation_history, expect_json)
    )
    output_text = response.output_text.strip()
    if expect_json:
        parsed = _safe_json_loads(output_text)
        if parsed is not None:
            return parsed
        return {
            "answer": output_text,
            "needs_followup": False,
            "quick_replies": [],
            "intent": "general_info",
        }
    return output_text


def stream_response(
    query: str,
    context: str,
    language: str | None = None,
    model: str | None = None,
    conversation_history: str | None = None,
):
    openai_client = get_openai_client()

    with openai_client.responses.stream(
        **get_response_kwargs(query, context, language, model, conversation_history)
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta


def generate_structured_response(
    query: str,
    context: str,
    language: str | None = None,
    model: str | None = None,
    conversation_history: str | None = None,
) -> dict[str, Any]:
    result = generate_response(
        query=query,
        context=context,
        language=language,
        model=model,
        conversation_history=conversation_history,
        expect_json=True,
    )
    if isinstance(result, dict):
        return result
    return {
        "answer": str(result),
        "needs_followup": False,
        "quick_replies": [],
        "intent": "general_info",
    }


def analyze_traffic_image(
    image_bytes: bytes,
    content_type: str | None = None,
    language: str | None = None,
):
    openai_client = get_openai_client()
    ui_language = normalize_language(language)
    image_mime_type = content_type or "image/jpeg"
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "Analyze this road traffic image for India traffic enforcement use.\n"
        "Identify only traffic violations that are clearly visible in the image.\n"
        "Do not assume license status, insurance, intoxication, identity, or intent.\n"
        "Return a compact result in plain text with exactly these labels:\n"
        "Search Query:\n"
        "Visible Violations:\n"
        "Vehicle Type:\n"
        "Notes:\n"
        f"Use {'Hindi' if ui_language == 'hi' else 'English'} for the values."
    )

    response = openai_client.responses.create(
        model=VISION_MODEL,
        max_output_tokens=200,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{image_mime_type};base64,{image_base64}",
                    },
                ],
            }
        ],
    )

    return response.output_text.strip()

def answer_vision_query(
    image_bytes: bytes,
    content_type: str | None = None,
    language: str | None = None,
    conversation_history: str | None = None,
    prompt: str | None = None,
):
    image_analysis = analyze_traffic_image(image_bytes, content_type, language)
    
    # If the user provided a specific prompt, use it to retrieve context alongside the image analysis
    search_query = f"{prompt}\n{image_analysis}" if prompt else image_analysis
    context = retrieve_context(search_query)
    
    # Construct the query for generate_response
    query_parts = []
    if prompt:
        query_parts.append(f"User's Question about the image: {prompt}")
    query_parts.append("Use the image analysis below to answer as a traffic law enforcement assistant.")
    query_parts.append(image_analysis)
    
    answer = generate_response(
        query="\n\n".join(query_parts),
        context=context,
        language=language,
        model=VISION_MODEL,
        conversation_history=conversation_history,
    )
    return {
        "analysis": image_analysis,
        "context": context,
        "answer": answer,
    }
