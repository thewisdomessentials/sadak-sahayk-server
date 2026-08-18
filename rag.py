import os
import base64
import json
import logging
import re
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
MAX_OUTPUT_TOKENS = 600

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
  "reasoning": "STRICT MAXIMUM 1 SENTENCE. Briefly state why you chose the authority.",
  "answer": "string",
  "needs_followup": true,
  "quick_replies": ["string"],
  "intent": "string"
}

Rules for JSON:
- Return valid JSON only. Do not wrap it in markdown fences.
- `quick_replies` can be an empty array.
- `needs_followup` must be true when the query is ambiguous and a clarifying choice would meaningfully change the answer.
- When `needs_followup` is true, `answer` should be a short clarifying question, and `quick_replies` must contain 2–4 clickable options.
- When `needs_followup` is false, `answer` must be the full structured response.
- `intent` should be a short label like `penalty_lookup`, `violation_check`, `seizure_info`, or `general_info`.
- `enforcement_agency` (JSON field): Populate based on mapping principles.
- `resolution_authority` (JSON field): Populate based on mapping principles.

FOLLOW-UP TRIGGER CONDITIONS (set needs_followup: true when ANY of these apply):
- Vehicle type is not mentioned AND the penalty differs across vehicle types (two-wheeler vs four-wheeler vs commercial).
- The offence type is vague and multiple violations could apply (e.g. "police stopped me" without reason).
- The penalty depends on whether it is a first offence or a repeat offence.
- The enforcement action depends on a specific circumstance the user has not stated (e.g. "can police seize my vehicle" without stating why).
- Engine number/Chassis number not provided in query and are needed to decide limits and fine for example Two-Wheeler Noise LimitsT(CPCB) regulates motorcycle and scooter noise based on engine displacement as follows:Up to 80 cc: Max limit of \\(75 \\text{ dB(A)}\\)81 cc to 175 cc
DO NOT set needs_followup: true for:
- Queries that explicitly mention specific vehicles like "motorcycle", "bike", "scooter" (implies two-wheeler), "car", "SUV" (implies four-wheeler), or "truck", "bus" (implies heavy/commercial). You must infer the category from these terms instead of asking a follow-up.
- Queries that are complete and unambiguous.
- Non-traffic questions (use the domain restriction refusal instead).
- Image-based queries (answer from visible evidence only).
""" if expect_json else ""

    json_formatting_rule = "Ensure you return a single valid JSON object. Inside the 'answer' string itself, you MUST append these two lines at the very bottom strictly formatted:\n\nEnforcement Agency: [Category] - [Specific]\nResolution Authority: [Category] - [Specific]"
    text_formatting_rule = "At the very end of your response text, you MUST append these two lines strictly formatted:\n\nEnforcement Agency: [Category] - [Specific]\nResolution Authority: [Category] - [Specific]"
    formatting_rule_text = json_formatting_rule if expect_json else text_formatting_rule

    return f"""{json_block}

You MUST respond using the language specified by the UI language setting.

UI Language:
{ui_language}

AUTHORITY MAPPING PRINCIPLES:
Deduce the Enforcement Agency and Resolution Authority.
Format them exactly as "[Category] - [Specific Authority]".

Categories MUST be exactly one of:
- Enforcement Category: 'Police', 'RTO', 'Other'
- Resolution Category: 'Spot Fine / Portal', 'Court', 'RTO', 'Other'

Specific Authority Normalization (Indian Context):
Do not use foreign or overly technical judicial ranks. Normalize the specific authority text:
- 'Highway Patrol' -> map to 'Traffic Police'
- 'Judicial Magistrate' or 'Magistrate First Class' -> map to 'District Court' or 'Traffic Court'

Rules for mapping:
1. Imprisonment: If the penalty explicitly mandates imprisonment (e.g., Drunk Driving), Resolution is 'Court - District Court'.
2. Administrative: Vehicle documents (Fitness, Permits, Registration) go to Enforcement 'RTO - Regional Transport Office'.
3. Moving / Behavioral: On-road violations (Speeding, Red Light) that only involve fines go to Enforcement 'Police - Traffic Police' and Resolution 'Spot Fine / Portal - Spot Fine'.
4. License Disqualification: (like No Helmet) Enforcement is 'Police - Traffic Police' and Resolution is 'RTO - Regional Transport Office' (since RTO suspends licenses).

FORMATTING RULE:
{formatting_rule_text}

ABSOLUTE RULES:
- If the UI Language is "hi", respond ONLY in Hindi.
- If the UI Language is "en", respond ONLY in English.
- Never display the UI language value in the response.
- PRIMARY PROVISION RULE: Every violation answer must identify the primary statutory provision governing the actual offence. Supporting provisions may be listed separately but must never replace the primary offence provision.

NORMALIZATION RULE (HINDI/HINGLISH):
Understand Hindi, Hinglish, transliterated Hindi and colloquial traffic terminology, but map the facts to the precise legal concept before selecting a section or deciding if a penalty/compounding amount applies. For example, map:
- चालान -> challan / enforcement
- लाइसेंस घर पर है -> licence not produced
- लाइसेंस सस्पेंड -> suspended/disqualified licence
- परमिट नहीं है -> no permit
- परमिट है लेकिन रूट गलत -> permit-condition violation
- फिटनेस है -> fitness certificate
- कागज पूरे हैं -> documents claimed valid — verify
- मोबाइल हाथ में -> handheld mobile-phone use
- शराब की गंध -> suspicion requiring testing
- ओवरलोड -> excess weight requiring weighment
- एल प्लेट नहीं -> learner-condition violation
- नंबर प्लेट गंदी -> registration mark obscured/unreadable
- मालिक गाड़ी चला रहा था? -> driver identity must be established

MANDATORY INTERNAL REASONING (CHAIN OF THOUGHT):
Before generating the final output, you MUST internally follow these 5 steps strictly in order:
1. Violation Extraction: Break the user's scenario down into distinct individual violations (e.g. 1. Handheld mobile phone, 2. Seat-belt violation, 3. Expired insurance).
2. Vehicle Category Identification: Explicitly determine the vehicle category (Two-wheeler / Car / Truck / Bus / Auto / Other). The applicable provision often depends on this.
3. Exact Legal Provision Retrieval: Search the provided context for the exact legal provision corresponding to each violation for the identified vehicle.
4. Mandatory Section Verification: Verify: Does the retrieved Section actually describe the violation? If NO, do not guess or hallucinate. Only use sections explicitly supported by the text.
5. Separate Penalty Retrieval: Find the penalty in the authoritative text separately from the violation. Is the penalty amount explicitly supported by the text for this specific violation and vehicle? If YES, provide it. If NO or ambiguous, DO NOT GUESS. You must state: "Penalty requires verification from the current applicable notification/schedule."

RESPONSE FORMAT (MANDATORY — only when needs_followup is false)
If LANGUAGE=en, use exactly these headers:
Violation:
Relevant Law / Section:
Penalty: (If the penalty amount is not clearly verified from the source text , mention it from which source and state exactly: "Penalty requires verification from the current applicable notification/schedule.")
Enforcement Action: (Describe the exact on-ground protocol, evidence preservation, and documentation steps. Do NOT just write the agency name here.)
Short Explanation: (MAXIMUM 15 words)

If LANGUAGE=hi, use exactly these headers:
उल्लंघन:
प्रासंगिक कानून / धारा:
दण्ड: (If the penalty amount is not clearly verified from the source text, state exactly: "जुर्माने की पुष्टि वर्तमान लागू अधिसूचना/अनुसूची से की जानी चाहिए।")
प्रवर्तन कार्यवाही: (Describe the exact on-ground protocol, evidence preservation, and documentation steps in Hindi. Do NOT just write the agency name here.)
संक्षिप्त विवरण: (MAXIMUM 15 words)

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
- Bharatiya Nyaya Sanhita (BNS) & BNSS (for criminal traffic offences)
- Automotive Industry Standards (AIS) & Bureau of Indian Standards (BIS)
- Ministry of Road Transport and Highways (MoRTH) Notifications & Circulars
- Official Gazette Documents
- State-specific policies (e.g., State EV policies, Taxation Acts, Bus Transport Rules)

CRITICAL RULE: You must base all criminal offence answers exclusively on the Bharatiya Nyaya Sanhita (BNS) and BNSS. Under no circumstances should you cite the IPC (Indian Penal Code) or CrPC. If a user specifically asks about an old IPC section, acknowledge it, but DO NOT use the exact acronym 'IPC' or 'CrPC' in your response. Refer to it strictly as 'the former penal code' or 'the old section', and provide the current governing law under the new BNS framework. Furthermore, LIMIT all criminal-law content strictly to evidence protection, preservation, documentation, and required protocol relevant to traffic enforcement / incident handling. Do NOT provide broad criminal legal advice beyond traffic incident protocol.

PRIORITY RULE: If there is conflicting information (e.g. fine amounts) across different sources, you MUST give maximum priority/weightage to the source 'data/MVA 1988 till 2025 may.pdf' as it is the most up-to-date document.

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
- Traffic rules, violations, penalties, and enforcement procedures
- Motor Vehicles Act, 1988 and CMVR
- Automotive engineering and safety standards (AIS & BIS)
- Road safety compliance and infrastructure guidelines (MoRTH Circulars)
- State-specific transport regulations and EV policies
- Traffic-related criminal offences (BNS/BNSS)
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


def preprocess_search_query(query: str) -> str:
    """Translates Hindi/Hinglish and legacy IPC/CrPC queries into pure English natural language for better RAG retrieval."""
    openai_client = get_openai_client()
    prompt = (
        "You are an expert Indian law query translator. Your job is to prepare the user's query for an English-only semantic search database. "
        "1. If the query is in Hindi or Hinglish, translate it to pure English. "
        "2. If the query mentions an old IPC or CrPC section, replace it with the natural language offence (e.g., 'IPC 279' = 'rash driving'). "
        "Return ONLY the optimized English search string. Do not return the acronyms IPC or CrPC. "
        "If the query is already in English and doesn't mention old laws, return it unchanged."
    )
    
    try:
        response = openai_client.responses.create(
            model=CHAT_MODEL,
            max_output_tokens=50,
            input=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        rewritten = response.output_text.strip()
        print(f"Query Translator: '{query}' -> '{rewritten}'")
        logger.info(f"Query Translator: '{query}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.error(f"Query Translator failed: {e}")
        return query


def retrieve_context(query: str, limit: int = 10):
    q_client = get_qdrant_client()
    openai_client = get_openai_client()
    
    # Pre-process the query (Translate Hinglish to English, IPC to BNS concepts)
    search_query = preprocess_search_query(query)

    if count_tokens(search_query) > MAX_INPUT_TOKENS:
        search_query = truncate_text(search_query, MAX_INPUT_TOKENS)

    emb = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[search_query],
    ).data[0].embedding

    results = q_client.query_points(
        collection_name=COLLECTION,
        query=emb,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points

    # Prioritize 'data/MVA 1988 till 2025 may.pdf' by sorting it to the top of the context
    results.sort(key=lambda x: 0 if x.payload.get("source", "") == "data/MVA 1988 till 2025 may.pdf" else 1)

    context_chunks = []
    total_tokens = 0

    for result in results:
        source = result.payload.get("source", "Unknown")
        chunk_text = result.payload.get("text", "")
        chunk = f"Source: {source}\nText: {chunk_text}"
        
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
    
    # SAFETY NET: Prevent IPC/CrPC Hallucinations
    if re.search(r'\b(IPC|CrPC)\b', output_text, re.IGNORECASE):
        fallback_msg = "Error: The system attempted to cite outdated IPC/CrPC laws. Please verify against current BNS/BNSS guidelines."
        logger.warning("Regex output filter caught IPC hallucination. Overriding response.")
        if expect_json:
            return {
                "reasoning": "Fallback Triggered",
                "answer": fallback_msg + "\n\nEnforcement Agency: Other - N/A\nResolution Authority: Other - N/A",
                "needs_followup": False,
                "quick_replies": [],
                "intent": "safety_override"
            }
        return fallback_msg

    if expect_json:
        parsed = _safe_json_loads(output_text)
        if parsed is not None:
            return parsed
        return {
            "reasoning": "Parse Failed",
            "answer": output_text + "\n\nEnforcement Agency: Other - Unknown\nResolution Authority: Other - Unknown",
            "needs_followup": False,
            "quick_replies": [],
            "intent": "general_info"
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
    safe_query = preprocess_search_query(query)

    accumulated_text = ""
    fallback_triggered = False

    with openai_client.responses.stream(
        **get_response_kwargs(safe_query, context, language, model, conversation_history)
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                accumulated_text += event.delta
                
                # SAFETY NET: Check for IPC/CrPC hallucination mid-stream
                if not fallback_triggered and re.search(r'\b(IPC|CrPC)\b', accumulated_text, re.IGNORECASE):
                    fallback_triggered = True
                    logger.warning("Regex output filter caught IPC hallucination in stream. Aborting stream.")
                    yield "\n\n[Error: The system attempted to cite outdated IPC/CrPC laws. Please verify against current BNS/BNSS guidelines.]"
                    break
                
                if not fallback_triggered:
                    yield event.delta


def generate_structured_response(
    query: str,
    context: str,
    language: str | None = None,
    model: str | None = None,
    conversation_history: str | None = None,
) -> dict[str, Any]:
    
    # Pre-process the query so the LLM doesn't see legacy acronyms and gets a clean English query
    safe_query = preprocess_search_query(query)
    
    result = generate_response(
        query=safe_query,
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
        "enforcement_agency": "Unknown",
        "resolution_authority": "Unknown"
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
