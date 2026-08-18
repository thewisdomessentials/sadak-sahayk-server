# CC-VECTOR-PIPELINE Technical Document

## 1. Purpose

CC-VECTOR-PIPELINE is a FastAPI-based retrieval augmented generation (RAG) service for Indian traffic law enforcement assistance. It answers traffic-rule, violation, penalty, and enforcement-procedure questions using vector search over indexed legal documents and OpenAI model responses.

The service supports:

- Text chat queries.
- Server-sent event streaming chat responses.
- Audio transcription for voice input.
- Image-based traffic violation analysis followed by legal answer generation.
- English and Hindi response modes.

The domain is intentionally restricted to traffic law and road-safety compliance, mainly around the Motor Vehicles Act, 1988 and Central Motor Vehicle Rules (CMVR).

## 2. Repository Overview

```text
.
├── main.py              # FastAPI application and route handlers
├── rag.py               # RAG retrieval, prompt construction, chat, streaming, and vision logic
├── audio.py             # Audio transcription helper
├── clients.py           # Cached Qdrant and OpenAI client factories
├── config.py            # Environment and Azure Key Vault secret resolution
├── models.py            # Pydantic request models
├── util.py              # Token counting and truncation helpers
├── requirements.txt     # Python dependencies
├── DockerFile           # Container image definition
├── data/                # Source legal PDFs in English and Hindi
└── docs/
    └── technical-document.md
```

The current repository contains the runtime API and source PDFs, but it does not contain a visible ingestion job that parses PDFs, chunks content, creates embeddings, and upserts records into Qdrant. The Qdrant collection is therefore assumed to be populated by an external process.

## 3. High-Level Architecture

```text
Client
  |
  | Bearer token
  v
FastAPI app (main.py)
  |
  +--> /chat
  |      |
  |      +--> retrieve_context()
  |      |      +--> OpenAI embeddings
  |      |      +--> Qdrant vector search
  |      |
  |      +--> generate_response() or stream_response()
  |             +--> OpenAI Responses API
  |
  +--> /transcribe
  |      |
  |      +--> OpenAI audio transcription
  |
  +--> /vision
         |
         +--> OpenAI vision analysis
         +--> retrieve_context()
         +--> OpenAI legal response generation
```

External services:

- OpenAI API for embeddings, chat response generation, image analysis, and transcription.
- Qdrant for vector similarity search.
- Azure Key Vault for secrets when local environment variables are not set.

## 4. Runtime Components

### 4.1 FastAPI Application

File: `main.py`

The application exposes four routes:

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| `GET` | `/health` | Basic health check | No |
| `POST` | `/chat` | RAG-backed text answer | Bearer token required |
| `POST` | `/transcribe` | Audio-to-text transcription | Bearer token required |
| `POST` | `/vision` | Image violation analysis and legal answer | Bearer token required |

Authentication is intentionally simple. `verify_token()` only checks that the `Authorization` header exists and starts with `Bearer `. It returns the token string, but the token is not validated against a signing key, identity provider, database, or allowlist.

### 4.2 Data Model

File: `models.py`

`ChatRequest` contains:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `message` | `str` | Required | User query text |
| `language` | `str | None` | `None` | UI language hint. Supported values are normalized to `en` or `hi` |
| `stream` | `bool` | `False` | Enables server-sent event streaming for `/chat` |

### 4.3 RAG Layer

File: `rag.py`

Core constants:

| Setting | Environment Variable | Default |
| --- | --- | --- |
| Qdrant collection | `QDRANT_COLLECTION` | `sadaksahayak-documents` |
| Embedding model | `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` |
| Chat model | `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` |
| Vision model | `OPENAI_VISION_MODEL` | `gpt-4o-mini` |
| Max input tokens | Code constant | `500` |
| Max context tokens | Code constant | `1500` |
| Max output tokens | Code constant | `600` |

Main functions:

- `normalize_language(language)`: Accepts only `en` and `hi`; defaults unsupported values to `en`.
- `build_developer_prompt(context, language)`: Builds the legal assistant instruction set and injects retrieved context.
- `get_response_kwargs(query, context, language, model)`: Creates OpenAI Responses API request parameters.
- `retrieve_context(query, limit=10)`: Embeds the query, searches Qdrant, and packs result payload text until the context token limit is reached.
- `generate_response(query, context, language, model)`: Produces a complete response using the configured chat model.
- `stream_response(query, context, language, model)`: Streams text deltas from the OpenAI Responses API.
- `analyze_traffic_image(image_bytes, content_type, language)`: Uses the vision model to identify clearly visible violations.
- `answer_vision_query(image_bytes, content_type, language)`: Chains image analysis, context retrieval, and final answer generation.

### 4.4 Client Factories

File: `clients.py`

The service creates cached clients with `functools.lru_cache()`:

- `get_qdrant_client()` creates a `QdrantClient` using `qdrant-url` and `qdrant-api-key`.
- `get_openai_client()` creates an `OpenAI` client using `openai-api-key`.

Caching avoids rebuilding clients on every request.

### 4.5 Secret Resolution

File: `config.py`

Secret lookup follows this order:

1. Load `.env` using `python-dotenv`.
2. Check mapped environment variables:
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `OPENAI_API_KEY`
3. If no environment value exists, fetch from Azure Key Vault using `AZURE_KEY_VAULT_URL` and `DefaultAzureCredential`.

Azure Key Vault secret names used by the application:

- `qdrant-url`
- `qdrant-api-key`
- `openai-api-key`

### 4.6 Audio Transcription

File: `audio.py`

`transcribe_audio_file()` passes the uploaded file stream to OpenAI audio transcription. The model is configured by `OPENAI_TRANSCRIBE_MODEL` and defaults to `whisper-1`.

The optional `language` form value is forwarded to the transcription API when provided.

### 4.7 Token Utilities

File: `util.py`

Token handling uses `tiktoken` with the `cl100k_base` encoding:

- `count_tokens(text)` returns token count.
- `truncate_text(text, max_tokens)` truncates text by token count.

These helpers are used to bound user input and retrieved context before generation.

## 5. Request Flows

### 5.1 Text Chat

Endpoint: `POST /chat`

Request body:

```json
{
  "message": "What is the penalty for driving without a helmet?",
  "language": "en",
  "stream": false
}
```

Flow:

1. Validate the `Authorization: Bearer <token>` header shape.
2. Call `retrieve_context(message)`.
3. Truncate the query if it exceeds `MAX_INPUT_TOKENS`.
4. Create an embedding with the configured embedding model.
5. Query Qdrant for up to 10 nearest points.
6. Read `payload["text"]` from returned points.
7. Pack chunks until `MAX_CONTEXT_TOKENS` is reached.
8. Build the domain-specific developer prompt.
9. Generate a response with the configured chat model.
10. Return the final answer and the length of the joined context string.

Response:

```json
{
  "answer": "Violation:\n...",
  "context_used": 1234
}
```

Note: `context_used` is currently the character length of the joined context string, not the number of Qdrant chunks or tokens.

### 5.2 Streaming Chat

Endpoint: `POST /chat`

When `stream` is `true`, the service returns `text/event-stream`.

Example event payloads:

```text
data: {"delta": "Violation:"}

data: {"delta": "..."}

data: {"done": true, "context_used": 1234}
```

If an exception occurs inside the stream generator, the stream emits:

```text
data: {"error": "..."}
```

### 5.3 Audio Transcription

Endpoint: `POST /transcribe`

Input:

- Multipart field `file`: uploaded audio file.
- Multipart field `language`: optional language hint.

Flow:

1. Validate bearer header shape.
2. Seek the uploaded file stream to the beginning.
3. Preserve the uploaded file extension when constructing the file tuple.
4. Call OpenAI audio transcription.
5. Return transcript text.

Response:

```json
{
  "text": "..."
}
```

### 5.4 Vision Query

Endpoint: `POST /vision`

Input:

- Multipart field `file`: image upload.
- Multipart field `language`: optional response language.

Flow:

1. Validate bearer header shape.
2. Reject uploads whose content type does not start with `image/`.
3. Read image bytes.
4. Base64 encode the image as a data URL.
5. Ask the vision model to produce compact traffic-violation analysis.
6. Use that analysis text as the retrieval query.
7. Generate a legal answer using the retrieved context and vision model.
8. Return the answer, image analysis, and context length.

Response:

```json
{
  "answer": "...",
  "analysis": "Search Query:\n...",
  "context_used": 1234
}
```

## 6. Prompting and Response Contract

The developer prompt enforces:

- UI language selection: `en` or `hi`.
- Fixed response sections.
- Traffic-law-only domain restriction.
- No invented laws, sections, penalties, or enforcement actions.
- Image-specific caution against assuming license, intoxication, insurance, identity, or intent.
- Vehicle-type handling when the user does not specify a vehicle type.
- Use of only retrieved context for legal answers.

English response headers:

- `Violation:`
- `Relevant Law / Section:`
- `Penalty:`
- `Enforcement Action:`
- `Short Explanation:`

Hindi response headers:

- `उल्लंघन:`
- `प्रासंगिक कानून / धारा:`
- `दंड:`
- `प्रवर्तन कार्रवाई:`
- `संक्षिप्त विवरण:`

For non-traffic queries, the model is instructed to return a fixed refusal in the selected language.

## 7. Data and Vector Store

The `data/` directory includes English and Hindi PDFs for:

- Motor Vehicles Act.
- CMVR chapters 1 through 8.
- Appendices.

Expected Qdrant payload shape:

```json
{
  "text": "chunk content used as context"
}
```

Expected collection behavior:

- Collection name defaults to `sadaksahayak-documents`.
- Vector dimensions must match the configured OpenAI embedding model.
- Search uses `query_points()` with `limit=10`, `with_payload=True`, and `with_vectors=False`.

Missing from this repository:

- PDF extraction code.
- Chunking strategy.
- Embedding and Qdrant upsert job.
- Collection creation or migration script.
- Metadata schema for source document, chapter, page, language, or legal section.

## 8. Configuration

Required local environment variables or Key Vault secrets:

| Purpose | Environment Variable | Key Vault Secret |
| --- | --- | --- |
| Qdrant URL | `QDRANT_URL` | `qdrant-url` |
| Qdrant API key | `QDRANT_API_KEY` | `qdrant-api-key` |
| OpenAI API key | `OPENAI_API_KEY` | `openai-api-key` |

Required when using Azure Key Vault fallback:

| Variable | Description |
| --- | --- |
| `AZURE_KEY_VAULT_URL` | Key Vault URL used by `SecretClient` |

Optional runtime variables:

| Variable | Default | Description |
| --- | --- | --- |
| `QDRANT_COLLECTION` | `sadaksahayak-documents` | Qdrant collection name |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4.1-mini` | Text chat model |
| `OPENAI_VISION_MODEL` | `gpt-4o-mini` | Vision analysis and vision-answer model |
| `OPENAI_TRANSCRIBE_MODEL` | `whisper-1` | Audio transcription model |

## 9. Local Development

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set required environment variables in the shell or `.env`.

Run the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Example chat request:

```bash
curl -X POST http://localhost:8080/chat \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -d '{"message":"Penalty for riding without helmet","language":"en","stream":false}'
```

## 10. Container Deployment

The `DockerFile` builds from `python:3.11-slim`, installs `requirements.txt`, copies the repository, exposes port `8080`, and starts Uvicorn:

```bash
docker build -f DockerFile -t cc-vector-pipeline .
docker run --env-file .env -p 8080:8080 cc-vector-pipeline
```

The container expects secrets to be available through environment variables or Azure Key Vault identity.

## 11. Error Handling

Current behavior:

- Route handlers wrap most failures as HTTP `500` with the exception text.
- `/vision` returns HTTP `400` for non-image uploads.
- `/chat` streaming sends an SSE error object when generation fails after the stream starts.
- Missing bearer header shape returns HTTP `401`.

Operational implications:

- Upstream OpenAI, Qdrant, Azure credential, and Key Vault errors may be exposed to callers in response details.
- There is no structured application error type or error code system.
- There is no retry policy around external service calls in application code.

## 12. Security Considerations

Current controls:

- API routes require a bearer-token-shaped header.
- Secrets can be sourced from environment variables or Azure Key Vault.
- OpenAI and Qdrant clients are cached rather than recreated per request.

Important gaps:

- Bearer tokens are not verified.
- There is no authorization policy, token expiry validation, or issuer/audience check.
- There is no rate limiting.
- Uploaded audio and image file size limits are not enforced in application code.
- Exception details may leak operational information.
- `.env` exists locally and should never be committed or baked into container images.

## 13. Observability

The current application has no explicit logging, tracing, metrics, request IDs, latency tracking, or external-service timing. Recommended additions:

- Structured logs for route, request ID, latency, status code, and external dependency failures.
- Metrics for Qdrant latency, OpenAI latency, token usage, request count, and error count.
- Health checks that optionally validate Qdrant and OpenAI connectivity separately from basic process health.

## 14. Scalability and Performance

Current performance safeguards:

- Input query truncation at 500 tokens.
- Retrieved context packing limit of 1500 tokens.
- Output cap of 300 tokens.
- Cached OpenAI and Qdrant client instances.
- Streaming support for lower perceived latency.

Potential bottlenecks:

- Each `/chat` request performs one embedding call, one Qdrant search, and one generation call.
- Each `/vision` request performs vision analysis, embedding, Qdrant search, and final generation.
- There is no caching of repeated retrievals or answers.
- Synchronous external SDK calls are used inside async FastAPI routes, which can reduce concurrency under load.

## 15. Testing Status

No automated tests are currently present in the repository.

Recommended test coverage:

- Unit tests for language normalization and token truncation.
- Unit tests for prompt construction and non-traffic refusal contract.
- Mocked tests for `/chat`, `/transcribe`, and `/vision`.
- Mocked tests for Qdrant payload packing and context token limits.
- Integration tests against a test Qdrant collection.
- Contract tests for streaming SSE event format.

## 16. Known Gaps and Recommended Next Steps

1. Add a document ingestion pipeline for the PDFs in `data/`.
2. Store chunk metadata in Qdrant, such as source file, page, language, chapter, and section.
3. Return context provenance in API responses for auditability.
4. Replace placeholder bearer-token checking with real authentication.
5. Add upload size and content validation for audio and image endpoints.
6. Add structured logging and basic metrics.
7. Convert blocking OpenAI and Qdrant calls to a concurrency-safe execution pattern or async clients where available.
8. Add tests for core RAG behavior and route contracts.
9. Pin dependency versions in `requirements.txt` for reproducible builds.
10. Rename `DockerFile` to the conventional `Dockerfile` if deployment tooling expects that casing.
