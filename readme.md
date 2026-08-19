# SadakSahayak-Python Backend

FastAPI backend for the Sadak Sahayak app. It handles chat/RAG, vision analysis, case creation, branch chat sessions, location tracking, and FCM device registration.

## Requirements

- Python 3.11+
- Azure SQL Database
- Qdrant
- OpenAI API key
- Azure Key Vault access, if you use secrets from Key Vault
- Docker Desktop, if you want to build and run the container locally

## Environment Variables

Required for local or container runtime:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `OPENAI_API_KEY`
- `AZURE_SQL_CONNECTION_STRING` or `DATABASE_URL`

Required if using Key Vault fallback:

- `AZURE_KEY_VAULT_URL`

Optional:

- `QDRANT_COLLECTION` = `sadaksahayak-documents`
- `OPENAI_EMBEDDING_MODEL` = `text-embedding-3-large`
- `OPENAI_CHAT_MODEL` = `gpt-4.1-mini`
- `OPENAI_VISION_MODEL` = `gpt-4o-mini`
- `OPENAI_TRANSCRIBE_MODEL` = `whisper-1`
- `CHAT_IMAGES_CONTAINER` = `chat-images`
- `CASE_IMAGES_CONTAINER` = `case-images`

## Local Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your environment variables in `.env` or the shell.

4. Run the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. Check health:

```bash
curl http://localhost:8000/health
```

## Containerization

Build the image:

```bash
docker build -f Dockerfile -t ca6e8c56008bacr.azurecr.io/chat:v20 .
```

If your repo still uses the legacy filename, this also works:

```bash
docker build -f Dockerfile -t ca6e8c56008bacr.azurecr.io/chat:v20 .
```

Run locally:

```bash
docker run --env-file .env -p 8080:8000 ca6e8c56008bacr.azurecr.io/chat:v20
```

## Push to Azure Container Registry

Make sure you are logged into the Azure Container Registry first:
```bash
az acr login --name ca6e8c56008bacr
```

Then push the image:
```bash
docker push ca6e8c56008bacr.azurecr.io/sadaksahayak-chat:v1
```

## Azure Deployment Notes

- Make sure the container app or web app points to the new image tag: `chat:v20`
- Confirm the app service has all required environment variables or Key Vault bindings
- Verify Azure SQL connectivity and managed identity permissions if you use Key Vault

## Main Endpoints

- `GET /health`
- `POST /chat`
- `GET /chat-session`
- `GET /branch-sessions`
- `POST /branch-chat`
- `POST /transcribe`
- `POST /vision`
- `GET /cases`
- `POST /create-case`
- `POST /update-case`
- `POST /location`
- `POST /devices/register`

## Notes

- The backend stores chat history, case history, and uploaded images in Azure SQL / Blob Storage.
- The app relies on OpenAI for chat, vision, and transcription.
- FCM registration is handled server-side through `POST /devices/register`.

## Automated Batch Testing

You can easily test the RAG pipeline against bulk data locally using the included python scripts. These are particularly useful for validating enforcement agency and resolution authority mapping.

1. **Text Queries (`test_batch_queries.py`)**: 
   - Modify the `batch_queries` list inside the file.
   - Run `python test_batch_queries.py` to generate RAG responses in `batch_query_results.json`.

2. **Images / Vision (`test_batch_images.py`)**:
   - Run the script once to generate a `test_images/` folder.
   - Drop your `.png`, `.jpg`, or `.jpeg` files inside `test_images/`.
   - Run `python test_batch_images.py` to process the images via the multimodal pipeline. Results are saved to `batch_image_results.json`.

3. **Audio / Voice Transcriptions (`test_batch_audio.py`)**:
   - Run the script once to generate a `test_audio/` folder.
   - Drop your `.mp3`, `.wav`, or `.m4a` files inside `test_audio/`.
   - Run `python test_batch_audio.py` to transcribe the speech with Whisper and process the text via the RAG pipeline. Results are saved to `batch_audio_results.json`.
