import os
from pathlib import Path

try:
    from .clients import get_openai_client
except ImportError:
    from clients import get_openai_client

TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")


def transcribe_audio_file(upload_file, language: str | None = None) -> str:
    openai_client = get_openai_client()

    filename = upload_file.filename or "audio.wav"
    suffix = Path(filename).suffix or ".wav"
    content_type = getattr(upload_file, "content_type", None) or "application/octet-stream"
    upload_file.file.seek(0)

    kwargs = {
        "file": (f"upload{suffix}", upload_file.file, content_type),
        "model": TRANSCRIBE_MODEL,
    }

    if language:
        kwargs["language"] = language

    transcript = openai_client.audio.transcriptions.create(**kwargs)
    return transcript.text
