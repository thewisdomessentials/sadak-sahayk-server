from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote
from uuid import uuid4

from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings
from fastapi import UploadFile

try:
    from .config import get_secret
except ImportError:
    from config import get_secret


@lru_cache()
def get_blob_service_client() -> BlobServiceClient:
    try:
        connection_string = get_secret("azure-storage-connection-string")
    except Exception:
        connection_string = None

    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    account_url = get_secret("azure-storage-account-url")
    return BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())


def get_chat_images_container_name() -> str:
    try:
        return get_secret("azure-storage-chat-images-container")
    except Exception:
        return "chat-images"


async def upload_chat_message_image(
    session_id: int,
    user_id: str,
    image: UploadFile | bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> "UploadedChatImage":
    container_name = get_chat_images_container_name()
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    # Handle both UploadFile and raw bytes
    if isinstance(image, bytes):
        content = image
        actual_filename = filename or "image"
        actual_content_type = content_type or "image/jpeg"
    else:
        content = await image.read()
        actual_filename = image.filename or "image"
        actual_content_type = image.content_type or "image/jpeg"
    
    suffix = Path(actual_filename).suffix.lower() or ".jpg"
    blob_name = f"chat/{user_id}/{session_id}/{uuid4().hex}{suffix}"
    blob_client = container_client.get_blob_client(blob_name)

    digest = sha256(content).hexdigest()
    blob_client.upload_blob(
        content,
        overwrite=False,
        content_settings=ContentSettings(content_type=actual_content_type),
        metadata={
            "session_id": str(session_id),
            "user_id": user_id,
            "original_filename": quote(actual_filename, safe=""),
            "content_type": actual_content_type,
            "size_bytes": str(len(content)),
            "sha256": digest,
        },
    )

    return UploadedChatImage(
        url=blob_client.url,
        original_filename=actual_filename,
        content_type=actual_content_type,
        size_bytes=len(content),
        sha256=digest,
    )


@dataclass(frozen=True)
class UploadedChatImage:
    url: str
    original_filename: str | None
    content_type: str | None
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class UploadedCaseImage:
    url: str
    original_filename: str | None
    content_type: str | None
    size_bytes: int
    sha256: str


async def upload_case_image(case_id: int, image: UploadFile) -> UploadedCaseImage:
    container_name = get_cases_container_name()
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    suffix = Path(image.filename or "").suffix.lower()
    blob_name = f"cases/{case_id}/{uuid4().hex}{suffix}"
    blob_client = container_client.get_blob_client(blob_name)

    content = await image.read()
    digest = sha256(content).hexdigest()
    blob_client.upload_blob(
        content,
        overwrite=False,
        content_settings=ContentSettings(content_type=image.content_type),
        metadata={
            "case_id": str(case_id),
            "original_filename": quote(image.filename or "", safe=""),
            "content_type": image.content_type or "",
            "size_bytes": str(len(content)),
            "sha256": digest,
        },
    )

    return UploadedCaseImage(
        url=blob_client.url,
        original_filename=image.filename,
        content_type=image.content_type,
        size_bytes=len(content),
        sha256=digest,
    )
