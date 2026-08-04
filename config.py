import os
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv

load_dotenv()

SECRET_ENV_MAP = {
    "qdrant-url": "QDRANT_URL",
    "qdrant-api-key": "QDRANT_API_KEY",
    "openai-api-key": "OPENAI_API_KEY",
    "azure-tenant-id": "AZURE_TENANT_ID",
    "azure-client-id": "AZURE_CLIENT_ID",
    "azure-ad-audience": "AZURE_AD_AUDIENCE",
    "database-url": "DATABASE_URL",
    "azure-sql-connection-string": "AZURE_SQL_CONNECTION_STRING",
    "azure-storage-connection-string": "AZURE_STORAGE_CONNECTION_STRING",
    "azure-storage-account-url": "AZURE_STORAGE_ACCOUNT_URL",
    "azure-storage-cases-container": "AZURE_STORAGE_CASES_CONTAINER",
}

@lru_cache()
def get_kv_client():
    keyvault_url = os.getenv("AZURE_KEY_VAULT_URL")
    if not keyvault_url:
        raise RuntimeError("AZURE_KEY_VAULT_URL is not set.")

    credential = DefaultAzureCredential()
    return SecretClient(vault_url=keyvault_url, credential=credential)

@lru_cache()
def get_secret(name: str):
    env_name = SECRET_ENV_MAP.get(name)
    if env_name:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value

    client = get_kv_client()
    return client.get_secret(name).value
