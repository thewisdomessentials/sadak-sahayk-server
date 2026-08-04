import logging
from functools import lru_cache

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

try:
    from .config import get_secret
except ImportError:
    from config import get_secret

logger = logging.getLogger(__name__)


class AuthenticatedUser(dict):
    @property
    def user_id(self) -> str:
        return self.get("oid") or self.get("sub")


@lru_cache()
def get_jwks_client() -> PyJWKClient:
    tenant_id = get_secret("azure-tenant-id")

    jwks_url = (
        f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
    )
    return PyJWKClient(jwks_url)


def _expected_issuer() -> str:
    tenant_id = get_secret("azure-tenant-id")
    return f"https://login.microsoftonline.com/{tenant_id}/v2.0"


def _allowed_issuers() -> set[str]:
    tenant_id = get_secret("azure-tenant-id")
    return {
        f"https://login.microsoftonline.com/{tenant_id}/v2.0",
        f"https://sts.windows.net/{tenant_id}/",
    }


def _expected_audience() -> str:
    try:
        return get_secret("azure-ad-audience")
    except Exception:
        return get_secret("azure-client-id")


def verify_bearer_token(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        signing_key = get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=_expected_audience(),
            options={"require": ["exp", "iat", "iss", "aud"], "verify_iss": False},
        )

        if claims.get("iss") not in _allowed_issuers():
            raise jwt.InvalidIssuerError("Invalid issuer")
    except jwt.PyJWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    user = AuthenticatedUser(claims)
    if not user.user_id:
        raise HTTPException(status_code=401, detail="Token does not include a user id")

    return user
