from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai_business_assistant.shared.config import get_settings
from ai_business_assistant.shared.security.jwt import TokenData, decode_access_token


_bearer = HTTPBearer(auto_error=False)


def get_current_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenData:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    settings = get_settings()
    try:
        return decode_access_token(token=credentials.credentials, secret_key=settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_roles(*required: str) -> Callable[[TokenData], TokenData]:
    def _dep(token: TokenData = Depends(get_current_token)) -> TokenData:
        if required and not any(r in token.roles for r in required):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return token

    return _dep
