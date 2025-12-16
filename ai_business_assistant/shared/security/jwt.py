from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt


@dataclass(frozen=True)
class TokenData:
    user_id: str
    email: str
    roles: list[str]


def create_access_token(
    *,
    user_id: str,
    email: str,
    roles: list[str],
    secret_key: str,
    algorithm: str,
    expires_in_seconds: int,
) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "roles": roles,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in_seconds)).timestamp()),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_access_token(*, token: str, secret_key: str, algorithm: str) -> TokenData:
    payload = jwt.decode(token, secret_key, algorithms=[algorithm])
    user_id = str(payload.get("sub"))
    email = str(payload.get("email"))
    roles = payload.get("roles") or []
    if not isinstance(roles, list):
        roles = []
    return TokenData(user_id=user_id, email=email, roles=[str(r) for r in roles])
