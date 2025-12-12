from __future__ import annotations

import base64
import hashlib
import hmac
import os


_PBKDF2_ALG = "sha256"
_PBKDF2_ITERS = 210_000
_SALT_BYTES = 16
_DKLEN = 32


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password must not be empty")

    salt = os.urandom(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(_PBKDF2_ALG, password.encode("utf-8"), salt, _PBKDF2_ITERS, dklen=_DKLEN)
    return "pbkdf2_sha256$%d$%s$%s" % (
        _PBKDF2_ITERS,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(dk).decode("ascii").rstrip("="),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iters_s, salt_b64, dk_b64 = password_hash.split("$", 3)
    except ValueError:
        return False

    if scheme != "pbkdf2_sha256":
        return False

    try:
        iters = int(iters_s)
    except ValueError:
        return False

    def _b64decode(s: str) -> bytes:
        pad = "=" * (-len(s) % 4)
        return base64.urlsafe_b64decode((s + pad).encode("ascii"))

    salt = _b64decode(salt_b64)
    expected = _b64decode(dk_b64)

    dk = hashlib.pbkdf2_hmac(_PBKDF2_ALG, password.encode("utf-8"), salt, iters, dklen=len(expected))
    return hmac.compare_digest(dk, expected)
