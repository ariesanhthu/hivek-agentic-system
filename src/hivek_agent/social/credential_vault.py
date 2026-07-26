"""AES-256-GCM token vault compatible with the Next.js implementation."""

from __future__ import annotations

import base64
import binascii
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hivek_agent.domain import SocialCredential


class CredentialVaultError(RuntimeError):
    """Safe credential failure.  Messages never contain token material."""


class CredentialVault:
    def __init__(self, key_base64: str) -> None:
        self._key = _decode_key(key_base64)
        self._cipher = AESGCM(self._key) if self._key is not None else None

    @property
    def configured(self) -> bool:
        return self._cipher is not None

    def encrypt(
        self,
        token: str,
        *,
        workspace_id: str,
        provider: str,
        credential_id: str,
        key_version: int = 1,
    ) -> tuple[str, str, str]:
        cipher = self._require_cipher()
        iv = os.urandom(12)
        encrypted = cipher.encrypt(
            iv,
            token.encode("utf-8"),
            _aad(workspace_id, provider, credential_id, key_version),
        )
        # cryptography appends the 16-byte GCM tag; the shared Mongo schema stores it
        # separately so Node's createCipheriv and Python can read each other's output.
        ciphertext, tag = encrypted[:-16], encrypted[-16:]
        return (_b64(ciphertext), _b64(iv), _b64(tag))

    def decrypt(self, credential: SocialCredential) -> str:
        cipher = self._require_cipher()
        try:
            ciphertext = base64.b64decode(credential.token_ciphertext, validate=True)
            iv = base64.b64decode(credential.token_iv, validate=True)
            tag = base64.b64decode(credential.token_tag, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CredentialVaultError("credential encoding is invalid") from exc
        if len(iv) != 12 or len(tag) != 16:
            raise CredentialVaultError("credential IV/tag length is invalid")
        try:
            token = cipher.decrypt(
                iv,
                ciphertext + tag,
                _aad(
                    credential.workspace_id,
                    credential.provider,
                    credential.credential_id,
                    credential.key_version,
                ),
            )
        except InvalidTag as exc:
            raise CredentialVaultError("credential authentication failed") from exc
        try:
            return token.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CredentialVaultError("credential plaintext is invalid") from exc

    def _require_cipher(self) -> AESGCM:
        if self._cipher is None:
            raise CredentialVaultError("TOKEN_ENCRYPTION_KEY_BASE64 is not configured")
        return self._cipher


def _decode_key(value: str) -> bytes | None:
    if not value.strip():
        return None
    try:
        key = base64.b64decode(value.strip(), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CredentialVaultError("TOKEN_ENCRYPTION_KEY_BASE64 is invalid") from exc
    if len(key) != 32:
        raise CredentialVaultError("TOKEN_ENCRYPTION_KEY_BASE64 must decode to 32 bytes")
    return key


def _aad(workspace_id: str, provider: str, credential_id: str, key_version: int) -> bytes:
    return f"{workspace_id}|{provider}|{credential_id}|{key_version}".encode()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


__all__ = ["CredentialVault", "CredentialVaultError"]
