"""Authentication dependencies for the new social API boundary."""

from __future__ import annotations

import asyncio
import hmac
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, Request, status

from hivek_agent.config import Settings


def _bearer(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not value.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return value.strip()


async def require_social_access(request: Request) -> None:
    """Bind every user-facing social call to one configured demo workspace."""
    settings: Settings = request.app.state.settings
    if not settings.agentic_demo_access_token:
        raise HTTPException(status_code=503, detail="social demo access token is not configured")
    if not hmac.compare_digest(_bearer(request), settings.agentic_demo_access_token):
        raise HTTPException(status_code=401, detail="invalid bearer token")
    workspace_id = str(request.path_params.get("workspace_id") or "")
    header_workspace = request.headers.get("X-Workspace-Id", "")
    if (
        not workspace_id
        or workspace_id != settings.demo_workspace_id
        or header_workspace != workspace_id
    ):
        raise HTTPException(status_code=403, detail="workspace is not allowed")


async def require_internal_access(request: Request) -> None:
    settings: Settings = request.app.state.settings
    if not settings.agentic_internal_secret:
        raise HTTPException(status_code=503, detail="internal API secret is not configured")
    if not hmac.compare_digest(_bearer(request), settings.agentic_internal_secret):
        raise HTTPException(status_code=401, detail="invalid internal bearer token")

    request_id = request.headers.get("X-Request-Id", "").strip()
    timestamp = request.headers.get("X-Timestamp", "").strip()
    if not request_id or len(request_id) > 200 or not timestamp:
        raise HTTPException(status_code=401, detail="internal request metadata is missing")
    try:
        sent_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=UTC)
        sent_at = sent_at.astimezone(UTC)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="internal timestamp is invalid") from exc

    now = datetime.now(UTC)
    replay_window = timedelta(minutes=5)
    if sent_at < now - replay_window or sent_at > now + replay_window:
        raise HTTPException(status_code=401, detail="internal request timestamp is stale")

    lock: asyncio.Lock = request.app.state.internal_request_lock
    async with lock:
        seen: dict[str, datetime] = request.app.state.internal_request_ids
        cutoff = now - replay_window
        for key, recorded_at in list(seen.items()):
            if recorded_at < cutoff:
                seen.pop(key, None)
        if request_id in seen:
            raise HTTPException(status_code=409, detail="internal request was already processed")
        seen[request_id] = now


def enforce_internal_workspace(settings: Settings, workspace_id: str) -> None:
    if workspace_id != settings.demo_workspace_id:
        raise HTTPException(status_code=403, detail="workspace is not allowed")
