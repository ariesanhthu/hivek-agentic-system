"""Runtime configuration.

Every dependency degrades to a zero-infra default so the harness boots and can be
demoed without MongoDB or an LLM key. Set the corresponding env vars to upgrade.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent.parent
REPO_ROOT = APP_ROOT.parent.parent

# Atlas hands out a connection template containing this literal. Treating it as a real
# URI produces an auth failure at first query instead of at boot, so we detect it here.
MONGO_URI_PLACEHOLDER = "<db_password>"

StoreBackend = Literal["auto", "memory", "mongo"]
LLMProvider = Literal["auto", "mock", "gemini"]
SocialMode = Literal["mock", "sandbox", "live"]
InboundMode = Literal["polling", "webhook", "hybrid"]
AutoReplyMode = Literal["off", "suggestion", "low_risk"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env", REPO_ROOT / ".env"),
        env_file_encoding="utf-8-sig",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8100

    # Browser calls this service directly from the Next.js client, so CORS is required.
    cors_origins: str | list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # --- Persistence -------------------------------------------------------
    store_backend: StoreBackend = "auto"
    mongodb_uri: str = ""
    mongodb_db_name: str = "hivek"
    mongodb_timeout_ms: int = 5000
    brand_memory_vector_index: str = "brand_memory_vector_index"

    # LangGraph writes a checkpoint per node, so this collection grows fast (thousands
    # of rows from a single afternoon of testing) and nothing else prunes it. A resumable
    # run is only useful for so long; expire them rather than growing without bound.
    checkpoint_ttl_seconds: int = 7 * 24 * 3600

    # --- Model gateway -----------------------------------------------------
    # `ai_agent_provider` reuses the name already present in the Next.js client env.
    ai_agent_provider: LLMProvider = "auto"
    gemini_api_key: str = ""
    gemini_model_fast: str = "gemini-2.5-flash"
    gemini_model_strategy: str = "gemini-2.5-pro"
    gemini_model_validator: str = "gemini-2.5-flash"
    gemini_timeout_seconds: int = 60
    gemini_max_retries: int = 2

    # --- Skills ------------------------------------------------------------
    # The SKILL.md files live elsewhere in the monorepo. Deriving the path from the
    # package location works in the repo but breaks in a container or a standalone
    # install, so it is overridable.
    skills_dir: str = ""

    # --- Run policy --------------------------------------------------------
    # Hard ceilings; the blueprint forbids unbounded agent loops.
    max_steps_per_run: int = 24
    max_tool_calls_per_run: int = 32
    default_token_budget: int = 12000

    # --- Social demo -------------------------------------------------------
    social_mode: SocialMode = "mock"
    inbound_mode: InboundMode = "polling"
    auto_reply_mode: AutoReplyMode = "suggestion"
    auto_reply_min_confidence: float = 0.90
    auto_reply_allowed_intents: str | list[str] = Field(
        default_factory=lambda: ["greeting", "ask_location", "ask_basic_process"]
    )

    # The browser-facing social API is deliberately bound to one demo workspace.  The
    # legacy chat endpoints remain unchanged until their client is moved behind a BFF.
    demo_workspace_id: str = "demo_workspace"
    agentic_demo_access_token: str = ""
    agentic_internal_secret: str = ""
    token_encryption_key_base64: str = ""

    meta_graph_api_base_url: str = "https://graph.facebook.com"
    meta_graph_api_version: str = "v25.0"
    meta_app_secret: str = ""
    threads_api_base_url: str = "https://graph.threads.net"
    # Threads documentation currently shows both versioned and unversioned paths.
    # Empty means unversioned; deployments can pin e.g. `v1.0` without code changes.
    threads_api_version: str = ""
    threads_app_secret: str = ""
    provider_http_timeout_seconds: float = 20.0
    provider_max_retries: int = 2

    webhook_verify_token: str = ""
    webhook_signature_required: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("auto_reply_allowed_intents", mode="before")
    @classmethod
    def _split_intents(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("meta_graph_api_version", "threads_api_version")
    @classmethod
    def _normalise_api_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        return value if value.startswith("v") else f"v{value}"

    @field_validator("auto_reply_min_confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("AUTO_REPLY_MIN_CONFIDENCE must be between 0 and 1")
        return value

    @property
    def mongo_uri_is_usable(self) -> bool:
        """False when the URI is absent or still holds the Atlas password template."""
        uri = self.mongodb_uri.strip()
        return bool(uri) and MONGO_URI_PLACEHOLDER not in uri

    @property
    def mongo_uri_is_placeholder(self) -> bool:
        return MONGO_URI_PLACEHOLDER in self.mongodb_uri

    @property
    def resolved_store_backend(self) -> Literal["memory", "mongo"]:
        if self.store_backend == "mongo":
            return "mongo"
        if self.store_backend == "memory":
            return "memory"
        return "mongo" if self.mongo_uri_is_usable else "memory"

    @property
    def resolved_llm_provider(self) -> Literal["mock", "gemini"]:
        if self.ai_agent_provider == "gemini":
            return "gemini"
        if self.ai_agent_provider == "mock":
            return "mock"
        return "gemini" if self.gemini_api_key.strip() else "mock"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
