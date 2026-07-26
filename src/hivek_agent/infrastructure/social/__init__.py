"""Social provider connector contracts and implementations."""

from hivek_agent.infrastructure.social.base import (
    ActivationResult,
    FetchInboundResult,
    ProviderAPIError,
    ProviderCapabilityError,
    ProviderSendResult,
    SocialConnector,
)
from hivek_agent.infrastructure.social.factory import SocialConnectorFactory
from hivek_agent.infrastructure.social.mock import MockSocialConnector

__all__ = [
    "ActivationResult",
    "FetchInboundResult",
    "MockSocialConnector",
    "ProviderAPIError",
    "ProviderCapabilityError",
    "ProviderSendResult",
    "SocialConnector",
    "SocialConnectorFactory",
]
