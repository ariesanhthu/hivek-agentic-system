"""Model routing.

Maps a task to the cheapest model tier that can do it. The blueprint is explicit that
routing, tagging and extraction must not run on the expensive reasoning model.
"""

from __future__ import annotations

from hivek_agent.config import Settings, get_settings
from hivek_agent.domain import ModelRoute

# task -> (tier, max_output_tokens, temperature)
_ROUTES: dict[str, tuple[str, int, float]] = {
    "intent_classification": ("fast", 256, 0.0),
    "fact_extraction": ("fast", 1024, 0.0),
    "gap_question": ("fast", 512, 0.3),
    "content_plan": ("reasoning", 2048, 0.4),
    "content_compose": ("creative", 1800, 0.85),
    "content_validate": ("fast", 768, 0.0),
    "edit_analysis": ("fast", 768, 0.1),
    "conflict_merge": ("reasoning", 1024, 0.2),
    "performance_summary": ("fast", 1024, 0.3),
    "inbound_reply": ("fast", 512, 0.2),
}

_DEFAULT = ("fast", 1024, 0.4)


class ModelRouter:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _model_for_tier(self, tier: str) -> str:
        settings = self._settings
        if tier == "reasoning":
            return settings.gemini_model_strategy
        if tier == "creative":
            return settings.gemini_model_strategy
        if tier == "local":
            return settings.gemini_model_fast
        return settings.gemini_model_fast

    def route(self, task: str) -> ModelRoute:
        tier, max_output_tokens, temperature = _ROUTES.get(task, _DEFAULT)
        primary = self._model_for_tier(tier)
        return ModelRoute(
            task=task,
            model_tier=tier,  # type: ignore[arg-type]
            model_name=primary,
            fallback_chain=self._fallbacks(primary),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            # Creative output is sampled per request, so caching it would defeat variety.
            cacheable=tier != "creative",
        )

    def _fallbacks(self, primary: str) -> list[str]:
        """Cheaper alternatives to try if the primary model is unavailable.

        Free-tier Gemini keys routinely have a hard quota of 0 on Pro models while Flash
        works, so a Pro-tier task must degrade rather than fail. Ordered cheapest-last:
        Flash first, then Flash-Lite, which has the most headroom and is the likeliest
        to still answer once the others are exhausted for the day.
        """
        chain = [
            self._settings.gemini_model_fast,
            "gemini-flash-lite-latest",
            "gemini-2.0-flash",
        ]
        return [model for model in dict.fromkeys(chain) if model and model != primary]
