"""The learning loop: user edits -> deterministic features -> voice rules.

Nothing here calls a language model. Edits are diffed arithmetically and rules earn
their status through repetition, so the voice profile can always be explained to the
user in terms of the edits that produced it.
"""

from hivek_agent.learning.edit_analysis import (
    LearningService,
    compute_feature_delta,
    count_emoji,
    infer_preferences,
    split_sentences,
)

__all__ = [
    "LearningService",
    "compute_feature_delta",
    "count_emoji",
    "infer_preferences",
    "split_sentences",
]
