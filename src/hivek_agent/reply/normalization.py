"""Vietnamese-friendly normalization shared by policy and TF-IDF retrieval."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s*[%₫$])?", re.UNICODE)


def fold_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(stripped.replace("đ", "d").split())


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def tokens(text: str, *, bigrams: bool = True) -> list[str]:
    words = _WORD.findall(fold_text(text))
    if not bigrams:
        return words
    return words + [f"{left}::{right}" for left, right in zip(words, words[1:], strict=False)]


def numbers(text: str) -> set[str]:
    return {" ".join(match.group(0).casefold().split()) for match in _NUMBER.finditer(text)}
