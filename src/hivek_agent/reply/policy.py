"""Intent and risk rules that run before any model call."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hivek_agent.reply.normalization import fold_text, tokens


class PolicyVerdict(BaseModel):
    intent: str
    risk_labels: list[str] = Field(default_factory=list)
    handoff: bool = False
    ignore: bool = False
    low_risk: bool = False


_HANDOFF: dict[str, tuple[str, ...]] = {
    "refund": ("hoan tien", "tra tien", "doi tien"),
    "complaint": ("khieu nai", "lua dao", "te qua", "that vong", "buc xuc"),
    "payment": ("thanh toan", "chuyen khoan", "so tai khoan", "the tin dung"),
    "legal": ("kien", "phap ly", "luat su", "to cao"),
    "guarantee": ("cam ket", "bao dam", "chac chan 100"),
    "personal_data": ("can cuoc", "cccd", "so dien thoai", "dia chi nha"),
    "request_human": ("gap nhan vien", "noi chuyen voi nguoi", "tu van vien"),
}

_REVIEW: dict[str, tuple[str, ...]] = {
    "price": ("gia", "bao nhieu", "hoc phi", "chi phi"),
    "schedule": ("lich", "khi nao", "ngay nao", "gio nao"),
    "availability": ("con cho", "con hang", "con lich", "available"),
}

_LOW_RISK: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("greeting", ("xin chao", "chao", "hello", "hi", "alo")),
    ("ask_location", ("o dau", "dia chi", "vi tri")),
    ("ask_business_hours", ("mo cua", "lam viec may gio", "gio lam viec")),
    ("ask_basic_process", ("quy trinh", "bat dau the nao", "dang ky the nao")),
    ("request_link", ("xin link", "gui link", "duong dan")),
)


def evaluate_policy(text: str) -> PolicyVerdict:
    folded = fold_text(text)
    word_count = len(tokens(text, bigrams=False))
    if not folded or word_count == 0:
        return PolicyVerdict(intent="empty", ignore=True)
    if word_count > 120 or folded.count("http") > 3:
        return PolicyVerdict(intent="spam", risk_labels=["spam"], ignore=True)

    risks = [label for label, phrases in _HANDOFF.items() if any(p in folded for p in phrases)]
    if risks:
        return PolicyVerdict(intent=risks[0], risk_labels=risks, handoff=True)

    review = [label for label, phrases in _REVIEW.items() if any(p in folded for p in phrases)]
    if review:
        return PolicyVerdict(intent=review[0], risk_labels=review)

    for intent, phrases in _LOW_RISK:
        if any(phrase in folded for phrase in phrases):
            return PolicyVerdict(intent=intent, low_risk=True)
    return PolicyVerdict(intent="general_question")
