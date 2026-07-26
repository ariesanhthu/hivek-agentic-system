"""Tool registry.

Two rules from the blueprint are enforced here:
  1. an agent never sees the whole registry - tools are filtered by intent and scope;
  2. anything with an external side effect requires human approval and cannot be
     triggered by the model on its own.
"""

from __future__ import annotations

from hivek_agent.domain import Intent, ToolPolicy

TOOL_REGISTRY: dict[str, ToolPolicy] = {
    "knowledge.search.facts": ToolPolicy(
        tool_name="knowledge.search.facts",
        risk="read",
        required_scopes=["knowledge:read"],
        description="Tìm dữ kiện đã xác minh trong workspace.",
    ),
    "knowledge.search.graph": ToolPolicy(
        tool_name="knowledge.search.graph",
        risk="read",
        required_scopes=["knowledge:read"],
        description="Duyệt quan hệ product/offer/audience.",
    ),
    "knowledge.upsert_candidate": ToolPolicy(
        tool_name="knowledge.upsert_candidate",
        risk="write",
        required_scopes=["knowledge:write"],
        description="Ghi nhận dữ kiện ứng viên kèm nguồn.",
        idempotent=False,
    ),
    "content.read.assets": ToolPolicy(
        tool_name="content.read.assets",
        risk="read",
        required_scopes=["content:read"],
        description="Đọc bài đã duyệt/bị từ chối để làm ví dụ.",
    ),
    "content.save_draft": ToolPolicy(
        tool_name="content.save_draft",
        risk="write",
        required_scopes=["content:write"],
        description="Lưu bản nháp ở trạng thái chờ duyệt.",
    ),
    "feedback.write": ToolPolicy(
        tool_name="feedback.write",
        risk="write",
        required_scopes=["content:write"],
        description="Ghi nhận phản hồi approve/edit/reject.",
    ),
    "analytics.read.performance": ToolPolicy(
        tool_name="analytics.read.performance",
        risk="read",
        required_scopes=["analytics:read"],
        description="Đọc chỉ số hiệu suất đã chuẩn hóa.",
    ),
    # External side effects. Always gated - the model may queue, never send.
    "publishing.queue.post": ToolPolicy(
        tool_name="publishing.queue.post",
        risk="external_side_effect",
        required_scopes=["publish:write"],
        requires_human_approval=True,
        idempotent=True,
        description="Xếp lịch đăng bài sau khi người dùng duyệt.",
    ),
    "connectors.sync.drive": ToolPolicy(
        tool_name="connectors.sync.drive",
        risk="external_side_effect",
        required_scopes=["connector:sync"],
        requires_human_approval=True,
        timeout_seconds=300,
        description="Đồng bộ tệp từ Google Drive.",
    ),
}

# Least privilege per intent: a setup turn cannot reach publishing.
_INTENT_TOOLS: dict[Intent, tuple[str, ...]] = {
    "setup": ("knowledge.upsert_candidate", "knowledge.search.facts"),
    "update_knowledge": (
        "knowledge.search.facts",
        "knowledge.search.graph",
        "knowledge.upsert_candidate",
    ),
    "create_content_plan": (
        "knowledge.search.facts",
        "knowledge.search.graph",
        "content.read.assets",
    ),
    "create_post": (
        "knowledge.search.facts",
        "knowledge.search.graph",
        "content.read.assets",
        "content.save_draft",
    ),
    "analyze_performance": ("analytics.read.performance", "content.read.assets"),
    "smalltalk": (),
}

DEFAULT_SCOPES: tuple[str, ...] = (
    "knowledge:read",
    "knowledge:write",
    "content:read",
    "content:write",
    "analytics:read",
)


def authorized_tools(intent: Intent | None, scopes: list[str]) -> list[str]:
    """Tools visible for this turn: intent-relevant AND within the caller's scopes."""
    if intent is None:
        return []
    granted = set(scopes)
    return [
        name
        for name in _INTENT_TOOLS.get(intent, ())
        if (policy := TOOL_REGISTRY.get(name)) and set(policy.required_scopes) <= granted
    ]


def get_policy(tool_name: str) -> ToolPolicy | None:
    return TOOL_REGISTRY.get(tool_name)


def requires_approval(tool_name: str) -> bool:
    policy = TOOL_REGISTRY.get(tool_name)
    # Unknown tools are treated as dangerous rather than allowed through.
    return True if policy is None else policy.requires_human_approval
