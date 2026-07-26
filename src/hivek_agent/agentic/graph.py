"""LangGraph wiring.

LangGraph owns branching and checkpointing; the node functions in `nodes.py` own the
work. This module deliberately contains no business logic - it is the diagram from the
blueprint, expressed as edges.

  START -> authenticate -> route -> [setup | plan | post | analyze | smalltalk] -> END

Durability: with Mongo configured the checkpointer is Mongo-backed, so a paused run
survives a worker restart. Independently of that, every draft is persisted as a
ContentAsset with `status=needs_review`, so approval never depends on a live graph -
that is what makes the review cycle work across processes and days.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from hivek_agent.agentic.nodes import HarnessNodes, NodeDeps
from hivek_agent.domain import HarnessState

logger = logging.getLogger(__name__)

GRAPH_NAME = "chat"

Branch = Literal["setup", "plan", "post", "analyze", "smalltalk"]

# HarnessState carries our own Pydantic models, which LangGraph msgpacks into each
# checkpoint. It currently warns when deserializing unregistered types and will BLOCK
# them in a future release, so the modules are declared explicitly rather than left to
# break on upgrade.
_ALLOWED_MSGPACK_MODULES = (
    "hivek_agent.domain.knowledge",
    "hivek_agent.domain.content",
    "hivek_agent.domain.harness",
    "hivek_agent.domain.learning",
)


def _build_serde():
    """Serializer that trusts only this package's domain modules."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


def _route_branch(state: HarnessState) -> Branch:
    intent = state.intent
    if intent == "setup":
        return "setup"
    if intent == "create_content_plan":
        return "plan"
    if intent == "create_post":
        return "post"
    if intent == "analyze_performance":
        return "analyze"
    if intent == "update_knowledge":
        return "setup"
    return "smalltalk"


def _facts_gate(state: HarnessState) -> Literal["blocked", "ready"]:
    """Blocking gaps stop the turn and ask the user; nothing downstream runs."""
    return "blocked" if state.status == "needs_user_input" else "ready"


def _draft_gate(state: HarnessState) -> Literal["failed", "ready"]:
    return "failed" if state.status == "failed" or state.draft is None else "ready"


def build_chat_graph(deps: NodeDeps, *, checkpointer: Any | None = None):
    """Compile the chat graph.

    `HarnessState` is a Pydantic model, which LangGraph accepts as a state schema;
    nodes return the whole state, so updates are full replacements rather than partial
    merges. That keeps state transitions explicit and easy to reason about.
    """
    nodes = HarnessNodes(deps)
    graph = StateGraph(HarnessState)

    graph.add_node("authenticate", nodes.authenticate_and_authorize)
    graph.add_node("route", nodes.route_request)
    graph.add_node("load_knowledge", nodes.load_knowledge)
    graph.add_node("validate_required_facts", nodes.validate_required_facts)
    graph.add_node("compile_context", nodes.compile_context)
    graph.add_node("create_content_plan", nodes.create_content_plan)
    graph.add_node("generate_draft", nodes.generate_draft)
    graph.add_node("validate_draft", nodes.validate_draft)
    graph.add_node("handle_setup", nodes.handle_setup)
    graph.add_node("analyze_performance", nodes.analyze_performance)
    graph.add_node("handle_smalltalk", nodes.handle_smalltalk)

    graph.add_edge(START, "authenticate")
    graph.add_edge("authenticate", "route")

    graph.add_conditional_edges(
        "route",
        _route_branch,
        {
            "setup": "handle_setup",
            "plan": "load_knowledge",
            "post": "load_knowledge",
            "analyze": "analyze_performance",
            "smalltalk": "handle_smalltalk",
        },
    )

    graph.add_edge("load_knowledge", "validate_required_facts")
    graph.add_conditional_edges(
        "validate_required_facts",
        _facts_gate,
        {"blocked": END, "ready": "compile_context"},
    )

    # Both plan and post need compiled context; the intent decides which runs next.
    graph.add_conditional_edges(
        "compile_context",
        lambda state: "plan" if state.intent == "create_content_plan" else "post",
        {"plan": "create_content_plan", "post": "generate_draft"},
    )

    graph.add_conditional_edges(
        "generate_draft",
        _draft_gate,
        {"failed": END, "ready": "validate_draft"},
    )

    graph.add_edge("validate_draft", END)
    graph.add_edge("create_content_plan", END)
    graph.add_edge("handle_setup", END)
    graph.add_edge("analyze_performance", END)
    graph.add_edge("handle_smalltalk", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver(serde=_build_serde()))


async def build_checkpointer(settings: Any) -> Any:
    """Mongo-backed checkpointer when available, in-memory otherwise.

    The optional dependency is imported lazily so the service still starts without
    `langgraph-checkpoint-mongodb`.

    Note: `MongoDBSaver` wraps a synchronous MongoClient. Its async methods are the
    base-class ones, which run the sync driver in a thread pool. That is acceptable for
    checkpoint writes (a few small ops per turn) but it is not a natively async driver.
    """
    if settings.resolved_store_backend != "mongo":
        return InMemorySaver(serde=_build_serde())

    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from pymongo import MongoClient
    except ImportError:
        logger.info(
            "langgraph-checkpoint-mongodb not installed - using in-memory checkpointer. "
            "Paused runs will not survive a restart."
        )
        return InMemorySaver(serde=_build_serde())

    try:
        client: Any = MongoClient(
            settings.mongodb_uri,
            appname="hivek-agentic-checkpoints",
            serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
        )
        client.admin.command("ping")
        saver = MongoDBSaver(
            client,
            db_name=settings.mongodb_db_name,
            # Namespaced like every other collection this service owns.
            checkpoint_collection_name="agentic_checkpoints",
            writes_collection_name="agentic_checkpoint_writes",
            ttl=settings.checkpoint_ttl_seconds,
            serde=_build_serde(),
        )
        logger.info("checkpointer=mongo db=%s", settings.mongodb_db_name)
        return saver
    except Exception as exc:
        logger.warning(
            "mongo checkpointer unavailable (%s: %s) - using in-memory",
            type(exc).__name__,
            exc,
        )
        return InMemorySaver(serde=_build_serde())
