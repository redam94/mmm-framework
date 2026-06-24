import copy
import json
import math
import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import aiosqlite
from fastapi import (
    FastAPI,
    Form,
    Request,
    UploadFile,
    File,
    Header,
    HTTPException,
    Depends,
)
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mmm_framework.agents.graph import create_agent_graph
from mmm_framework.agents.spec_locks import apply_spec_patch, is_spec_patch
from mmm_framework.api import sessions as sessions_store
from mmm_framework.auth import store as auth_store
from mmm_framework.auth.deps import (
    ensure_project_access,
    get_current_principal,
    require_project_access,
)
from mmm_framework.auth.models import AuthContext, Role
from mmm_framework.auth.ratelimit import require_org_rate_limit
from mmm_framework.auth.routes import create_auth_router
from mmm_framework.auth.service import initialize_auth

logger = logging.getLogger("mmm_api")
logging.basicConfig(level=logging.INFO)


def _payload_hash(p: dict) -> str:
    """Stable content hash for artifact dedup across chat turns."""
    import hashlib

    return hashlib.md5(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def safe_json_dumps(obj: dict) -> str:
    """JSON serializer that handles NaN/Inf, numpy scalars, and numpy arrays.

    Emits STRICT JSON (``allow_nan=False``): ``default`` only fires for objects
    json can't natively serialize, so a NATIVE ``float('nan')`` would otherwise
    slip through as the ``NaN`` token and make ``JSONResponse(allow_nan=False)``
    raise on render. We normalize via ``default`` first, then recursively coerce
    any remaining non-finite floats to None before the final strict dump.
    """
    try:
        import numpy as np

        _NP = (np.integer, np.floating, np.bool_, np.ndarray)
    except ImportError:
        _NP = ()

    def _default(o):
        if _NP and isinstance(o, np.integer):
            return int(o)
        if _NP and isinstance(o, np.floating):
            if np.isnan(o) or np.isinf(o):
                return None
            return float(o)
        if _NP and isinstance(o, np.bool_):
            return bool(o)
        if _NP and isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        t = type(o).__name__
        if t in ("Timestamp", "NaTType"):
            return str(o)
        if t in ("Series", "DataFrame"):
            return o.to_dict()
        try:
            return float(o)
        except (TypeError, ValueError):
            pass
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    def _strip_nonfinite(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _strip_nonfinite(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_strip_nonfinite(v) for v in o]
        return o

    normalized = json.loads(json.dumps(obj, default=_default))
    return json.dumps(_strip_nonfinite(normalized), allow_nan=False)


# Ref-list dashboard keys are stripped from per-message SSE events (the full
# dashboard is re-sent on every message; the accumulated ref lists go out once
# per turn in the dedicated dashboard_update event instead).
_REF_LIST_DASHBOARD_KEYS = ("plots", "tables")


def _message_dashboard(combined_dashboard: dict) -> dict:
    return {
        k: v
        for k, v in (combined_dashboard or {}).items()
        if k not in _REF_LIST_DASHBOARD_KEYS
    }


def _fold_dashboard_update(combined: dict, dd: dict, live_spec: dict) -> dict:
    """Fold one raw tool update's ``dashboard_data`` into the per-event
    ``combined`` dict (mutated in place), materializing ``model_spec`` patch
    envelopes against ``live_spec``. Returns the new live spec.

    stream_mode="updates" yields each tool's RAW Command update, so a
    single-setting ``update_model_setting`` arrives with ``dashboard_data``
    carrying a ``{"__spec_patch__": [...]}`` envelope (see spec_locks.py). The
    state reducer materializes it into the checkpoint, but the wire copy must
    be materialized here too — the frontend shallow-merges ``dashboard_data``,
    and an envelope would clobber its concrete spec. Applying each envelope
    against the latest spec also lets concurrent single-field updates in one
    ToolNode step accumulate instead of the last envelope winning.
    """
    dd = dict(dd)
    spec = dd.get("model_spec")
    if is_spec_patch(spec):
        live_spec = apply_spec_patch(live_spec, spec)
        dd["model_spec"] = live_spec
    elif isinstance(spec, dict):
        live_spec = spec
    combined.update(dd)
    return live_spec


# ── Persistent checkpointer ───────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "sessions.db"
memory: AsyncSqliteSaver | None = None
_aiosqlite_conn: aiosqlite.Connection | None = None
_ship_task: "asyncio.Task | None" = None
_sync_task: "asyncio.Task | None" = None


async def _audit_ship_loop(interval: float) -> None:
    """Periodically forward new audit records off-host (when configured)."""
    from mmm_framework.agents import audit_shipper

    while True:
        try:
            await asyncio.sleep(interval)
            await asyncio.to_thread(audit_shipper.flush_audit_to_remote)
        except asyncio.CancelledError:
            break
        except Exception:  # pragma: no cover - never let the loop die
            logger.exception("audit off-host ship failed")


async def _connection_sync_loop(interval: float) -> None:
    """Periodically refresh saved data connections whose schedule is due."""
    import time as _time

    from mmm_framework.api import connection_sync

    while True:
        try:
            await asyncio.sleep(interval)
            await asyncio.to_thread(connection_sync.sync_due_connections, _time.time())
        except asyncio.CancelledError:
            break
        except Exception:  # pragma: no cover - never let the loop die
            logger.exception("scheduled connection sync failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, _aiosqlite_conn
    # Fail-closed profile guard (PR-F.6): refuse to boot a HOSTED server on a
    # non-sandboxed kernel — a process that behaves hosted (drops cwd, rejects
    # guessable threads) while running untrusted code in-process is the §4
    # partial-enablement trap, worse than an honest single-user deployment.
    from mmm_framework.agents.profile import assert_hosted_sandbox
    from mmm_framework.agents.tools import _KERNELS

    assert_hosted_sandbox(_KERNELS.impl)

    # Phase 4d: tamper-evident audit sink for the mmm_audit events.
    try:
        from mmm_framework.agents.audit_sink import install_audit_sink

        install_audit_sink()
    except Exception:
        logger.exception("audit sink install failed")

    _aiosqlite_conn = await aiosqlite.connect(str(DB_PATH))
    # sessions.db is shared with the synchronous sessions_store (see
    # api/sessions.py:_conn). WAL + a busy timeout let the checkpointer and the
    # artifact/session writes coexist instead of racing into "database is locked".
    await _aiosqlite_conn.execute("PRAGMA busy_timeout=30000")
    await _aiosqlite_conn.execute("PRAGMA journal_mode=WAL")
    await _aiosqlite_conn.execute("PRAGMA synchronous=NORMAL")
    await _aiosqlite_conn.commit()
    memory = AsyncSqliteSaver(_aiosqlite_conn)
    await memory.setup()
    sessions_store.init_db()
    # Org/tenant auth: schema + optional bootstrap owner + one-time backfill of
    # pre-existing projects/users into the primary org. Inert (single dev org)
    # until MMM_AUTH_ENABLED=1; never blocks startup.
    try:
        sessions_store.ensure_default_project()
        initialize_auth()
    except Exception:
        logger.exception("auth initialization failed (continuing unauthenticated)")
    # Off-host audit shipper: a background tick that forwards new hash-chained
    # audit records to MMM_AUDIT_SHIP_URL so the local log isn't the only copy.
    # Inert unless that env var is set.
    global _ship_task
    try:
        from mmm_framework.agents import audit_shipper

        if audit_shipper.ship_url():
            interval = float(os.environ.get("MMM_AUDIT_SHIP_INTERVAL", "60"))
            _ship_task = asyncio.create_task(_audit_ship_loop(interval))
            logger.info("audit off-host shipper enabled (every %ss)", interval)
    except Exception:
        logger.exception("audit shipper start failed")
    # Scheduled data-connection sync: refresh saved connections on their interval
    # so warehouse data lands automatically. Inert when the interval is 0.
    global _sync_task
    try:
        from mmm_framework.api import connection_sync

        sync_interval = connection_sync.sync_interval_seconds()
        if sync_interval > 0:
            _sync_task = asyncio.create_task(_connection_sync_loop(sync_interval))
            logger.info("scheduled connection sync enabled (every %ss)", sync_interval)
    except Exception:
        logger.exception("connection sync start failed")
    yield
    if _ship_task is not None:
        _ship_task.cancel()
    if _sync_task is not None:
        _sync_task.cancel()
    try:  # reap any per-session subprocess kernels on graceful shutdown
        from mmm_framework.agents.tools import _KERNELS

        _KERNELS.shutdown_all()
    except Exception:
        pass
    if _aiosqlite_conn:
        await _aiosqlite_conn.close()


app = FastAPI(title="MMM Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth router: /auth/signup, /auth/login, /auth/refresh, /auth/me.
# Additive and inert until MMM_AUTH_ENABLED=1.
app.include_router(create_auth_router())

# Tenant-access guards for /projects/{project_id}/... routes, mounted via each
# route's `dependencies=[...]` (read=viewer, write=analyst, admin=owner-ish).
# Cross-tenant access 404s; dev principal (auth off) bypasses. See Phase 1.3.
_proj_read = Depends(require_project_access(Role.VIEWER))
_proj_write = Depends(require_project_access(Role.ANALYST))
_proj_admin = Depends(require_project_access(Role.ADMIN))

# Default principal for handlers invoked directly (unit tests / internal calls):
# FastAPI overrides it with the resolved principal over HTTP; a direct Python
# call gets this dev principal (is_dev → tenant checks bypass, pre-1.2 behavior).
_DEV_PRINCIPAL = AuthContext(
    user_id="dev-user",
    org_id="dev-org",
    email="dev@localhost",
    org_role=Role.OWNER,
    is_dev=True,
)
PrincipalDep = Annotated[AuthContext, Depends(get_current_principal)]


def _ensure_session_access(
    principal: AuthContext, thread_id: str | None, min_role: Role = Role.VIEWER
) -> None:
    """Tenant guard for routes keyed by a session ``thread_id`` (not a project
    path param).

    Resolves the session's owning ``project_id`` (``sessions_store.get_session``)
    and delegates to ``ensure_project_access`` — so a missing session, or one
    owned by another org, raises **404** (ids can't be probed across tenants).
    The dev principal (auth disabled) is a no-op. A legacy *unattributed* session
    (``project_id`` IS NULL) is allowed in the single-tenant posture but denied
    under hosted mode, where every session must be org-attributed.
    """
    if principal.is_dev:
        return
    if not thread_id:
        raise HTTPException(status_code=404, detail="Not found")
    sess = sessions_store.get_session(thread_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Not found")
    pid = sess.get("project_id")
    if pid is None:
        from mmm_framework.agents.profile import is_hosted

        if is_hosted():
            raise HTTPException(status_code=404, detail="Not found")
        return
    ensure_project_access(principal, pid, min_role)


def require_session_access(min_role: Role = Role.VIEWER, *, deny_missing: bool = False):
    """Dependency factory guarding a route keyed by a session ``thread_id``.

    ``thread_id`` binds from the route's path param (always present) OR, for
    report-style routes, from the query string (may be None). Mount via the
    decorator's ``dependencies=[...]`` so no handler body changes are needed::

        @app.get("/state/{thread_id}", dependencies=[_sess_read])

    ``deny_missing`` 404s a missing thread_id under real auth (used by the report
    routes, whose legacy thread-less CWD artifact is not tenant-scoped).
    """

    async def _dep(
        thread_id: str | None = None,
        principal: AuthContext = Depends(get_current_principal),
    ) -> AuthContext:
        if thread_id:
            _ensure_session_access(principal, thread_id, min_role)
        elif deny_missing and not principal.is_dev:
            raise HTTPException(status_code=404, detail="Not found")
        return principal

    return _dep


# Session/thread tenant guards (mounted via route-level dependencies=[...]).
_sess_read = Depends(require_session_access(Role.VIEWER))
_sess_write = Depends(require_session_access(Role.ANALYST))
_rep_read = Depends(require_session_access(Role.VIEWER, deny_missing=True))

# Per-org rate limits (abuse protection; off until MMM_RATELIMIT_ENABLED=1).
_rl_chat = Depends(require_org_rate_limit("chat"))
_rl_heavy = Depends(require_org_rate_limit("heavy"))


def _org_project_ids(principal: AuthContext) -> set[str] | None:
    """The set of project ids the principal's org owns, or ``None`` for the dev
    principal (meaning 'no scoping' — used to filter cross-session aggregates)."""
    if principal.is_dev:
        return None
    return set(auth_store.list_org_project_ids(principal.org_id))


class _StubLLM:
    """No-op LLM for read-only graph operations (state/history/rewind).

    These endpoints only touch the checkpointer; they never invoke the LLM.
    Using a stub avoids requiring an API key for navigation actions.
    """

    def bind_tools(self, _tools):
        return self

    def invoke(self, *_a, **_kw):
        raise RuntimeError("LLM not configured; this graph is read-only")


def _admin_graph():
    return create_agent_graph(_StubLLM(), checkpointer=memory)


def get_llm(
    model_name: str | None,
    api_key: str | None,
    base_url: str | None = None,
    provider: str | None = None,
):
    """Build the agent's chat model from the server's model configuration.

    The provider, model, and (for Vertex AI) GCP project/region/credentials come
    from the model configuration file -- see ``config/model_config.example.yaml``
    and ``mmm_framework.agents.llm``. The per-request ``X-Model-Name`` /
    ``X-API-Key`` / ``X-Base-Url`` headers are passed through as overrides, but
    when the server is configured for a Vertex provider, Application Default
    Credentials remain authoritative and a client-supplied key/base_url is
    ignored. ``X-Base-Url`` is honored only for a local endpoint provider
    (LM Studio), so a client can't redirect a cloud deployment.
    """
    from mmm_framework.agents.llm import build_llm

    return build_llm(
        provider=provider, model_name=model_name, api_key=api_key, base_url=base_url
    )


# ── Chat ──────────────────────────────────────────────────────────────────────


async def _hard_reset_thread(thread_id: str) -> None:
    """Delete all LangGraph checkpoints for thread_id directly from SQLite.

    Nuclear recovery: preserves session metadata (artifacts, assumptions, files)
    but wipes the LangGraph conversation state so the next request starts fresh.
    """
    if _aiosqlite_conn is None:
        return
    try:
        await _aiosqlite_conn.execute(
            "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
        )
        await _aiosqlite_conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
        )
        await _aiosqlite_conn.commit()
        logger.warning("Hard-reset LangGraph checkpoints for thread %s", thread_id)
    except Exception:
        logger.exception("Hard reset failed for thread %s", thread_id)


async def _repair_orphan_tool_calls(thread_id: str) -> None:
    """Backfill stub ToolMessages for any AI tool_call that never got a result.

    A cancelled stream can leave state ending in AIMessage(tool_calls) with no
    matching ToolMessage. Anthropic's API rejects this on the next turn. We
    append synthetic ToolMessages with a 'cancelled' note so the conversation
    remains valid for subsequent /chat calls.

    If the state itself is unreadable (e.g. corrupted checkpoint), falls back to
    _hard_reset_thread so the next request can proceed on a clean slate.
    """
    try:
        g = _admin_graph()
        cfg = {"configurable": {"thread_id": thread_id}}
        state = await g.aget_state(cfg)
        if not state or not state.values:
            return
        msgs = list(state.values.get("messages", []))
        if not msgs:
            return

        # Collect tool_call_ids from AI messages and the set already answered.
        outstanding: list[str] = []
        answered: set[str] = set()
        for m in msgs:
            if isinstance(m, ToolMessage) and m.tool_call_id:
                answered.add(m.tool_call_id)
            elif isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    tcid = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None)
                    )
                    if tcid:
                        outstanding.append(tcid)

        missing = [tid for tid in outstanding if tid not in answered]
        if not missing:
            return

        stubs = [
            ToolMessage(content="[cancelled by user]", tool_call_id=tid, status="error")
            for tid in missing
        ]
        await g.aupdate_state(cfg, {"messages": stubs})
        logger.info("Repaired %d orphan tool_call(s) for %s", len(missing), thread_id)
    except Exception:
        logger.exception(
            "Failed to repair orphan tool_calls for %s; hard-resetting", thread_id
        )
        await _hard_reset_thread(thread_id)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"
    # Where the user is in the app (set by the floating guide bubble), so the
    # agent can answer "what am I looking at?" — prepended to the message as a
    # bracketed app-context note, visibly distinct from the user's own words.
    page_context: str | None = None


@app.post("/chat", dependencies=[_rl_chat])
async def chat_endpoint(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: str | None = Header(None),
    x_model_name: str | None = Header(None),
    x_base_url: str | None = Header(None),
    x_provider: str | None = Header(None),
    x_expert_model: str | None = Header(None),
    x_expert_provider: str | None = Header(None),
    x_expert_api_key: str | None = Header(None),
    x_expert_base_url: str | None = Header(None),
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    # Hosted (PR-F.6): thread_id is a bearer capability, so it must be a
    # server-minted session — reject the guessable default and any client-invented
    # id (no silent auto-create), so one tenant can't address another's session.
    from mmm_framework.agents.profile import is_hosted

    if is_hosted():
        tid = request.thread_id
        if (
            not tid
            or tid == "default_thread"
            or sessions_store.get_session(tid) is None
        ):
            raise HTTPException(
                status_code=403,
                detail="hosted mode requires a server-minted session (POST /sessions)",
            )

    # Tenant guard: you may start a NEW thread, but not drive an EXISTING session
    # owned by another org. New/unattributed threads stay allowed (so chat-create
    # works); hosted mode already requires a server-minted session above.
    if not principal.is_dev:
        _sess = sessions_store.get_session(request.thread_id)
        if _sess is not None and _sess.get("project_id"):
            ensure_project_access(principal, _sess["project_id"], Role.ANALYST)

    # The session's modeling mode selects the prompt framing + tool set. Unset /
    # legacy sessions read as "mmm" (historical behavior).
    from mmm_framework.agents.modes import normalize_mode

    _mode = normalize_mode(
        (sessions_store.get_session(request.thread_id) or {}).get("modeling_mode")
    )
    # The expert override + modeling mode ride in `configurable` alongside thread_id;
    # LangGraph propagates them to delegate_to_expert via the injected RunnableConfig,
    # where the expert sub-agent's LLM (X-Expert-* headers) and mode-gated tools are built.
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "modeling_mode": _mode,
            "expert_model": x_expert_model,
            "expert_provider": x_expert_provider,
            "expert_api_key": x_expert_api_key,
            "expert_base_url": x_expert_base_url,
        }
    }
    # Mark this session active so the thread-scoped model cache + workspace dir
    # resolve correctly for tools run during this request.
    from mmm_framework.agents.runtime import set_current_thread

    set_current_thread(request.thread_id)
    llm = get_llm(x_model_name, x_api_key, x_base_url, x_provider)
    # The fast "chat" tier orchestrates: it gets the orchestrator toolset (heavy /
    # code-gen tools removed) and must delegate hard work to the expert tier via
    # the delegate_to_expert tool. The expert sub-agent (strong model, full
    # toolset) is built lazily inside that tool, sharing this thread's session.
    from mmm_framework.agents.tools import get_tools_for_mode

    agent_graph = create_agent_graph(
        llm,
        checkpointer=memory,
        tools=get_tools_for_mode(_mode, role="orchestrator"),
        role="orchestrator",
        mode=_mode,
    )

    # The "400 => corrupted history => hard reset" recovery is an Anthropic
    # behaviour (its API 400s when the message list ends on an unanswered
    # tool_call). For other endpoints (LM Studio / OpenAI-compatible) a 400 is
    # usually a real, recurring problem (context overflow, weak tool support),
    # so we surface it instead of silently wiping the thread.
    try:
        from mmm_framework.agents.llm import load_model_config

        _provider = (load_model_config().provider or "").lower()
    except Exception:
        _provider = "anthropic"
    _anthropic_family = _provider in ("anthropic", "vertex_anthropic")

    sessions_store.touch_session(request.thread_id)

    async def event_generator():
        # Heal any orphaned tool_calls from a previous cancelled/errored stream
        # BEFORE we build the new astream. Anthropic returns 400 if the message
        # history ends with an AIMessage(tool_calls) that has no ToolMessage reply.
        await _repair_orphan_tool_calls(request.thread_id)

        user_text = request.message
        if request.page_context:
            # page_context is set only by the floating project guide (the
            # workspace chat never sends it), so its presence marks a guide
            # turn. Steer the guide to GROUND answers in the project knowledge
            # base — the onboarding brief, uploaded data dictionaries, and
            # prior analyses are ingested there — rather than answering project
            # questions from generic MMM knowledge alone.
            guide_directive = (
                "You are this project's guide. Before answering anything about "
                "the client, their goals, KPIs, channels, data definitions, "
                "constraints, or prior work, call `search_knowledge_base` to "
                "ground the answer in the project's documents (the onboarding "
                "brief and any uploaded files live there) and cite the source "
                "document(s). For pure UI/orientation questions ('what is this "
                "page?') the app context below is enough — no search needed. If "
                "the knowledge base has nothing relevant, say so briefly and "
                "answer from general MMM expertise. Keep answers concise. Stay "
                "on the project and the platform: if asked something unrelated "
                "to marketing measurement or this project, decline in one "
                "sentence and point back to what you can help with."
            )
            user_text = (
                f"[Guide instructions — not typed by the user: {guide_directive}]\n\n"
                f"[App context — not typed by the user: {request.page_context}]\n\n"
                f"{request.message}"
            )
        initial_message = HumanMessage(content=user_text)

        # Pre-populate dedup keys from artifacts already saved in this session so
        # we don't re-add model_run / report / code artifacts on every new request
        # (LangGraph state persists dashboard_data across turns, so without this
        # every chat would re-insert the model_run artifact).
        captured_artifact_keys: set[str] = set()
        # Track execute_python tool-call ids seen in THIS stream so we can pair
        # each python tool result with a persisted text_output artifact.
        python_call_ids: set[str] = set()
        for _art in sessions_store.list_artifacts(request.thread_id):
            _k, _p = _art.get("kind", ""), _art.get("payload", {})
            if _k == "model_run":
                captured_artifact_keys.add(f"model_run::{_p.get('run_id', '')}")
            elif _k == "report":
                captured_artifact_keys.add(f"report::{_p.get('path', '')}")
            elif _k == "project_report":
                captured_artifact_keys.add(f"project_report::{_p.get('path', '')}")
            elif _k == "project_slides":
                captured_artifact_keys.add(f"project_slides::{_p.get('path', '')}")
            elif _k == "client_report":
                captured_artifact_keys.add(f"client_report::{_p.get('path', '')}")
            elif _k == "client_slides":
                captured_artifact_keys.add(f"client_slides::{_p.get('path', '')}")
            elif _k == "code_snippet":
                captured_artifact_keys.add(f"code::{_p.get('call_id', '')}")
            elif _k in ("experiment_design", "budget_optimization"):
                captured_artifact_keys.add(f"{_k}::{_payload_hash(_p)}")
            elif _k == "text_output":
                captured_artifact_keys.add(f"text_output::{_p.get('call_id', '')}")

        # Base spec for materializing streamed spec-patch envelopes (see
        # _fold_dashboard_update).
        live_spec: dict = {}
        try:
            _snap = await agent_graph.aget_state(config)
            if _snap and _snap.values:
                live_spec = (
                    (_snap.values.get("dashboard_data") or {}).get("model_spec")
                    or _snap.values.get("model_spec")
                    or {}
                )
        except Exception:
            logger.exception("Could not load base model_spec for patch streaming")

        try:
            stream = agent_graph.astream(
                {"messages": [initial_message]}, config, stream_mode="updates"
            )
            async for event in stream:
                if await raw_request.is_disconnected():
                    logger.info(
                        "Client disconnected; aborting stream for %s", request.thread_id
                    )
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    break

                for node_name, state_update in event.items():
                    updates = (
                        state_update
                        if isinstance(state_update, list)
                        else [state_update]
                    )

                    all_messages = []
                    combined_dashboard: dict = {}
                    for upd in updates:
                        if not isinstance(upd, dict):
                            continue
                        all_messages.extend(upd.get("messages", []))
                        dd = upd.get("dashboard_data")
                        if dd:
                            live_spec = _fold_dashboard_update(
                                combined_dashboard, dd, live_spec
                            )

                    if not all_messages:
                        continue

                    for msg in all_messages:
                        msg_type = msg.type
                        content = msg.content
                        if isinstance(content, list):
                            text_parts = []
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    text_parts.append(block.get("text", ""))
                                elif isinstance(block, str):
                                    text_parts.append(block)
                            content = "\n".join(text_parts)

                        tool_calls = getattr(msg, "tool_calls", []) or []
                        tool_call_id = getattr(msg, "tool_call_id", None)

                        # Persist execute_python code as an artifact (once per call_id)
                        for tc in tool_calls:
                            if tc.get("name") == "execute_python":
                                python_call_ids.add(tc.get("id"))
                                key = f"code::{tc.get('id')}"
                                if key not in captured_artifact_keys:
                                    captured_artifact_keys.add(key)
                                    code = (tc.get("args") or {}).get("code") or ""
                                    if code.strip():
                                        sessions_store.add_artifact(
                                            request.thread_id,
                                            "code_snippet",
                                            {"call_id": tc.get("id"), "code": code},
                                        )

                        # Persist the python TEXT OUTPUT (stdout/tables/errors) so it
                        # survives reload and is reusable (req 5/6). One per call_id.
                        if msg_type == "tool" and tool_call_id in python_call_ids:
                            key = f"text_output::{tool_call_id}"
                            if key not in captured_artifact_keys:
                                captured_artifact_keys.add(key)
                                out_text = (
                                    content
                                    if isinstance(content, str)
                                    else str(content)
                                )
                                import re as _re

                                _m = _re.search(r"Generated (\d+) Plotly", out_text)
                                sessions_store.add_artifact(
                                    request.thread_id,
                                    "text_output",
                                    {
                                        "call_id": tool_call_id,
                                        "stdout": out_text,
                                        "plot_count": int(_m.group(1)) if _m else 0,
                                        "is_error": "Error executing code" in out_text,
                                    },
                                )

                        msg_dashboard = _message_dashboard(combined_dashboard)
                        data = {
                            "node": node_name,
                            "type": msg_type,
                            "content": content,
                            "tool_calls": tool_calls,
                            "tool_call_id": tool_call_id,
                            "dashboard_data": msg_dashboard,
                        }
                        try:
                            yield f"data: {safe_json_dumps(data)}\n\n"
                        except Exception as e:
                            logger.error(f"Failed to serialize message: {e}")
                            fallback = {
                                "node": node_name,
                                "type": msg_type,
                                "content": (
                                    f"[Result could not be serialized: {str(e)[:120]}]"
                                    if msg_type == "tool"
                                    else ""
                                ),
                                "tool_call_id": tool_call_id,
                                "tool_calls": [],
                                "dashboard_data": {},
                            }
                            try:
                                yield f"data: {json.dumps(fallback)}\n\n"
                            except Exception:
                                pass
                        await asyncio.sleep(0.01)

                    if combined_dashboard.get("plots") or combined_dashboard.get(
                        "tables"
                    ):
                        refs_payload = {}
                        if combined_dashboard.get("plots"):
                            refs_payload["plots"] = combined_dashboard["plots"]
                        if combined_dashboard.get("tables"):
                            refs_payload["tables"] = combined_dashboard["tables"]
                        plots_event = {
                            "type": "dashboard_update",
                            "dashboard_data": refs_payload,
                        }
                        try:
                            yield f"data: {safe_json_dumps(plots_event)}\n\n"
                            logger.info(
                                "Sent %d plot(s) / %d table(s) to frontend",
                                len(refs_payload.get("plots") or []),
                                len(refs_payload.get("tables") or []),
                            )
                        except Exception as e:
                            logger.error(f"Failed to serialize plots: {e}")
                        await asyncio.sleep(0.01)

                    # Persist report_path artifact when it appears in state
                    report_path = combined_dashboard.get("report_path")
                    if report_path:
                        key = f"report::{report_path}"
                        if key not in captured_artifact_keys:
                            captured_artifact_keys.add(key)
                            sessions_store.add_artifact(
                                request.thread_id, "report", {"path": str(report_path)}
                            )

                    # Persist project report / client report artifacts
                    for art_key, art_kind in (
                        ("project_report_path", "project_report"),
                        ("project_slides_path", "project_slides"),
                        ("client_report_path", "client_report"),
                        ("client_slides_path", "client_slides"),
                    ):
                        art_path = combined_dashboard.get(art_key)
                        if art_path:
                            key = f"{art_kind}::{art_path}"
                            if key not in captured_artifact_keys:
                                captured_artifact_keys.add(key)
                                sessions_store.add_artifact(
                                    request.thread_id, art_kind, {"path": str(art_path)}
                                )

                    # Persist model_run artifact when fit_mmm_model completes
                    model_run = combined_dashboard.get("model_run")
                    if model_run:
                        run_id = model_run.get("run_id") or ""
                        key = f"model_run::{run_id}"
                        if key not in captured_artifact_keys:
                            captured_artifact_keys.add(key)
                            sessions_store.add_artifact(
                                request.thread_id, "model_run", model_run
                            )

                    # Persist decision-layer payloads (budget optimizer /
                    # experiment design) so the home page can surface the
                    # latest recommendation; deduped by content hash since the
                    # dashboard persists across turns.
                    for _plan_kind in ("experiment_design", "budget_optimization"):
                        _plan = combined_dashboard.get(_plan_kind)
                        if _plan:
                            _pkey = f"{_plan_kind}::{_payload_hash(_plan)}"
                            if _pkey not in captured_artifact_keys:
                                captured_artifact_keys.add(_pkey)
                                sessions_store.add_artifact(
                                    request.thread_id, _plan_kind, _plan
                                )

        except asyncio.CancelledError:
            logger.info("Stream cancelled for %s", request.thread_id)
            await _repair_orphan_tool_calls(request.thread_id)
            raise
        except Exception as e:
            logger.exception("Chat stream error for %s", request.thread_id)
            err_text = str(e)
            err_lower = err_text.lower()
            is_api_400 = any(
                kw in err_lower
                for kw in ("400", "bad request", "invalid_request_error", "badrequest")
            )
            if is_api_400 and _anthropic_family:
                # Anthropic 400 ≈ corrupted history (ends on an unanswered
                # tool_call). Hard-reset is the recovery.
                logger.warning(
                    "Anthropic 400 for %s; hard-resetting thread", request.thread_id
                )
                await _hard_reset_thread(request.thread_id)
                recovery = (
                    "The conversation history was corrupted and has been automatically reset. "
                    "Your session is now fresh — please resend your message."
                )
                yield f"data: {json.dumps({'type': 'error', 'content': recovery})}\n\n"
            elif is_api_400:
                # Local / OpenAI-compatible endpoint (e.g. LM Studio): surface the
                # real error rather than wiping the thread. The two usual causes:
                #   1) the model's context window is too small for the system
                #      prompt + 42 tool schemas + the tool output, or
                #   2) the loaded model doesn't fully support tool calling.
                hint = (
                    "The model endpoint returned a 400 (Bad Request). With LM Studio this "
                    "is usually (1) the model's context window being too small for the "
                    "conversation plus the tool definitions and tool output, or (2) the "
                    "loaded model not fully supporting tool calling. Try a model with a "
                    "larger context window and good tool-calling support, raise the "
                    "context length in LM Studio, or start a fresh chat. "
                )
                yield f"data: {json.dumps({'type': 'error', 'content': hint + 'Endpoint said: ' + err_text[:800]})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': err_text})}\n\n"
        finally:
            # If the stream ended (cancel, disconnect, or error) with an
            # AI(tool_calls) message whose ToolMessage results never arrived,
            # the next /chat call would 400 on Anthropic. Backfill stub
            # ToolMessages so the conversation stays valid.
            await _repair_orphan_tool_calls(request.thread_id)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── State (per-thread) ────────────────────────────────────────────────────────


def _serialize_state(values: dict) -> dict:
    messages = []
    for msg in values.get("messages", []):
        messages.append(
            {
                "type": msg.type,
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", []),
                "tool_call_id": getattr(msg, "tool_call_id", None),
            }
        )
    return {
        "messages": messages,
        "dashboard_data": values.get("dashboard_data", {}),
    }


@app.get("/state/{thread_id}", dependencies=[_sess_read])
async def get_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        agent_graph = _admin_graph()
        state = await agent_graph.aget_state(config)
        if not state or not state.values:
            return JSONResponse(content={"messages": [], "dashboard_data": {}})
        return JSONResponse(
            content=safe_json_dumps_load(_serialize_state(state.values))
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def safe_json_dumps_load(obj):
    """Round-trip through safe_json_dumps so the response only contains JSON-safe primitives."""
    return json.loads(safe_json_dumps(obj))


@app.delete("/state/{thread_id}", dependencies=[_sess_write])
async def clear_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        agent_graph = _admin_graph()
        await agent_graph.aupdate_state(config, {"messages": [], "dashboard_data": {}})
        return JSONResponse(content={"status": "cleared"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── History / Rewind (back & retry) ───────────────────────────────────────────


@app.get("/history/{thread_id}", dependencies=[_sess_read])
async def get_history(thread_id: str):
    """Return the checkpoint timeline for this thread, newest first.

    Each entry includes the checkpoint id, parent id, a short label derived
    from the next node, and the count of messages at that point. The frontend
    uses this to drive Back and Retry.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        agent_graph = _admin_graph()
        timeline = []
        async for snap in agent_graph.aget_state_history(config):
            raw_msgs = snap.values.get("messages", []) if snap.values else []
            # Match the frontend's view: ToolMessages are filtered out client-side.
            visible = [m for m in raw_msgs if getattr(m, "type", None) != "tool"]
            last_human_idx = next(
                (
                    i
                    for i in range(len(visible) - 1, -1, -1)
                    if getattr(visible[i], "type", None) == "human"
                ),
                None,
            )
            timeline.append(
                {
                    "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
                    "parent_checkpoint_id": (snap.parent_config or {})
                    .get("configurable", {})
                    .get("checkpoint_id"),
                    "next": list(snap.next) if snap.next else [],
                    "message_count": len(visible),
                    "last_human_index": last_human_idx,
                    "created_at": snap.created_at,
                }
            )
        return JSONResponse(content=timeline)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class RewindRequest(BaseModel):
    checkpoint_id: str


@app.post("/rewind/{thread_id}", dependencies=[_sess_write])
async def rewind(thread_id: str, body: RewindRequest):
    """Fork the thread back to an earlier checkpoint.

    `update_state(target_config, {})` writes a new checkpoint whose parent is
    `target_config`'s checkpoint_id and inherits its values. The thread's tip
    moves to this fork, so subsequent reads return the rewound state.
    """
    try:
        agent_graph = _admin_graph()
        # Locate the snapshot by id; we need its full config (incl. checkpoint_ns)
        # because the checkpointer rejects partial configs.
        snap = None
        async for s in agent_graph.aget_state_history(
            {"configurable": {"thread_id": thread_id}}
        ):
            if s.config["configurable"].get("checkpoint_id") == body.checkpoint_id:
                snap = s
                break
        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail="checkpoint not found")

        # Fork from the target by writing a new checkpoint whose parent is this
        # one. Passing {"messages": []} is a no-op under the operator.add
        # reducer for the messages channel but forces a fresh checkpoint to be
        # persisted, advancing the thread's tip to the fork.
        await agent_graph.aupdate_state(snap.config, {"messages": []})

        live = await agent_graph.aget_state({"configurable": {"thread_id": thread_id}})
        return JSONResponse(
            content=safe_json_dumps_load(_serialize_state(live.values or {}))
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rewind failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Sessions ──────────────────────────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    name: str | None = None
    project_id: str | None = None
    modeling_mode: str | None = None


class RenameSessionRequest(BaseModel):
    name: str


class SetSessionModeRequest(BaseModel):
    modeling_mode: str


@app.get("/sessions")
async def list_sessions_endpoint(
    project_id: str | None = None,
    skip: int = 0,
    limit: int = 50,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    if not principal.is_dev and project_id is not None:
        ensure_project_access(principal, project_id, Role.VIEWER)
    rows = sessions_store.list_sessions(project_id=project_id)
    if not principal.is_dev:
        allowed = auth_store.list_org_project_ids(principal.org_id)
        rows = [r for r in rows if r.get("project_id") in allowed]
    enriched = []
    for row in rows:
        detail = sessions_store.get_session(row["thread_id"])
        enriched.append(detail if detail else row)
    total = len(enriched)
    page = enriched[skip : skip + limit]
    return JSONResponse(content={"sessions": page, "total": total})


@app.post("/sessions")
async def create_session_endpoint(
    body: CreateSessionRequest, principal: PrincipalDep = _DEV_PRINCIPAL
):
    # Can't plant a session inside another org's project.
    if not principal.is_dev and body.project_id is not None:
        ensure_project_access(principal, body.project_id, Role.ANALYST)
    from mmm_framework.agents.modes import normalize_mode

    return JSONResponse(
        content=sessions_store.create_session(
            body.name,
            project_id=body.project_id,
            modeling_mode=normalize_mode(body.modeling_mode),
        )
    )


@app.get("/sessions/{thread_id}", dependencies=[_sess_read])
async def get_session_endpoint(thread_id: str):
    session = sessions_store.get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    artifacts = sessions_store.list_artifacts(thread_id)
    assumptions = sessions_store.list_assumptions(thread_id)
    workflow = sessions_store.get_workflow_overrides(thread_id)
    return JSONResponse(
        content={
            **session,
            "artifacts": artifacts,
            "assumptions": assumptions,
            "workflow_steps": workflow,
        }
    )


@app.patch("/sessions/{thread_id}", dependencies=[_sess_write])
async def rename_session_endpoint(thread_id: str, body: RenameSessionRequest):
    if not sessions_store.rename_session(thread_id, body.name):
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(content={"status": "ok"})


@app.patch("/sessions/{thread_id}/mode", dependencies=[_sess_write])
async def set_session_mode_endpoint(thread_id: str, body: SetSessionModeRequest):
    """Switch a session's modeling mode (mmm | causal_inference | general_bayes |
    descriptive). The next /chat turn reads it to select the prompt framing + tools."""
    from mmm_framework.agents.modes import is_valid_mode

    if not is_valid_mode(body.modeling_mode):
        raise HTTPException(status_code=400, detail="invalid modeling_mode")
    if not sessions_store.update_session(thread_id, modeling_mode=body.modeling_mode):
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(content={"status": "ok", "modeling_mode": body.modeling_mode})


@app.delete("/sessions/{thread_id}", dependencies=[_sess_write])
async def delete_session_endpoint(thread_id: str):
    sessions_store.delete_session(thread_id)
    return JSONResponse(content={"status": "ok"})


# ── Analysis Plans ────────────────────────────────────────────────────────────


class LockPlanRequest(BaseModel):
    thread_id: str
    name: str = "Analysis Plan"
    dag: dict | None = None
    research_question: dict | None = None
    assumptions: list | None = None
    extra: dict | None = None


@app.get("/analysis-plans")
async def list_analysis_plans_endpoint(
    thread_id: str | None = None,
    limit: int = 20,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    if thread_id is not None:
        _ensure_session_access(principal, thread_id, Role.VIEWER)
    plans = sessions_store.list_analysis_plans(thread_id=thread_id)
    if thread_id is None and not principal.is_dev:
        allowed = _org_project_ids(principal) or set()
        plans = [
            pl
            for pl in plans
            if (_s := sessions_store.get_session(pl.get("thread_id")))
            and _s.get("project_id") in allowed
        ]
    page = plans[:limit]
    return JSONResponse(content={"plans": page, "total": len(plans)})


@app.get("/analysis-plans/{plan_id}")
async def get_analysis_plan_endpoint(
    plan_id: str, principal: PrincipalDep = _DEV_PRINCIPAL
):
    plan = sessions_store.get_analysis_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    _ensure_session_access(principal, plan.get("thread_id"), Role.VIEWER)
    return JSONResponse(content=plan)


@app.post("/analysis-plans")
async def lock_analysis_plan_endpoint(
    body: LockPlanRequest, principal: PrincipalDep = _DEV_PRINCIPAL
):
    _ensure_session_access(principal, body.thread_id, Role.ANALYST)
    payload: dict = {}
    if body.dag is not None:
        payload["dag"] = body.dag
    if body.research_question is not None:
        payload["research_question"] = body.research_question
    if body.assumptions is not None:
        payload["assumptions"] = body.assumptions
    if body.extra:
        payload.update(body.extra)
    plan = sessions_store.lock_analysis_plan(
        thread_id=body.thread_id,
        name=body.name,
        payload=payload,
    )
    return JSONResponse(content=plan, status_code=201)


@app.delete("/analysis-plans/{plan_id}")
async def delete_analysis_plan_endpoint(
    plan_id: str, principal: PrincipalDep = _DEV_PRINCIPAL
):
    plan = sessions_store.get_analysis_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    _ensure_session_access(principal, plan.get("thread_id"), Role.ANALYST)
    if not sessions_store.delete_analysis_plan(plan_id):
        raise HTTPException(status_code=404, detail="plan not found")
    return JSONResponse(content={"status": "ok"})


# ── Models (stubs derived from session model_run artifacts) ───────────────────


@app.get("/models")
async def list_models_endpoint(
    limit: int = 8,
    status: str | None = None,
    project_id: str | None = None,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    """Return model_run artifacts from all sessions as lightweight ModelInfo stubs."""
    import time as _time

    if project_id is not None and not principal.is_dev:
        ensure_project_access(principal, project_id, Role.VIEWER)
    allowed = _org_project_ids(principal)
    all_sessions = sessions_store.list_sessions()
    models = []
    for s in all_sessions:
        if allowed is not None and s.get("project_id") not in allowed:
            continue
        tid = s["thread_id"]
        for art in sessions_store.list_artifacts(tid):
            if art["kind"] != "model_run":
                continue
            p = art.get("payload", {})
            run_id = p.get("run_id") or art["id"]
            ts = art.get("created_at", _time.time())
            models.append(
                {
                    "model_id": art[
                        "id"
                    ],  # artifact id is unique; run_id can repeat across sessions
                    "name": p.get("run_name")
                    or p.get("name")
                    or run_id
                    or f"Model {art['id'][:8]}",
                    "data_id": p.get("data_id") or "",
                    "config_id": p.get("config_id") or "",
                    "project_id": s.get("project_id"),
                    "status": p.get("status") or "completed",
                    "progress": p.get("progress") or 100,
                    "created_at": str(ts),
                    "completed_at": str(ts),
                    "thread_id": tid,
                }
            )
    models.sort(key=lambda m: m["created_at"], reverse=True)
    return JSONResponse(content={"models": models[:limit], "total": len(models)})


def _find_model_artifact(model_id: str) -> tuple[dict | None, str | None]:
    """Find a model_run artifact by its ID. Returns (artifact, thread_id)."""
    for s in sessions_store.list_sessions():
        for art in sessions_store.list_artifacts(s["thread_id"]):
            if art["kind"] == "model_run" and art["id"] == model_id:
                return art, s["thread_id"]
    return None, None


@app.get("/models/{model_id}")
async def get_model_endpoint(model_id: str, principal: PrincipalDep = _DEV_PRINCIPAL):
    import time as _time

    art, tid = _find_model_artifact(model_id)
    if art is None:
        raise HTTPException(status_code=404, detail="model not found")
    _ensure_session_access(principal, tid, Role.VIEWER)
    p = art["payload"]
    run_id = p.get("run_id") or art["id"]
    ts = art.get("created_at", _time.time())
    return JSONResponse(
        content={
            "model_id": art["id"],
            "name": p.get("run_name") or p.get("name") or run_id,
            "data_id": "",
            "config_id": "",
            "status": "completed",
            "progress": 100,
            "created_at": str(ts),
            "completed_at": str(ts),
            "thread_id": tid,
        }
    )


@app.get("/models/{model_id}/dashboard")
async def get_model_dashboard_endpoint(
    model_id: str, principal: PrincipalDep = _DEV_PRINCIPAL
):
    """Return roi_metrics + decomposition + summary from the model's LangGraph thread."""
    art, tid = _find_model_artifact(model_id)
    if art is None:
        raise HTTPException(status_code=404, detail="model not found")
    _ensure_session_access(principal, tid, Role.VIEWER)
    try:
        g = _admin_graph()
        snap = await g.aget_state({"configurable": {"thread_id": tid}})
        dashboard = (
            (snap.values.get("dashboard_data") or {}) if snap and snap.values else {}
        )
    except Exception:
        dashboard = {}
    p = art["payload"]
    return JSONResponse(
        content={
            "model_id": model_id,
            "thread_id": tid,
            "run_id": p.get("run_id"),
            "run_name": p.get("run_name"),
            "kpi": p.get("kpi"),
            "channels": p.get("channels", []),
            "controls": p.get("controls", []),
            "n_obs": p.get("n_obs"),
            "n_channels": p.get("n_channels"),
            "inference": p.get("inference", {}),
            "trend": p.get("trend"),
            "seasonality": p.get("seasonality", {}),
            "summary": p.get("summary") or dashboard.get("summary"),
            "roi_metrics": dashboard.get("roi_metrics") or [],
            "decomposition": dashboard.get("decomposition") or [],
            "report_path": p.get("report_path") or dashboard.get("report_path"),
            "model_path": p.get("model_path"),
        }
    )


# ── Projects ──────────────────────────────────────────────────────────────────


class ProjectCreateRequest(BaseModel):
    name: str
    description: str | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None


@app.get("/projects")
async def list_projects_endpoint(
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    # Dev principal (auth disabled) sees all projects; a real tenant is scoped.
    org_id = None if principal.is_dev else principal.org_id
    projects = sessions_store.list_projects(org_id=org_id)
    return JSONResponse(content={"projects": projects, "total": len(projects)})


@app.post("/projects")
async def create_project_endpoint(
    body: ProjectCreateRequest,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    org_id = None if principal.is_dev else principal.org_id
    if org_id is not None:
        from mmm_framework.auth.plans import PlanLimitError, assert_within_project_limit

        try:
            assert_within_project_limit(org_id)
        except PlanLimitError as exc:
            raise HTTPException(status_code=402, detail=str(exc))
    return JSONResponse(
        content=sessions_store.create_project(
            body.name, body.description, org_id=org_id
        )
    )


@app.get("/projects/{project_id}", dependencies=[_proj_read])
async def get_project_endpoint(project_id: str):
    proj = sessions_store.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(content=proj)


@app.get("/projects/{project_id}/onboarding-status", dependencies=[_proj_read])
async def onboarding_status_endpoint(project_id: str):
    """Self-serve onboarding checklist + next action for a project."""
    from mmm_framework.api.onboarding import project_onboarding_status

    status = project_onboarding_status(project_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(content=status)


@app.get("/projects/{project_id}/data-quality", dependencies=[_proj_read])
async def data_quality_endpoint(project_id: str):
    """Latest pre-fit data-quality summary for a project (from the agent's EDA),
    surfaced inline at the onboarding 'add data' step. Reads the most-recently
    updated session that has an EDA envelope; returns {found: false} otherwise."""
    from mmm_framework.api.onboarding import summarize_eda_issues

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    sessions = sorted(
        sessions_store.list_sessions(project_id=project_id),
        key=lambda s: s.get("updated_at", 0),
        reverse=True,
    )
    g = _admin_graph()
    for s in sessions:
        try:
            snap = await g.aget_state({"configurable": {"thread_id": s["thread_id"]}})
            dd = (
                (snap.values.get("dashboard_data") or {})
                if snap and snap.values
                else {}
            )
            eda = dd.get("eda") or {}
            if eda:
                return JSONResponse(
                    content={
                        "found": True,
                        "thread_id": s["thread_id"],
                        "updated_at": eda.get("updated_at"),
                        **summarize_eda_issues(eda.get("issues") or []),
                    }
                )
        except Exception:
            continue
    return JSONResponse(content={"found": False})


@app.patch("/projects/{project_id}", dependencies=[_proj_write])
async def update_project_endpoint(project_id: str, body: ProjectUpdateRequest):
    if not sessions_store.update_project(project_id, body.name, body.description):
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(content=sessions_store.get_project(project_id))


@app.delete("/projects/{project_id}", dependencies=[_proj_admin])
async def delete_project_endpoint(project_id: str):
    if not sessions_store.delete_project(project_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete (not found or is the Default Project)",
        )
    return JSONResponse(content={"success": True})


# ── Project onboarding (client profile → KB project brief) ───────────────────


class OnboardingRequest(BaseModel):
    client_name: str | None = None
    industry: str | None = None
    website: str | None = None
    markets: str | None = None
    goals: str | None = None
    kpis: str | None = None
    channels: str | None = None
    constraints: str | None = None
    audience: str | None = None
    notes: str | None = None
    members: list[dict] | None = None  # [{user_id, role}]


_BRIEF_FIELDS = [
    ("client_name", "Client"),
    ("industry", "Industry"),
    ("website", "Website"),
    ("markets", "Markets"),
    ("audience", "Target audience"),
    ("goals", "Business goals"),
    ("kpis", "KPI definitions"),
    ("channels", "Channel landscape"),
    ("constraints", "Known constraints"),
    ("notes", "Additional context"),
]


def _project_brief_md(project: dict, meta: dict) -> str:
    lines = [
        f"# Project brief — {project['name']}",
        "",
        "This brief was captured during project onboarding and is the canonical "
        "client/project context for this engagement. The copilot should ground "
        "client-specific answers (goals, KPI semantics, channel landscape, "
        "constraints) in this document.",
        "",
    ]
    if project.get("description"):
        lines += [f"**Project description:** {project['description']}", ""]
    for key, label in _BRIEF_FIELDS:
        if meta.get(key):
            lines += [f"## {label}", "", str(meta[key]), ""]
    return "\n".join(lines)


@app.post("/projects/{project_id}/onboarding", dependencies=[_proj_write])
async def project_onboarding_endpoint(project_id: str, body: OnboardingRequest):
    """Save the client/project profile, assign team members, and render the
    profile into a `project_brief.md` in the project's knowledge base — so
    both the global guide chat and every session chat can retrieve it. Safe
    to re-run: the meta merges and the brief is replaced."""
    from fastapi.concurrency import run_in_threadpool
    from mmm_framework.agents import knowledge_base as kb
    from mmm_framework.agents import workspace as _ws

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    # merge semantics: only provided fields change (None never deletes here)
    meta_fields = {
        k: getattr(body, k) for k, _ in _BRIEF_FIELDS if getattr(body, k) is not None
    }
    meta_fields["onboarded"] = True
    project = sessions_store.set_project_meta(project_id, meta_fields)

    members: list[dict] = []
    if body.members is not None:
        try:
            members = sessions_store.set_project_members(project_id, body.members)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Render + (re)ingest the brief. KB ingestion needs an embedding backend;
    # when none is configured the brief file still lands in the KB dir and the
    # doc row records the error — onboarding itself never fails on it.
    brief_status = "skipped"
    try:
        brief_md = _project_brief_md(project, project.get("meta") or {})
        kb_dir = _ws.project_kb_dir(project_id)
        dest = str(_ws.safe_join(kb_dir, "project_brief.md"))
        for doc in sessions_store.list_kb_documents(project_id):
            if doc.get("name") == "project_brief.md":
                sessions_store.delete_kb_document(doc["id"])
        with open(dest, "w") as f:
            f.write(brief_md)
        doc = sessions_store.add_kb_document(
            project_id=project_id,
            name="project_brief.md",
            path=dest,
            kind="markdown",
            size_bytes=os.path.getsize(dest),
            status="pending",
            meta={"source": "onboarding"},
        )
        doc = await run_in_threadpool(kb.ingest_document, doc["id"])
        brief_status = doc.get("status", "unknown")
    except Exception:
        logger.exception("project brief ingestion failed (onboarding saved)")
        brief_status = "error"

    return JSONResponse(
        content=safe_json_dumps_load(
            {"project": project, "members": members, "brief_status": brief_status}
        )
    )


# ── Project guide (the floating per-page assistant) ──────────────────────────

GUIDE_SESSION_NAME = "✦ Project guide"


@app.post("/projects/{project_id}/guide", dependencies=[_proj_read])
async def project_guide_session_endpoint(project_id: str):
    """The project's global guide thread: one persistent session per project
    that the floating chat bubble talks to (full agent + project KB). Created
    on first use, reused after — it's a normal session, so it also shows up in
    the Workspace if the user wants the full view."""
    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    for s in sessions_store.list_sessions(project_id=project_id):
        if s.get("name") == GUIDE_SESSION_NAME:
            return JSONResponse(content={"thread_id": s["thread_id"], "created": False})
    sess = sessions_store.create_session(GUIDE_SESSION_NAME, project_id=project_id)
    return JSONResponse(content={"thread_id": sess["thread_id"], "created": True})


# ── Users (team roster) ───────────────────────────────────────────────────────


class UserCreateRequest(BaseModel):
    name: str
    email: str | None = None
    role: str = "analyst"


class UserUpdateRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    role: str | None = None


class MembersRequest(BaseModel):
    members: list[dict]  # [{user_id, role?}]


@app.get("/users")
async def list_users_endpoint():
    users = sessions_store.list_users()
    return JSONResponse(content={"users": users, "total": len(users)})


@app.post("/users")
async def create_user_endpoint(body: UserCreateRequest):
    try:
        return JSONResponse(
            content=sessions_store.create_user(body.name, body.email, body.role)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.patch("/users/{user_id}")
async def update_user_endpoint(user_id: str, body: UserUpdateRequest):
    try:
        return JSONResponse(
            content=sessions_store.update_user(
                user_id, name=body.name, email=body.email, role=body.role
            )
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404 if "Unknown user" in str(e) else 400, detail=str(e)
        )


@app.delete("/users/{user_id}")
async def delete_user_endpoint(user_id: str):
    if not sessions_store.delete_user(user_id):
        raise HTTPException(status_code=404, detail="user not found")
    return JSONResponse(content={"status": "ok"})


@app.get("/projects/{project_id}/members", dependencies=[_proj_read])
async def list_members_endpoint(project_id: str):
    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    members = sessions_store.list_project_members(project_id)
    return JSONResponse(content={"members": members, "total": len(members)})


@app.put("/projects/{project_id}/members", dependencies=[_proj_admin])
async def set_members_endpoint(project_id: str, body: MembersRequest):
    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    try:
        members = sessions_store.set_project_members(project_id, body.members)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content={"members": members, "total": len(members)})


# ── Preferences & client branding ────────────────────────────────────────────


class PreferenceUpdateRequest(BaseModel):
    key: str | None = None
    value: Any = None
    preferences: dict[str, Any] | None = None  # bulk form


@app.get("/preferences")
async def get_preferences_endpoint():
    """Global (deployment-wide) user preference defaults."""
    return JSONResponse(
        content={"preferences": sessions_store.list_preferences("global")}
    )


@app.put("/preferences")
async def put_preferences_endpoint(body: PreferenceUpdateRequest):
    """Set global preference(s). 403 in the hosted multi-tenant profile —
    there is no per-user identity, so 'global' would leak across tenants
    (project-scoped branding remains available: project ids are capabilities)."""
    from mmm_framework.agents.profile import is_hosted

    if is_hosted():
        raise HTTPException(
            status_code=403,
            detail="Global preferences are disabled in the hosted profile; "
            "use per-project branding instead.",
        )
    updated: dict[str, Any] = {}
    if body.preferences:
        for k, v in body.preferences.items():
            sessions_store.set_preference("global", k, v)
            updated[k] = v
    if body.key:
        sessions_store.set_preference("global", body.key, body.value)
        updated[body.key] = body.value
    if not updated:
        raise HTTPException(status_code=422, detail="Provide key/value or preferences.")
    return JSONResponse(
        content={"preferences": sessions_store.list_preferences("global")}
    )


class DataConnectionCreate(BaseModel):
    name: str
    kind: str  # gcs | bigquery
    config: dict = {}


def _require_connection(project_id: str, connection_id: str) -> dict:
    conn = sessions_store.get_data_connection(connection_id)
    if conn is None or conn.get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn


@app.get("/projects/{project_id}/data-connections", dependencies=[_proj_read])
async def list_data_connections_endpoint(project_id: str):
    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(
        content={"connections": sessions_store.list_data_connections(project_id)}
    )


@app.post("/projects/{project_id}/data-connections", dependencies=[_proj_write])
async def create_data_connection_endpoint(project_id: str, body: DataConnectionCreate):
    """Save a reusable data-source connection (a non-secret bucket/query/table
    reference; credentials stay ambient via ADC)."""
    from mmm_framework.integrations import IntegrationError, data_source_kinds

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    if body.kind not in data_source_kinds():
        raise HTTPException(
            status_code=422,
            detail=f"Unknown kind {body.kind!r}; known: {data_source_kinds()}",
        )
    if not (body.name or "").strip():
        raise HTTPException(status_code=422, detail="name is required")
    # Build the connector once to catch malformed config early (no network).
    from mmm_framework.integrations import build_data_source
    from mmm_framework.integrations.connections import _split_ref

    try:
        cfg, _ref = _split_ref(body.config or {})
        build_data_source(body.kind, cfg)
    except IntegrationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid connection: {exc}")
    except Exception as exc:  # pydantic validation, etc.
        raise HTTPException(status_code=422, detail=f"Invalid connection config: {exc}")
    conn = sessions_store.create_data_connection(
        project_id, body.name.strip(), body.kind, body.config or {}
    )
    return JSONResponse(content=conn)


class ConnectionScheduleUpdate(BaseModel):
    # minutes between auto-syncs; null disables scheduling (manual still works)
    sync_interval_minutes: float | None = None


@app.patch(
    "/projects/{project_id}/data-connections/{connection_id}",
    dependencies=[_proj_write],
)
async def update_data_connection_schedule_endpoint(
    project_id: str, connection_id: str, body: ConnectionScheduleUpdate
):
    """Set or clear a connection's auto-sync interval."""
    _require_connection(project_id, connection_id)
    if body.sync_interval_minutes is not None and not (
        0 < body.sync_interval_minutes <= 43200  # 1 min .. 30 days
    ):
        raise HTTPException(
            status_code=422,
            detail="sync_interval_minutes must be 1..43200 (30 days), or null to disable",
        )
    conn = sessions_store.set_data_connection_schedule(
        connection_id, body.sync_interval_minutes
    )
    return JSONResponse(content=conn)


@app.delete(
    "/projects/{project_id}/data-connections/{connection_id}",
    dependencies=[_proj_write],
)
async def delete_data_connection_endpoint(project_id: str, connection_id: str):
    conn = _require_connection(project_id, connection_id)
    # Best-effort cleanup of the cached snapshot so deletes don't orphan files.
    snap = conn.get("snapshot_path")
    if snap:
        try:
            from pathlib import Path as _Path

            _Path(snap).unlink(missing_ok=True)
        except Exception:
            pass
    sessions_store.delete_data_connection(connection_id)
    return JSONResponse(content={"deleted": True})


@app.post(
    "/projects/{project_id}/data-connections/{connection_id}/test",
    dependencies=[_proj_read],
)
async def test_data_connection_endpoint(project_id: str, connection_id: str):
    conn = _require_connection(project_id, connection_id)
    from mmm_framework.integrations import probe_connection

    status = await asyncio.to_thread(
        probe_connection, conn["kind"], conn.get("config") or {}
    )
    return JSONResponse(content=status.as_dict())


@app.post(
    "/projects/{project_id}/data-connections/{connection_id}/preview",
    dependencies=[_proj_write],
)
async def preview_data_connection_endpoint(
    project_id: str, connection_id: str, rows: int = 20
):
    """Read the first ``rows`` of a connection (capped) for a UI preview.

    Gated as a write: it executes the connection's (billable) query/read and
    stamps last_synced.
    """
    conn = _require_connection(project_id, connection_id)
    from mmm_framework.integrations import (
        IntegrationError,
        read_connection_dataframe,
        scrub_cloud_error,
    )

    n = max(1, min(int(rows), 200))
    try:
        df = await asyncio.to_thread(
            read_connection_dataframe,
            conn["kind"],
            conn.get("config") or {},
            max_rows=n,
        )
    except IntegrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface read/auth failure as 502
        # Scrub project ids / SA emails / credential paths from the error.
        raise HTTPException(
            status_code=502, detail=scrub_cloud_error(f"{type(exc).__name__}: {exc}")
        )
    head = df.head(n)
    sessions_store.touch_data_connection_synced(connection_id)
    return JSONResponse(
        content=safe_json_dumps_load(
            {
                "columns": [str(c) for c in head.columns],
                "rows": head.to_dict("records"),
                "n_preview": int(len(head)),
            }
        )
    )


@app.get("/projects/{project_id}/branding", dependencies=[_proj_read])
async def get_branding_endpoint(project_id: str):
    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(content=sessions_store.get_project_branding(project_id) or {})


@app.put("/projects/{project_id}/branding", dependencies=[_proj_admin])
async def put_branding_endpoint(project_id: str, body: dict):
    """Validate + save project branding. A manual PUT counts as confirmation
    unless the payload explicitly says otherwise."""
    from mmm_framework.agents.branding import Branding

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    payload = dict(body or {})
    payload.setdefault("source", "manual")
    payload.setdefault("confirmed", True)
    try:
        branding = Branding.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid branding: {exc}")
    saved = branding.model_dump()
    sessions_store.set_project_branding(project_id, saved)
    return JSONResponse(content=saved)


class BrandingExtractRequest(BaseModel):
    url: str
    save: bool = True


@app.post(
    "/projects/{project_id}/branding/extract",
    dependencies=[_proj_write, _rl_heavy],
)
async def extract_branding_endpoint(project_id: str, body: BrandingExtractRequest):
    """Extract a branding proposal from a client website (SSRF-guarded,
    server-side). Saved with confirmed=false — the UI/user must confirm
    before it styles any output."""
    from fastapi.concurrency import run_in_threadpool

    from mmm_framework.agents.brand_extract import (
        BrandExtractError,
        extract_brand_from_url,
    )

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    try:
        proposal = await run_in_threadpool(extract_brand_from_url, body.url)
    except BrandExtractError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Brand extraction failed")
        raise HTTPException(status_code=502, detail="Brand extraction failed.")
    if body.save:
        sessions_store.set_project_branding(project_id, proposal)
    return JSONResponse(content=proposal)


# ── Budget Plans (stub — agent API has no budget plan management) ──────────────


@app.get("/budget-plans")
async def list_budget_plans_endpoint():
    return JSONResponse(content={"plans": [], "total": 0})


# ── Experiments (lift-test registry) ──────────────────────────────────────────


class ExperimentUpsertRequest(BaseModel):
    id: str | None = None
    project_id: str | None = None
    thread_id: str | None = None
    channel: str | None = None
    design_type: str | None = None
    status: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    estimand: str | None = None
    value: float | None = None
    se: float | None = None
    notes: str | None = None
    recommending_run_id: str | None = None
    design: dict | None = None
    readout: dict | None = None
    priority: dict | None = None


class ExperimentTransitionRequest(BaseModel):
    status: str
    note: str | None = None
    # readout fields, used when transitioning to 'completed'
    value: float | None = None
    se: float | None = None
    estimand: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    readout: dict | None = None
    # used when transitioning to 'calibrated'
    calibrated_run_id: str | None = None


@app.get("/experiments")
async def list_experiments_endpoint(
    project_id: str | None = None,
    status: str | None = None,
    channel: str | None = None,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    if not principal.is_dev and project_id is not None:
        ensure_project_access(principal, project_id, Role.VIEWER)
    exps = sessions_store.list_experiments(
        project_id=project_id, status=status, channel=channel
    )
    if not principal.is_dev:
        allowed = auth_store.list_org_project_ids(principal.org_id)
        exps = [e for e in exps if e.get("project_id") in allowed]
    return JSONResponse(content={"experiments": exps, "total": len(exps)})


@app.get("/experiments/{experiment_id}")
async def get_experiment_endpoint(
    experiment_id: str,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    exp = sessions_store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    ensure_project_access(principal, exp.get("project_id"), Role.VIEWER)
    return JSONResponse(content=exp)


@app.post("/experiments")
async def upsert_experiment_endpoint(
    body: ExperimentUpsertRequest,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    if not principal.is_dev and body.project_id:
        ensure_project_access(principal, body.project_id, Role.ANALYST)
    try:
        exp = sessions_store.upsert_experiment(
            experiment_id=body.id,
            project_id=body.project_id,
            thread_id=body.thread_id,
            channel=body.channel,
            design_type=body.design_type,
            status=body.status,
            start_date=body.start_date,
            end_date=body.end_date,
            estimand=body.estimand,
            value=body.value,
            se=body.se,
            notes=body.notes,
            recommending_run_id=body.recommending_run_id,
            design=body.design,
            readout=body.readout,
            priority=body.priority,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=exp)


@app.post("/experiments/{experiment_id}/transition")
async def transition_experiment_endpoint(
    experiment_id: str,
    body: ExperimentTransitionRequest,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    """Validated lifecycle move (draft→planned→running→completed→calibrated,
    abandoned from any active state). 409 on an illegal transition so the UI
    can distinguish state-machine conflicts from bad input."""
    _exp = sessions_store.get_experiment(experiment_id)
    if _exp is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    ensure_project_access(principal, _exp.get("project_id"), Role.ANALYST)
    try:
        exp = sessions_store.transition_experiment(
            experiment_id,
            body.status,
            note=body.note,
            value=body.value,
            se=body.se,
            estimand=body.estimand,
            start_date=body.start_date,
            end_date=body.end_date,
            readout=body.readout,
            calibrated_run_id=body.calibrated_run_id,
        )
    except ValueError as e:
        msg = str(e)
        raise HTTPException(
            status_code=409 if "Illegal transition" in msg else 400, detail=msg
        )
    return JSONResponse(content=exp)


@app.delete("/experiments/{experiment_id}")
async def delete_experiment_endpoint(
    experiment_id: str,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    _exp = sessions_store.get_experiment(experiment_id)
    if _exp is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    ensure_project_access(principal, _exp.get("project_id"), Role.ANALYST)
    if not sessions_store.delete_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="experiment not found")
    return JSONResponse(content={"status": "ok"})


# ── Direct model load (UI button — no LLM round-trip) ────────────────────────


class LoadModelRequest(BaseModel):
    name: str


@app.post("/sessions/{thread_id}/load-model", dependencies=[_sess_write, _rl_heavy])
async def load_model_endpoint(thread_id: str, body: LoadModelRequest):
    """Load a saved fitted model into the session directly. UI buttons call
    this instead of asking the agent to run the load_fitted_model tool."""
    from mmm_framework.agents.tools import load_model_core

    g = _admin_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = await g.aget_state(cfg)
    values = snap.values or {}
    res = load_model_core(
        thread_id,
        body.name,
        values.get("model_spec"),
        values.get("dataset_path"),
    )
    if not res["ok"]:
        raise HTTPException(status_code=400, detail=res["message"])

    dashboard = dict(values.get("dashboard_data") or {})
    dashboard["model_status"] = "completed"
    dashboard["summary"] = res["message"]
    # Attributed to the agent node (terminal write, same as the /spec endpoints);
    # the next agent turn sees model_status=completed in CURRENT STATE.
    await g.aupdate_state(
        cfg,
        {"model_status": "completed", "dashboard_data": dashboard},
        as_node="agent",
    )
    return JSONResponse(content={"status": "ok", "message": res["message"]})


# ── Run lineage (MLflow-style tracking) ───────────────────────────────────────


@app.get("/runs")
async def list_runs_endpoint(
    project_id: str | None = None, principal: PrincipalDep = _DEV_PRINCIPAL
):
    """The model-run lineage timeline: every fit with dataset fingerprint,
    spec diff vs the previous run, and the assumptions added/revised (the
    versioned data + model + rationale record)."""
    from mmm_framework.api.runs import build_run_timeline

    if project_id is not None and not principal.is_dev:
        ensure_project_access(principal, project_id, Role.VIEWER)
    runs = build_run_timeline(project_id)
    allowed = _org_project_ids(principal)
    if allowed is not None:
        runs = [r for r in runs if r.get("project_id") in allowed]
    return JSONResponse(
        content=safe_json_dumps_load({"runs": runs, "total": len(runs)})
    )


# ── Portfolio (home page aggregation) ─────────────────────────────────────────


@app.get("/portfolio-benchmark")
async def portfolio_benchmark_endpoint(
    stale_after_days: int = 90,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    """Cross-brand benchmarking + governance over the org's projects' run metrics
    (the agency/holding-co view: rank a brand's channel ROIs against the
    portfolio, see model freshness + calibration coverage)."""
    import time as _t

    from mmm_framework.api.portfolio_benchmark import build_portfolio_benchmark

    org_id = None if principal.is_dev else principal.org_id
    projects = sessions_store.list_projects(org_id=org_id)
    runs_by_project = {
        p["project_id"]: sessions_store.list_run_metrics(project_id=p["project_id"])
        for p in projects
    }
    calibrated_by_project = {
        p["project_id"]: len(
            sessions_store.list_experiments(
                project_id=p["project_id"], status="calibrated"
            )
        )
        for p in projects
    }
    payload = build_portfolio_benchmark(
        projects,
        runs_by_project,
        now_ts=_t.time(),
        calibrated_by_project=calibrated_by_project,
        stale_after_days=stale_after_days,
    )
    return JSONResponse(content=safe_json_dumps_load(payload))


@app.get("/portfolio")
async def portfolio_endpoint(
    project_id: str | None = None,
    stale_after_days: int = 90,
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    """Everything the home page tracks in one call: model-run history, the
    experiment log, the latest budget/experiment-design recommendations, and
    computed next actions (calibrate completed experiments / refresh a stale
    model / run the recommended next experiment)."""
    import time as _time

    if project_id is not None and not principal.is_dev:
        ensure_project_access(principal, project_id, Role.VIEWER)
    _allowed = _org_project_ids(principal)
    sessions = sessions_store.list_sessions(project_id=project_id)
    if _allowed is not None:
        sessions = [s for s in sessions if s.get("project_id") in _allowed]
    model_runs: list[dict] = []
    latest_design: dict | None = None
    latest_budget: dict | None = None
    for s in sessions:
        tid = s["thread_id"]
        for art in sessions_store.list_artifacts(tid):
            kind, p = art.get("kind"), art.get("payload", {})
            if kind == "model_run":
                model_runs.append(
                    {
                        "model_id": art["id"],
                        "thread_id": tid,
                        "project_id": s.get("project_id"),
                        "run_name": p.get("run_name") or p.get("run_id"),
                        "kpi": p.get("kpi"),
                        "channels": p.get("channels", []),
                        "trend": p.get("trend"),
                        "n_obs": p.get("n_obs"),
                        "summary": (p.get("summary") or "")[:300],
                        "report_path": p.get("report_path"),
                        "created_at": art.get("created_at"),
                    }
                )
            elif kind == "experiment_design":
                if latest_design is None or art["created_at"] > latest_design.get(
                    "created_at", 0
                ):
                    latest_design = {
                        "created_at": art["created_at"],
                        "thread_id": tid,
                        **p,
                    }
            elif kind == "budget_optimization":
                if latest_budget is None or art["created_at"] > latest_budget.get(
                    "created_at", 0
                ):
                    latest_budget = {
                        "created_at": art["created_at"],
                        "thread_id": tid,
                        **p,
                    }
    model_runs.sort(key=lambda m: m.get("created_at") or 0, reverse=True)
    experiments = sessions_store.list_experiments(project_id=project_id)

    now = _time.time()
    last_fit_at = model_runs[0]["created_at"] if model_runs else None
    next_actions: list[dict] = []

    # 1. Completed-but-uncalibrated experiments are the highest-value refresh
    completed = [e for e in experiments if e["status"] == "completed"]
    if completed:
        chs = sorted({e["channel"] for e in completed})
        next_actions.append(
            {
                "type": "calibrate",
                "urgency": "high",
                "title": f"Calibrate the model with {len(completed)} completed experiment(s)",
                "detail": (
                    f"Measured results for {', '.join(chs)} are not folded into a "
                    "fit yet. Refit with the experiment(s) as calibration "
                    "likelihoods so ROI estimates reflect the causal evidence."
                ),
            }
        )

    # 2. Model staleness
    if last_fit_at is None:
        next_actions.append(
            {
                "type": "fit",
                "urgency": "medium",
                "title": "No model fitted yet",
                "detail": "Start an agent session to configure and fit the first MMM.",
            }
        )
    elif (now - last_fit_at) > stale_after_days * 86400:
        age_days = int((now - last_fit_at) / 86400)
        next_actions.append(
            {
                "type": "refresh",
                "urgency": "medium",
                "title": f"Latest model is {age_days} days old",
                "detail": (
                    f"Older than the {stale_after_days}-day refresh window — "
                    "refit on current data (and fold in any new experiments)."
                ),
            }
        )

    # 3. Next recommended experiment (skip channels already being tested)
    if latest_design:
        active = {
            e["channel"] for e in experiments if e["status"] in ("planned", "running")
        }
        pick = next(
            (d for d in latest_design.get("designs", []) if d["channel"] not in active),
            None,
        )
        if pick:
            next_actions.append(
                {
                    "type": "experiment",
                    "urgency": "medium",
                    "title": f"Plan the next experiment: {pick['channel']}",
                    "detail": pick.get("why", ""),
                    "design": pick,
                }
            )

    # 4. Re-test triggers: calibrated evidence that information decay has
    # pushed back over the EIG threshold (see planning.eig).
    if project_id:
        try:
            from mmm_framework.api.history import build_calibration_coverage

            cov = build_calibration_coverage(project_id)
            due = [c for c in cov["channels"] if c.get("retest_due")]
            if due:
                chs = ", ".join(c["channel"] for c in due)
                next_actions.append(
                    {
                        "type": "retest",
                        "urgency": "medium",
                        "title": f"Re-test {len(due)} channel(s) — evidence has decayed",
                        "detail": (
                            f"Experimental evidence for {chs} is old enough that a "
                            "fresh test clears the information-gain threshold. "
                            "Recompute priorities and schedule re-tests."
                        ),
                        "channels": [c["channel"] for c in due],
                    }
                )
        except Exception:
            pass

    return JSONResponse(
        content=safe_json_dumps_load(
            {
                "model_runs": model_runs,
                "experiments": experiments,
                "latest_experiment_design": latest_design,
                "latest_budget_optimization": latest_budget,
                "last_fit_at": last_fit_at,
                "next_actions": next_actions,
            }
        )
    )


# ── Measurement-program history (per-project, from run_metrics snapshots) ─────


@app.get("/projects/{project_id}/history", dependencies=[_proj_read])
async def project_history_endpoint(project_id: str):
    """Cycle-over-cycle trajectory series for the Performance page: per-channel
    ROI posteriors (CI contraction), spend shares (allocation migration),
    share gaps, calibration status, and the portfolio series (misallocation
    proxy, marginal ROI, EVPI). Assembled purely from stored run_metrics rows
    — no model loads."""
    from mmm_framework.api.history import build_history_series

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(content=safe_json_dumps_load(build_history_series(project_id)))


@app.get("/projects/{project_id}/calibration-coverage", dependencies=[_proj_read])
async def calibration_coverage_endpoint(project_id: str, as_of: str | None = None):
    """Channels × evidence tier (calibrated / stale / model_only) with
    information decay applied at read time, plus coverage percentages."""
    from mmm_framework.api.history import build_calibration_coverage

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(
        content=safe_json_dumps_load(
            build_calibration_coverage(project_id, as_of=as_of)
        )
    )


@app.get("/projects/{project_id}/experiment-priorities", dependencies=[_proj_read])
async def experiment_priorities_endpoint(project_id: str, as_of: str | None = None):
    """The latest EIG/EVOI priority grid with decay + registry state applied
    at read time. 404 when the project has no run metrics yet (fit a model
    first, or backfill: python -m mmm_framework.api.backfill)."""
    from mmm_framework.api.history import build_priorities_payload

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    payload = build_priorities_payload(project_id, as_of=as_of)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No run metrics for this project yet. Fit a model (metrics are "
                "recorded automatically) or backfill saved runs with "
                "`python -m mmm_framework.api.backfill`."
            ),
        )
    return JSONResponse(content=safe_json_dumps_load(payload))


# ── Experiment design studio (geo lift / matched-market DiD / flighting) ─────


def _design_inputs(project_id: str, channel: str) -> tuple[str, str]:
    """(dataset_path, kpi) for design computation, from the latest run."""
    import os as _os

    from mmm_framework.api.history import latest_model_run_payload

    run = latest_model_run_payload(project_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail="No model runs in this project yet — fit a model first so the "
            "designer knows the dataset and KPI.",
        )
    dataset_path = run.get("dataset_path")
    kpi = run.get("kpi") or (run.get("spec") or {}).get("kpi")
    if not dataset_path or not _os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"The run's dataset is not on disk ({dataset_path}) — re-upload "
            "or refit before designing.",
        )
    if not kpi:
        raise HTTPException(status_code=404, detail="The run records no KPI.")
    if channel and channel not in (run.get("channels") or [channel]):
        raise HTTPException(
            status_code=400,
            detail=f"'{channel}' is not a channel of the latest run "
            f"({', '.join(run.get('channels') or [])}).",
        )
    return dataset_path, kpi


@app.get("/projects/{project_id}/experiment-design/options", dependencies=[_proj_read])
async def experiment_design_options_endpoint(project_id: str, channel: str):
    """What designs the project's data supports (geo designs need >= 4 geos),
    plus the recommended one."""
    from mmm_framework.planning.design import design_options

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    dataset_path, kpi = _design_inputs(project_id, channel)
    try:
        opts = design_options(dataset_path, kpi, channel)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=safe_json_dumps_load({**opts, "kpi": kpi}))


class ExperimentDesignRequest(BaseModel):
    channel: str
    design_key: str | None = None  # geo_lift | matched_market_did | national_flighting
    # geo designs
    design: str = "scaling"  # holdout | scaling
    intensity_pct: float = 50.0
    n_pairs: int | None = None
    duration: int = 8
    # flighting
    amplitude_pct: float = 50.0
    block_weeks: int = 2
    levels: list[float] | None = None  # multi-level spend multipliers (curve)
    seed: int = 42


@app.post(
    "/projects/{project_id}/experiment-design", dependencies=[_proj_write, _rl_heavy]
)
async def experiment_design_endpoint(project_id: str, body: ExperimentDesignRequest):
    """Compute a runnable experiment design: randomized matched-pair geo lift
    (or observational matched-market DiD) with DiD power/MDE curves, or a
    budget-neutral randomized flighting schedule for national data. Pure data
    computation — no model load."""
    from mmm_framework.planning.design import design_experiment, design_options

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    dataset_path, kpi = _design_inputs(project_id, body.channel)
    try:
        design_key = (
            body.design_key
            or design_options(dataset_path, kpi, body.channel)["recommended"]
        )
        if design_key == "national_flighting":
            kwargs: dict = {
                "duration": int(body.duration),
                "seed": int(body.seed),
                "amplitude_pct": body.amplitude_pct,
                "block_weeks": int(body.block_weeks),
            }
            if body.levels:
                kwargs["levels"] = tuple(float(m) for m in body.levels)
        else:
            kwargs = {
                "duration": int(body.duration),
                "seed": int(body.seed),
                "design": body.design,
                "intensity_pct": body.intensity_pct,
            }
            if body.n_pairs is not None:
                kwargs["n_pairs"] = int(body.n_pairs)
        design = design_experiment(
            dataset_path, kpi, body.channel, design_key=design_key, **kwargs
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=safe_json_dumps_load(design))


# ── Model-anchored experiment economics (async; loads the latest model) ───────


class ExperimentSimulateRequest(BaseModel):
    channel: str
    design_key: str | None = None
    design: str = "scaling"  # holdout | scaling
    intensity_pct: float = 50.0
    n_pairs: int | None = None
    duration: int = 8
    amplitude_pct: float = 50.0
    block_weeks: int = 2
    levels: list[float] | None = None  # multi-level flighting multipliers
    margin: float | None = None
    price: float | None = None
    kpi_kind: str = "revenue"
    seed: int = 42
    max_draws: int = 100


# Strong refs to in-flight jobs so a bare create_task can't be GC'd mid-run.
_SIM_TASKS: set = set()


def _sim_job_patch(job_id: str, **patch) -> None:
    art = sessions_store.get_artifact(job_id)
    if art is None:
        return
    payload = dict(art.get("payload") or {})
    payload.update(patch)
    sessions_store.update_artifact_payload(job_id, payload)


def _resolve_project_margin(
    project_id: str, body_margin: float | None, body_price: float | None
) -> tuple[float | None, float | None]:
    """(margin, price) — explicit body wins, else the project's saved economics
    preference (set via the preferences store), else (None, None)."""
    if body_margin is not None:
        return body_margin, body_price
    try:
        econ = sessions_store.get_preference(project_id, "economics")
        if isinstance(econ, dict):
            return econ.get("gross_margin"), econ.get("price", body_price)
    except Exception:
        pass
    return None, body_price


def _load_and_run_op(
    synthetic_tid: str,
    run_name: str,
    spec: dict | None,
    dataset_path: str | None,
    op_name: str,
    op_kwargs: dict,
) -> dict:
    """SYNC worker (one asyncio.to_thread call): set the thread, load the latest
    saved model into the in-process cache, and run the named model op DIRECTLY
    against it. All three MUST share one worker context (F11: to_thread copies
    the caller context; a ContextVar set in a separate call won't survive). The
    ops are pure read-only compute, so a direct call is safe and avoids the
    kernel-impl ambiguity (the model lives in the in-process MODEL_CACHE)."""
    from mmm_framework.agents import model_ops
    from mmm_framework.agents.runtime import MODEL_CACHE, set_current_thread
    from mmm_framework.agents.tools import load_model_core

    set_current_thread(synthetic_tid)
    load_res = load_model_core(synthetic_tid, run_name, spec, dataset_path)
    if not load_res.get("ok"):
        return {"error": load_res.get("message", "Could not load the saved model.")}
    mmm = MODEL_CACHE.get("fitted_model")
    results = MODEL_CACHE.get("fit_results")
    op = model_ops.OPS.get(op_name)
    if op is None:
        return {"error": f"Unknown model op: {op_name}"}
    return op(mmm, results, **op_kwargs)


async def _run_model_op_job(
    job_id: str,
    synthetic_tid: str,
    run: dict | None,
    op_name: str,
    op_kwargs: dict,
    result_key: str,
) -> None:
    """Run a model op in the background and persist its result_key projection
    onto the job artifact (pending → running → done/error). Never lets the job
    vanish: a raising worker writes status='error'."""
    try:
        _sim_job_patch(job_id, status="running")
        if not run or not run.get("run_name"):
            _sim_job_patch(
                job_id,
                status="error",
                error="No saved model run for this project — fit a model first.",
            )
            return
        res = await asyncio.to_thread(
            _load_and_run_op,
            synthetic_tid,
            run.get("run_name"),
            run.get("spec"),
            run.get("dataset_path"),
            op_name,
            op_kwargs,
        )
        if res.get("error"):
            _sim_job_patch(job_id, status="error", error=res["error"])
            return
        out = (res.get("dashboard") or {}).get(result_key)
        _sim_job_patch(
            job_id,
            status="done",
            result=safe_json_dumps_load(out) if out else None,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Model-op job failed (%s): %s", op_name, job_id)
        _sim_job_patch(job_id, status="error", error=str(e))


def _spawn_job_task(coro) -> None:
    """Fire-and-forget a background job with a strong ref (a bare create_task
    can be GC'd mid-run) and crash logging."""
    task = asyncio.create_task(coro)
    _SIM_TASKS.add(task)
    task.add_done_callback(
        lambda f: (
            _SIM_TASKS.discard(f),
            f.cancelled()
            or (f.exception() and logger.error("Job crashed: %s", f.exception())),
        )
    )


async def _run_simulation_job(
    job_id: str, synthetic_tid: str, run: dict | None, op_kwargs: dict
) -> None:
    await _run_model_op_job(
        job_id,
        synthetic_tid,
        run,
        "experiment_economics",
        op_kwargs,
        "experiment_economics",
    )


@app.post(
    "/projects/{project_id}/experiment-design/simulate",
    dependencies=[_proj_write, _rl_heavy],
)
async def start_experiment_simulation(project_id: str, body: ExperimentSimulateRequest):
    """Start a NON-BLOCKING model-anchored economics + A/A·A/B simulation. Loads
    the project's latest saved model in the background and runs the
    experiment_economics op; poll the returned job_id for the result."""
    from mmm_framework.api.history import latest_model_run_payload

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    dataset_path, kpi = _design_inputs(project_id, body.channel)
    run = latest_model_run_payload(project_id)

    margin, price = _resolve_project_margin(project_id, body.margin, body.price)
    design_params = {
        "dataset_path": dataset_path,
        "kpi": kpi,
        "channel": body.channel,
        "design_key": body.design_key,
        "duration": int(body.duration),
        "design": body.design,
        "intensity_pct": float(body.intensity_pct),
        "amplitude_pct": float(body.amplitude_pct),
        "block_weeks": int(body.block_weeks),
        "seed": int(body.seed),
    }
    if body.n_pairs is not None:
        design_params["n_pairs"] = int(body.n_pairs)
    if body.levels:
        design_params["levels"] = [float(m) for m in body.levels]
    op_kwargs = {
        "design_params": design_params,
        "run_simulation": True,
        "margin": margin,
        "price": price,
        "kpi_kind": body.kpi_kind,
        "max_draws": int(body.max_draws),
    }

    # Server-minted, project-scoped thread id (never client-supplied → hosted-safe).
    synthetic_tid = f"__simjobs__{project_id}"
    job = sessions_store.add_artifact(
        synthetic_tid,
        "experiment_simulation",
        {
            "status": "pending",
            "project_id": project_id,
            "channel": body.channel,
            "result": None,
            "error": None,
        },
    )
    job_id = job["id"]
    _spawn_job_task(_run_simulation_job(job_id, synthetic_tid, run, op_kwargs))
    return JSONResponse(
        status_code=202, content={"job_id": job_id, "status": "pending"}
    )


@app.get(
    "/projects/{project_id}/experiment-design/simulate/{job_id}",
    dependencies=[_proj_read],
)
async def get_experiment_simulation(project_id: str, job_id: str):
    """Poll an experiment-simulation job: {status, result|null, error|null}."""
    art = sessions_store.get_artifact(job_id)
    if art is None or (art.get("payload") or {}).get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Simulation job not found.")
    return JSONResponse(content=safe_json_dumps_load(art["payload"]))


# ── Model Garden registry (org-scoped) ───────────────────────────────────────
# Backs the Atelier UI + governance: register (from the editor), discover,
# re-test, the human PUBLISH gate, and retirement. Registration shares its core
# with the agent's `register_garden_model` tool (agents/garden_registry.py).


class GardenPromoteRequest(BaseModel):
    note: str = ""


class GardenRegisterRequest(BaseModel):
    source_code: str
    name: str
    docs: str = ""
    version: int | None = None
    tags: list | None = None
    dataset_schema: dict | None = None
    recommended_fit: dict | None = None


class GardenSourceRequest(BaseModel):
    """A bare source payload for the editor's IDE tools (lint / format)."""

    source_code: str = ""


class GardenDocsRequest(BaseModel):
    """In-place docs (markdown) edit for a non-published garden version."""

    docs: str = ""


class CopilotTurn(BaseModel):
    role: str = "user"  # "user" | "assistant"
    content: str = ""


class NotebookCopilotContext(BaseModel):
    """Notebook context attached to a copilot turn so the assistant can diagnose
    a failed cell: the cell's code + traceback, the dataset preview, and the
    sibling cells (variables flow between cells in one shared kernel)."""

    cell_code: str = ""
    traceback: str = ""
    dataset_preview: str | None = None
    other_cells: list[str] = []
    is_error: bool = False


class GardenCopilotRequest(BaseModel):
    """A modeling-copilot turn: the running chat plus the current editor source
    so the assistant grounds its answer in the user's actual code. When invoked
    from the Atelier notebook, ``notebook`` carries the failing/active cell so the
    same copilot diagnoses cell-execution errors and rewrites the cell."""

    messages: list[CopilotTurn] = []
    source_code: str = ""
    notebook: NotebookCopilotContext | None = None


class CopilotChatSaveRequest(BaseModel):
    """Persist the Atelier copilot chat for one ``(model, version, surface)``.

    The chat is scoped per model/version so each model keeps its own running
    conversation; ``surface`` separates the editor copilot from the notebook
    copilot. PUT an empty ``messages`` to clear it. ``messages`` are stored as
    free-form dicts (id/role/content/targetCellId) so the client owns the shape.
    """

    name: str
    version: int | None = None
    surface: str = "editor"  # "editor" | "notebook"
    messages: list[dict[str, Any]] = []


def _garden_org(principal: AuthContext) -> str:
    """Org that scopes garden visibility — matches the agent-side resolution
    (``sessions_store.resolve_org_id``) so both surfaces see the same models."""
    return principal.org_id or sessions_store.DEFAULT_ORG_ID


@app.post("/model-garden")
async def register_garden_model_endpoint(
    body: GardenRegisterRequest, principal: PrincipalDep
):
    """Register a bespoke model (source defining a BayesianMMM subclass) as a
    DRAFT. Analyst+ role. The source is AST-validated (never executed here) and
    stored in the org's garden; POST the test endpoint next to fit + grade it."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="Registering a model requires an analyst+ role."
        )
    from mmm_framework.agents.garden_registry import register_garden_model_core

    try:
        row = register_garden_model_core(
            org_id=_garden_org(principal),
            source_code=body.source_code,
            name=body.name,
            docs=body.docs,
            version=body.version,
            tags=body.tags,
            dataset_schema=body.dataset_schema,
            recommended_fit=body.recommended_fit,
            owner_user_id=principal.user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return JSONResponse(status_code=201, content=safe_json_dumps_load(row))


def _merge_lint_problems(
    contract_problems: list[dict], ruff_problems: list[dict]
) -> list[dict]:
    """Combine the AST contract checks with ruff's real Python diagnostics.

    - When the source doesn't parse, ruff's E999 (which carries a column) is the
      precise marker, so drop the contract's generic "Syntax error" stand-in.
    - Drop the contract "No issues found" all-clear note once ruff has anything
      to say.
    - Sort line-anchored problems first (by line), global notes last.
    """
    ruff_has_error = any(p.get("severity") == "error" for p in ruff_problems)
    contract = list(contract_problems)
    if ruff_has_error:
        contract = [
            p
            for p in contract
            if not str(p.get("message", "")).startswith("Syntax error")
        ]
    if ruff_problems:
        contract = [
            p
            for p in contract
            if not (
                p.get("severity") == "info"
                and str(p.get("message", "")).startswith("No issues found")
            )
        ]
    combined = ruff_problems + contract
    combined.sort(key=lambda p: (p.get("line") is None, p.get("line") or 0))
    return combined


@app.post("/model-garden/lint")
async def garden_lint_endpoint(body: GardenSourceRequest, principal: PrincipalDep):
    """"Problems" check for the Atelier editor: real Python diagnostics (ruff —
    undefined names, unused imports, redefinitions, syntax, with line/column
    spans) merged with the AST-only garden-contract conventions. Neither path
    executes the source. Analyst+ role."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="Linting requires an analyst+ role."
        )
    from mmm_framework.agents.garden_authoring import ruff_lint, static_authoring_lint

    # Offload the AST parse + blocking ruff subprocess off the event loop — the
    # editor fires this on an 800 ms debounced auto-lint loop, so a synchronous
    # subprocess.run would stall concurrent SSE/fit requests (module convention).
    def _lint() -> tuple[str | None, list[dict]]:
        cls, contract = static_authoring_lint(body.source_code)
        return cls, _merge_lint_problems(contract, ruff_lint(body.source_code))

    class_name, problems = await asyncio.to_thread(_lint)
    return JSONResponse(
        content={
            "class_name": class_name,
            "problems": problems,
            "ok": not any(p["severity"] == "error" for p in problems),
        }
    )


@app.post("/model-garden/format")
async def garden_format_endpoint(body: GardenSourceRequest, principal: PrincipalDep):
    """Format the editor source (ruff, black fallback) for the IDE *Format*
    button. Returns the formatted source or a one-line error. Analyst+ role."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="Formatting requires an analyst+ role."
        )
    from mmm_framework.agents.garden_authoring import format_source

    # Offload the blocking ruff/black subprocess off the event loop (see lint).
    formatted, error = await asyncio.to_thread(format_source, body.source_code)
    return JSONResponse(content={"formatted": formatted, "error": error})


@app.post("/model-garden/copilot", dependencies=[_rl_chat])
async def garden_copilot_endpoint(
    body: GardenCopilotRequest,
    raw_request: Request,
    x_api_key: str | None = Header(None),
    x_model_name: str | None = Header(None),
    x_base_url: str | None = Header(None),
    x_provider: str | None = Header(None),
    principal: PrincipalDep = _DEV_PRINCIPAL,
):
    """Stream a Bayesian-modeling copilot answer (SSE) for the Atelier editor.

    Stateless: the client sends the running chat + the current editor source each
    turn; the server grounds a focused expert system prompt (the oracle contract
    + PyMC/MMM authoring knowledge) on that source and streams the model's tokens
    using the SAME ``data: {...}\\n\\n`` / ``[DONE]`` framing as ``/chat``.
    Analyst+ role."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="The modeling copilot requires an analyst+ role."
        )
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from mmm_framework.agents.garden_authoring import build_copilot_system_prompt

    system_prompt = build_copilot_system_prompt(
        body.source_code,
        notebook=body.notebook.model_dump() if body.notebook else None,
    )
    lc_messages: list = [SystemMessage(content=system_prompt)]
    # Keep the last dozen turns; drop empties.
    for turn in [t for t in body.messages if (t.content or "").strip()][-12:]:
        if turn.role == "assistant":
            lc_messages.append(AIMessage(content=turn.content))
        else:
            lc_messages.append(HumanMessage(content=turn.content))
    if len(lc_messages) == 1:
        raise HTTPException(status_code=400, detail="No message to answer.")

    async def event_generator():
        try:
            llm = get_llm(x_model_name, x_api_key, x_base_url, x_provider)
            async for chunk in llm.astream(lc_messages):
                if await raw_request.is_disconnected():
                    break
                text = _copilot_chunk_text(getattr(chunk, "content", ""))
                if text:
                    payload = {"type": "token", "content": text}
                    yield f"data: {safe_json_dumps(payload)}\n\n"
        except Exception as e:  # noqa: BLE001 — surface as an in-stream error event
            logger.exception("garden copilot stream failed")
            yield f"data: {safe_json_dumps({'type': 'error', 'content': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _copilot_chunk_text(content: Any) -> str:
    """Flatten a streamed chunk's content to text — handles a plain string or
    Anthropic-style lists of content blocks (``{"type": "text", "text": ...}``)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict):
                out.append(block.get("text") or block.get("content") or "")
        return "".join(out)
    return ""


# ── Atelier notebook ──────────────────────────────────────────────────────────
# A Jupyter-like demo/test space scoped to one bespoke model: upload a dataset,
# run free-form Python cells against the model (the LIVE editor buffer or a
# registered version), and track plot/table/markdown outputs. Cells execute in
# the same sandboxed session kernel the compatibility suite uses (so untrusted
# author source never imports in the host), via the existing non-blocking job
# machinery (mirrors the garden-test pattern). NB: these routes are registered
# BEFORE the parametric ``/model-garden/{name}/...`` routes so ``notebook`` is
# never captured as a ``{name}``.

# Per-notebook source revision last staged+imported into the kernel — lets us
# skip the (re)import setup cell when the author hasn't edited the model.
_NOTEBOOK_SOURCE_REV: dict[str, str] = {}


class NotebookCellRequest(BaseModel):
    """Run one code cell against the model. ``version=None`` => the live editor
    buffer in ``source_code`` (re-imported when ``source_rev`` changes); else a
    registered version (source resolved from the registry)."""

    name: str = "untitled"
    version: int | None = None
    source_code: str | None = None
    source_rev: str = ""
    code: str = ""
    dataset_path: str | None = None


class NotebookSaveRequest(BaseModel):
    name: str = "untitled"
    version: int | None = None
    cells: list[dict] = []
    dataset: dict | None = None


def _notebook_tid(org_id: str, name: str, version: int | None) -> str:
    """Deterministic synthetic thread id for a notebook (one warm kernel +
    workspace per (org, model, source)). Consistent with the existing
    ``__gardenjobs__{org}`` synthetic-context pattern."""
    import re

    raw = f"{name}__v{version}" if version is not None else f"{name}__draft"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:80]
    return f"__atelier_nb__{org_id}__{safe}"


# The staged model source is written as a NON-``.py`` file so a dev server run
# with ``uvicorn --reload`` (which watches ``*.py`` under the repo) does NOT
# restart on every cell — a restart wipes the warm in-process kernel mid-session
# (the `data` from an earlier cell would vanish). We import it via an explicit
# ``SourceFileLoader``, which reads Python source regardless of extension.
_NOTEBOOK_SRC_FILE = "notebook_model_source.txt"


def _notebook_stage_source(name: str, ver_seg: str, tid: str, text: str) -> str:
    """Write the model source into the notebook workspace as a non-``.py`` file.
    Content-aware: skips the write when the file is already identical (avoids
    disk churn and any reload watcher firing). Returns the staged path."""
    from mmm_framework.agents import workspace as _ws

    dest = _ws.garden_loaded_dir(name, ver_seg, tid) / _NOTEBOOK_SRC_FILE
    try:
        if dest.exists() and dest.read_text(encoding="utf-8") == (text or ""):
            return str(dest)
    except Exception:
        pass
    dest.write_text(text or "", encoding="utf-8")
    return str(dest)


def _notebook_setup_code(staged_path: str, rev: str) -> str:
    """Kernel-side cell that (re)imports the staged model source under a fresh
    module name and binds ``GardenModel`` to the resolved class. Runs ONLY in the
    session kernel — untrusted source never imports in the host. Uses an explicit
    ``SourceFileLoader`` so the non-``.py`` staged file still imports as Python."""
    import re

    mod = "garden_user_model_" + (re.sub(r"[^A-Za-z0-9_]+", "_", rev or "") or "x")
    return (
        "import importlib.util as _ilu, importlib.machinery as _ilm, sys as _sys\n"
        f"_loader = _ilm.SourceFileLoader({mod!r}, {staged_path!r})\n"
        f"_spec = _ilu.spec_from_loader({mod!r}, _loader)\n"
        "_mod = _ilu.module_from_spec(_spec)\n"
        f"_sys.modules[{mod!r}] = _mod\n"
        "_loader.exec_module(_mod)\n"
        "from mmm_framework.garden.contract import find_garden_class as _fgc\n"
        "GardenModel = _fgc(_mod)\n"
        "print('Loaded garden model class:', GardenModel.__name__)\n"
    )


def _notebook_cell_sync(req: dict) -> dict:
    """SYNC worker (ONE asyncio.to_thread per the ContextVar rule): set the
    thread, (re)stage+import the model source if it changed, then run the user
    cell in the session kernel and map its ExecuteResult to JSON-safe output
    refs (the same content-addressing ``execute_python`` uses)."""
    from mmm_framework.agents import tools as _tools
    from mmm_framework.agents import workspace as _ws
    from mmm_framework.agents.kernels import KernelContext
    from mmm_framework.agents.runtime import set_current_thread
    from mmm_framework.agents.tables import publish_tables

    tid = req["tid"]
    set_current_thread(tid)
    work_dir = str(_ws.thread_dir(tid))
    kernel = _tools._KERNELS.get_or_spawn(tid)

    # 1. (Re)stage + import the model source when the buffer/version changed.
    # Resolve the source TEXT: the live editor buffer, or the registered
    # version's file read off disk.
    src_text = req.get("source_code")
    if src_text is None and req.get("source_path"):
        try:
            with open(req["source_path"], "r", encoding="utf-8") as _f:
                src_text = _f.read()
        except Exception:
            src_text = None
    if src_text is not None and _NOTEBOOK_SOURCE_REV.get(tid) != req.get("source_rev"):
        staged = _notebook_stage_source(req["name"], req["ver_seg"], tid, src_text)
        setup = kernel.execute(
            _notebook_setup_code(staged, req.get("source_rev") or ""),
            KernelContext(
                thread_id=tid,
                work_dir=work_dir,
                dataset_path=req.get("dataset_path"),
            ),
        )
        if setup.is_error:
            # The model source itself failed to import — that IS the test
            # result the author needs. Don't cache the rev (retry next run).
            return {
                "stdout": setup.stdout,
                "plots": [],
                "tables": [],
                "is_error": True,
                "setup_error": True,
            }
        _NOTEBOOK_SOURCE_REV[tid] = req.get("source_rev") or ""

    # 2. Run the user cell.
    result = kernel.execute(
        req.get("code") or "",
        KernelContext(
            thread_id=tid, work_dir=work_dir, dataset_path=req.get("dataset_path")
        ),
    )

    # 3. Content-address plots/tables; keep only lightweight refs.
    plot_refs: list[dict] = []
    for fig in result.plots or []:
        try:
            pid = _ws.store_plot(fig, tid)
        except Exception:  # noqa: BLE001 — oversize/invalid figure: drop it
            continue
        layout = fig.get("layout") or {}
        t = layout.get("title")
        title = (
            t.get("text") if isinstance(t, dict) else (t if isinstance(t, str) else "")
        )
        plot_refs.append({"id": pid, "title": title or ""})
    table_refs: list[dict] = []
    if result.tables:
        table_refs, _dropped = publish_tables(result.tables, {}, tid)

    return {
        "stdout": result.stdout,
        "plots": plot_refs,
        "tables": table_refs,
        "is_error": bool(result.is_error),
    }


async def _run_notebook_cell_job(job_id: str, req: dict) -> None:
    try:
        _sim_job_patch(job_id, status="running")
        res = await asyncio.to_thread(_notebook_cell_sync, req)
        _sim_job_patch(job_id, status="done", result=safe_json_dumps_load(res))
    except Exception as e:  # noqa: BLE001
        logger.exception("notebook cell job failed: %s", job_id)
        _sim_job_patch(job_id, status="error", error=str(e))


def _notebook_starter(name: str) -> list[dict]:
    """A runnable starter notebook: it fits the model on a labelled synthetic
    world out of the box, and points the author at their uploaded data."""
    intro = (
        f"# Demo & test: **{name}**\n\n"
        "Run the cells below to fit your model and inspect what it recovers. "
        "Upload an **MFF long-format** CSV with the control above to test it on "
        "your own data — otherwise it runs on a labelled synthetic world. "
        "`GardenModel` is your model class (the live editor source); `df` is your "
        "uploaded data; `show_table(...)` and `fig.show()` render into this notebook."
    )
    load = (
        "# Use your uploaded dataset if present, else a labelled synthetic world\n"
        "# (so this notebook runs out of the box).\n"
        "try:\n"
        "    data, src = df, dataset_path\n"
        "    kpi, channels = 'Sales', None  # set/edit `channels` below if not MFF\n"
        "except NameError:\n"
        "    from mmm_framework.synth import generate_mff\n"
        "    data, _ans = generate_mff('realistic')\n"
        "    src = 'demo.csv'; data.to_csv(src, index=False)\n"
        "    kpi, channels = 'Sales', list(_ans.get('channels') or [])\n"
        "show_table(data.head(20), title='Input data (first 20 rows)')\n"
        "print('rows:', len(data), '| variables:', "
        "list(dict.fromkeys(data['VariableName'])) if 'VariableName' in data else list(data.columns))\n"
    )
    fit = (
        "# Build a spec from the data's MFF structure and fit your model (fast MAP).\n"
        "# Edit `channels`/`controls` to map your columns if it isn't standard MFF.\n"
        "all_vars = list(dict.fromkeys(data['VariableName'].tolist()))\n"
        "if not channels:\n"
        "    channels = [v for v in all_vars if v != kpi][: max(1, len(all_vars) // 2)]\n"
        "controls = [v for v in all_vars if v != kpi and v not in channels]\n"
        "spec = {\n"
        "    'kpi': kpi,\n"
        "    'media_channels': [{'name': c} for c in channels],\n"
        "    'control_variables': [{'name': c} for c in controls],\n"
        "    'trend': {'type': 'linear'},\n"
        "    'seasonality': {'yearly': 0, 'monthly': 0, 'weekly': 0},\n"
        "    'inference': {'method': 'map', 'chains': 1, 'draws': 200, 'tune': 200},\n"
        "}\n"
        "from mmm_framework.agents.fitting import build_model\n"
        "mmm = build_model(spec, src, model_cls=GardenModel)\n"
        "results = mmm.fit(method='map')\n"
        "print('Fitted', GardenModel.__name__, 'on channels:', channels)\n"
    )
    roi = (
        "# Recovered ROI by channel (with uncertainty) + a quick bar chart.\n"
        "from mmm_framework.reporting.helpers import compute_roi_with_uncertainty\n"
        "roi = compute_roi_with_uncertainty(mmm, hdi_prob=0.9)\n"
        "show_table(roi, title='Recovered ROI by channel')\n"
        "import plotly.express as px\n"
        "cols = list(roi.columns)\n"
        "ycol = next((c for c in cols if 'roi' in c.lower() or 'roas' in c.lower()), cols[1])\n"
        "fig = px.bar(roi, x=cols[0], y=ycol, title='Recovered ROI by channel')\n"
        "fig.show()\n"
    )
    estimands = (
        "# Declarative estimands: the counterfactual quantities your model declares\n"
        "# (DEFAULT_ESTIMANDS) or its built-in defaults, realized as mean + HDI.\n"
        "# Each is a pre-specified causal contrast (not a post-hoc summary).\n"
        "import pandas as pd\n"
        "res = mmm.evaluate_estimands()  # dict[name -> EstimandResult]\n"
        "rows = []\n"
        "for nm, r in res.items():\n"
        "    if r.status != 'ok':\n"
        "        # Still surface unsupported estimands with the reason (missing capability).\n"
        "        rows.append({'estimand': nm, 'mean': None, 'hdi_low': None,\n"
        "                     'hdi_high': None, 'status': r.status,\n"
        "                     'units': r.reason or r.units})\n"
        "        continue\n"
        "    rows.append({'estimand': nm, 'mean': r.mean, 'hdi_low': r.hdi_low,\n"
        "                 'hdi_high': r.hdi_high, 'status': r.status, 'units': r.units})\n"
        "df_est = pd.DataFrame(rows, columns=['estimand', 'mean', 'hdi_low',\n"
        "                                     'hdi_high', 'status', 'units'])\n"
        "show_table(df_est, title='Declared estimands (counterfactual quantities of interest)')\n"
    )
    return [
        {"id": "c1", "type": "markdown", "source": intro, "outputs": None},
        {"id": "c2", "type": "code", "source": load, "outputs": None},
        {"id": "c3", "type": "code", "source": fit, "outputs": None},
        {"id": "c4", "type": "code", "source": roi, "outputs": None},
        {"id": "c5", "type": "code", "source": estimands, "outputs": None},
    ]


@app.post("/model-garden/notebook/dataset", dependencies=[_rl_heavy])
async def notebook_upload_dataset(
    name: str,
    principal: PrincipalDep,
    file: UploadFile = File(...),
    version: int | None = None,
):
    """Stage a dataset into the notebook's workspace (so cells auto-bind it as
    ``df``). Analyst+ role; org-scoped synthetic thread (not a session, so no
    ``_sess_write``)."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="Uploading requires an analyst+ role."
        )
    from mmm_framework.agents import workspace as _ws

    org_id = _garden_org(principal)
    tid = _notebook_tid(org_id, name, version)
    upload_dir = str(_ws.thread_dir(tid))
    safe_name = _safe_upload_name(file.filename, "data.csv")
    dest = str(_ws.safe_join(Path(upload_dir), safe_name))
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    preview: str | None = None
    kind = "upload"
    if safe_name.lower().endswith((".csv", ".tsv", ".txt")):
        kind = "dataset"
        try:
            import itertools

            with open(dest, "r", errors="ignore") as f:
                preview = "".join(itertools.islice(f, 6))  # EOF-safe (short files)
        except Exception:
            preview = None
    elif safe_name.lower().endswith((".xlsx", ".xls", ".parquet")):
        kind = "dataset"
    return {
        "path": dest,
        "filename": safe_name,
        "kind": kind,
        "preview": preview,
        "size_bytes": os.path.getsize(dest),
    }


@app.post("/model-garden/notebook/cell", dependencies=[_rl_heavy])
async def start_notebook_cell(body: NotebookCellRequest, principal: PrincipalDep):
    """Start a NON-BLOCKING run of one code cell against the model. Returns a
    ``job_id`` to poll. Analyst+ role."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="Running cells requires an analyst+ role."
        )
    org_id = _garden_org(principal)
    tid = _notebook_tid(org_id, body.name, body.version)
    req: dict = {
        "tid": tid,
        "name": body.name,
        "ver_seg": "draft" if body.version is None else str(body.version),
        "source_code": body.source_code,
        "source_path": None,
        "source_rev": body.source_rev or "",
        "dataset_path": body.dataset_path,
        "code": body.code,
    }
    if body.version is not None:
        row = sessions_store.get_garden_model(
            org_id=org_id, name=body.name, version=int(body.version)
        )
        if row is None:
            raise HTTPException(
                status_code=404, detail="Garden model version not found."
            )
        req["source_code"] = None
        req["source_path"] = row["source_path"]
        req["source_rev"] = f"v{body.version}"
    job = sessions_store.add_artifact(
        tid,
        "notebook_cell_job",
        {"status": "pending", "org_id": org_id, "result": None, "error": None},
    )
    _spawn_job_task(_run_notebook_cell_job(job["id"], req))
    return JSONResponse(
        status_code=202, content={"job_id": job["id"], "status": "pending"}
    )


@app.get("/model-garden/notebook/cell/{job_id}")
async def get_notebook_cell(job_id: str, principal: PrincipalDep):
    """Poll a notebook cell run: {status, result|null, error|null}. result =
    {stdout, plots:[{id,title}], tables:[{id,title,...}], is_error}."""
    art = sessions_store.get_artifact(job_id)
    org_id = _garden_org(principal)
    if art is None or (art.get("payload") or {}).get("org_id") != org_id:
        raise HTTPException(status_code=404, detail="Cell job not found.")
    return JSONResponse(content=safe_json_dumps_load(art["payload"]))


@app.get("/model-garden/notebook")
async def get_notebook(name: str, principal: PrincipalDep, version: int | None = None):
    """The persisted notebook doc for this (model, source), or a seeded starter
    when none exists yet. Analyst+ role."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="The notebook requires an analyst+ role."
        )
    org_id = _garden_org(principal)
    tid = _notebook_tid(org_id, name, version)
    docs = [
        a
        for a in sessions_store.list_artifacts(tid)
        if a.get("kind") == "atelier_notebook"
    ]
    if docs:
        return JSONResponse(content=safe_json_dumps_load(docs[-1]["payload"]))
    # No persisted doc yet: seed the model's CURATED demo notebook from its
    # manifest (a registration-time walkthrough), else the generic starter.
    cells = _model_demo_notebook(org_id, name, version) or _notebook_starter(name)
    return JSONResponse(
        content={
            "cells": cells,
            "dataset": None,
            "name": name,
            "version": version,
            "seeded": True,
        }
    )


def _model_demo_notebook(org_id: str, name: str, version: int | None) -> list | None:
    """A model's curated demo notebook (``manifest["demo_notebook"]``) if it
    declared one at registration, else ``None`` (use the generic starter)."""
    try:
        if version is not None:
            row = sessions_store.get_garden_model(
                org_id=org_id, name=name, version=int(version)
            )
        else:
            row = sessions_store.get_latest_garden_model(org_id, name)
    except Exception:  # noqa: BLE001 — never block the notebook on lookup
        return None
    cells = (row or {}).get("manifest", {}).get("demo_notebook")
    return cells if isinstance(cells, list) and cells else None


@app.put("/model-garden/notebook")
async def save_notebook(body: NotebookSaveRequest, principal: PrincipalDep):
    """Upsert the notebook doc (one ``atelier_notebook`` artifact per notebook).
    Outputs are stored as content-addressed refs, so they survive. Analyst+."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(status_code=403, detail="Saving requires an analyst+ role.")
    org_id = _garden_org(principal)
    tid = _notebook_tid(org_id, body.name, body.version)
    payload = {
        "cells": body.cells,
        "dataset": body.dataset,
        "name": body.name,
        "version": body.version,
        "org_id": org_id,
    }
    docs = [
        a
        for a in sessions_store.list_artifacts(tid)
        if a.get("kind") == "atelier_notebook"
    ]
    if docs:
        sessions_store.update_artifact_payload(docs[-1]["id"], payload)
        art_id = docs[-1]["id"]
    else:
        art_id = sessions_store.add_artifact(tid, "atelier_notebook", payload)["id"]
    return JSONResponse(content={"saved": True, "id": art_id})


# --------------------------------------------------------------------------- #
# Atelier copilot chat persistence — one singleton chat per
# (org, model, version, surface), stored as a `copilot_chat` artifact under a
# synthetic thread id (mirrors the notebook-doc pattern). Scoping the chat per
# model/version is the whole point: each model keeps its own conversation, and
# Clear (PUT with empty messages) wipes only that model's chat.
# --------------------------------------------------------------------------- #
_COPILOT_CHAT_KIND = "copilot_chat"
#: Cap on persisted turns — bounds artifact size on a long-running conversation.
_COPILOT_CHAT_MAX_MESSAGES = 200


def _copilot_surface(surface: str | None) -> str:
    """Normalize the copilot surface to the two known values (default editor)."""
    return "notebook" if surface == "notebook" else "editor"


def _copilot_tid(org_id: str, name: str, version: int | None, surface: str) -> str:
    """Synthetic thread id scoping a persisted copilot chat to one
    (org, model, version, surface) — consistent with ``_notebook_tid``."""
    import re

    raw = f"{name}__v{version}" if version is not None else f"{name}__draft"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:80]
    return f"__atelier_copilot__{org_id}__{_copilot_surface(surface)}__{safe}"


@app.get("/model-garden/copilot/chat")
async def get_copilot_chat(
    name: str,
    principal: PrincipalDep,
    version: int | None = None,
    surface: str = "editor",
):
    """The persisted Atelier copilot chat for this (model, version, surface), or
    an empty chat when none exists yet. Analyst+ role."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="The copilot requires an analyst+ role."
        )
    org_id = _garden_org(principal)
    tid = _copilot_tid(org_id, name, version, surface)
    docs = [
        a
        for a in sessions_store.list_artifacts(tid)
        if a.get("kind") == _COPILOT_CHAT_KIND
    ]
    if docs:
        return JSONResponse(content=safe_json_dumps_load(docs[-1]["payload"]))
    return JSONResponse(
        content={
            "messages": [],
            "name": name,
            "version": version,
            "surface": _copilot_surface(surface),
        }
    )


@app.put("/model-garden/copilot/chat")
async def save_copilot_chat(body: CopilotChatSaveRequest, principal: PrincipalDep):
    """Upsert the Atelier copilot chat (one artifact per model/version/surface).
    PUT an empty ``messages`` to clear it. Analyst+ role."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(status_code=403, detail="Saving requires an analyst+ role.")
    org_id = _garden_org(principal)
    surface = _copilot_surface(body.surface)
    tid = _copilot_tid(org_id, body.name, body.version, surface)
    messages = [m for m in (body.messages or []) if isinstance(m, dict)][
        -_COPILOT_CHAT_MAX_MESSAGES:
    ]
    payload = {
        "messages": messages,
        "name": body.name,
        "version": body.version,
        "surface": surface,
        "org_id": org_id,
    }
    docs = [
        a
        for a in sessions_store.list_artifacts(tid)
        if a.get("kind") == _COPILOT_CHAT_KIND
    ]
    if docs:
        sessions_store.update_artifact_payload(docs[-1]["id"], payload)
        art_id = docs[-1]["id"]
    else:
        art_id = sessions_store.add_artifact(tid, _COPILOT_CHAT_KIND, payload)["id"]
    return JSONResponse(content={"saved": True, "id": art_id})


@app.get("/model-garden")
async def list_garden_models_endpoint(
    principal: PrincipalDep,
    status: str | None = None,
    name: str | None = None,
    all_versions: bool = False,
):
    """List the org's Model Garden models (latest version per name by default)."""
    org_id = _garden_org(principal)
    rows = sessions_store.list_garden_models(
        org_id,
        name=name,
        status=status,
        latest_only=(name is None and not all_versions),
    )
    return JSONResponse(content=safe_json_dumps_load({"models": rows}))


@app.get("/model-garden/{name}/versions")
async def list_garden_versions_endpoint(name: str, principal: PrincipalDep):
    """Every version of one garden model, newest first."""
    org_id = _garden_org(principal)
    return JSONResponse(
        content=safe_json_dumps_load(
            {"versions": sessions_store.list_garden_versions(org_id, name)}
        )
    )


@app.get("/model-garden/{name}/{version}")
async def get_garden_model_endpoint(name: str, version: int, principal: PrincipalDep):
    """One garden model version (incl. its manifest + compatibility report)."""
    org_id = _garden_org(principal)
    row = sessions_store.get_garden_model(
        org_id=org_id, name=name, version=int(version)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Garden model not found.")
    return JSONResponse(content=safe_json_dumps_load(row))


@app.get("/model-garden/{name}/{version}/source")
async def get_garden_source(name: str, version: int, principal: PrincipalDep):
    """The stored model source text (for the Atelier editor)."""
    org_id = _garden_org(principal)
    row = sessions_store.get_garden_model(
        org_id=org_id, name=name, version=int(version)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Garden model not found.")
    from mmm_framework.agents.garden_registry import read_garden_source

    return JSONResponse(content={"source_code": read_garden_source(row) or ""})


@app.patch("/model-garden/{name}/{version}")
async def update_garden_docs_endpoint(
    name: str, version: int, body: GardenDocsRequest, principal: PrincipalDep
):
    """Edit a garden version's docs (markdown) IN PLACE — no new version. Analyst+
    role. Docs are metadata, so this does not touch the source, manifest, or
    compatibility status. PUBLISHED versions are immutable (the store rejects the
    edit -> 409); use "Edit as new version" to change a published model's docs."""
    if not principal.has_role(Role.ANALYST):
        raise HTTPException(
            status_code=403, detail="Editing docs requires an analyst+ role."
        )
    org_id = _garden_org(principal)
    row = sessions_store.get_garden_model(
        org_id=org_id, name=name, version=int(version)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Garden model not found.")
    try:
        updated = sessions_store.upsert_garden_model(
            org_id=org_id, name=name, model_id=row["id"], docs=body.docs
        )
    except ValueError as e:  # published versions are immutable
        raise HTTPException(status_code=409, detail=str(e))
    return JSONResponse(content=safe_json_dumps_load(updated))


def _garden_test_sync(synthetic_tid: str, row: dict) -> dict:
    """One asyncio.to_thread context (per the ContextVar rule): stage the source
    into the session workspace and run the compatibility suite via the kernel —
    sandboxed in the hosted profile, so untrusted source never imports in the
    host process."""
    from mmm_framework.agents import tools as _tools
    from mmm_framework.agents.runtime import set_current_thread

    set_current_thread(synthetic_tid)
    dest = _tools._garden_copy_source_to_session(row, synthetic_tid)
    return _tools._KERNELS.get_or_spawn(synthetic_tid).run_model_op(
        "garden_compat",
        {
            "source_path": dest,
            "class_name": (row.get("manifest") or {}).get("class_name"),
        },
    )


async def _run_garden_test_job(job_id: str, synthetic_tid: str, model_id: str) -> None:
    try:
        _sim_job_patch(job_id, status="running")
        row = sessions_store.get_garden_model(model_id=model_id)
        if not row:
            _sim_job_patch(job_id, status="error", error="garden model not found")
            return
        res = await asyncio.to_thread(_garden_test_sync, synthetic_tid, row)
        if res.get("error"):
            _sim_job_patch(job_id, status="error", error=res["error"])
            return
        report = res.get("compat_report") or (res.get("dashboard") or {}).get(
            "garden_compat"
        )
        if report:
            sessions_store.set_garden_compat_report(model_id, report)
        promoted = False
        if report and report.get("blocking_passed") and row["status"] == "draft":
            try:
                sessions_store.transition_garden_model(model_id, "tested")
                promoted = True
            except ValueError:
                pass
        _sim_job_patch(
            job_id,
            status="done",
            result=safe_json_dumps_load(
                {
                    "blocking_passed": bool(report and report.get("blocking_passed")),
                    "score": (report or {}).get("score"),
                    "promoted": promoted,
                    "summary": (report or {}).get("summary"),
                    "tiers": (report or {}).get("tiers"),
                }
            ),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("garden test job failed: %s", job_id)
        _sim_job_patch(job_id, status="error", error=str(e))


@app.post("/model-garden/{name}/{version}/test", dependencies=[_rl_heavy])
async def start_garden_test(name: str, version: int, principal: PrincipalDep):
    """Start a NON-BLOCKING compatibility test; on pass the model is promoted
    draft→tested. Returns a job_id to poll."""
    org_id = _garden_org(principal)
    row = sessions_store.get_garden_model(
        org_id=org_id, name=name, version=int(version)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Garden model not found.")
    synthetic_tid = f"__gardenjobs__{org_id}"
    job = sessions_store.add_artifact(
        synthetic_tid,
        "garden_test_job",
        {
            "status": "pending",
            "org_id": org_id,
            "model_id": row["id"],
            "name": name,
            "version": int(version),
            "result": None,
            "error": None,
        },
    )
    _spawn_job_task(_run_garden_test_job(job["id"], synthetic_tid, row["id"]))
    return JSONResponse(
        status_code=202, content={"job_id": job["id"], "status": "pending"}
    )


@app.get("/model-garden/{name}/{version}/test/{job_id}")
async def get_garden_test(
    name: str, version: int, job_id: str, principal: PrincipalDep
):
    """Poll a garden test job: {status, result|null, error|null}."""
    art = sessions_store.get_artifact(job_id)
    org_id = _garden_org(principal)
    if art is None or (art.get("payload") or {}).get("org_id") != org_id:
        raise HTTPException(status_code=404, detail="Test job not found.")
    return JSONResponse(content=safe_json_dumps_load(art["payload"]))


@app.post("/model-garden/{name}/{version}/promote")
async def promote_garden_model(
    name: str, version: int, body: GardenPromoteRequest, principal: PrincipalDep
):
    """Human publish gate: promote a TESTED model to PUBLISHED so every project
    in the org can load it. Org admin/owner only; the model must be `tested`."""
    if not principal.has_role(Role.ADMIN):
        raise HTTPException(
            status_code=403, detail="Publishing requires an org admin/owner role."
        )
    org_id = _garden_org(principal)
    row = sessions_store.get_garden_model(
        org_id=org_id, name=name, version=int(version)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Garden model not found.")
    try:
        updated = sessions_store.transition_garden_model(
            row["id"], "published", note=body.note or None
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return JSONResponse(content=safe_json_dumps_load(updated))


@app.delete("/model-garden/{name}/{version}")
async def delete_garden_model_endpoint(
    name: str, version: int, principal: PrincipalDep
):
    """Delete a draft/deprecated garden model (org admin/owner only). Published
    history is immutable — deprecate instead."""
    if not principal.has_role(Role.ADMIN):
        raise HTTPException(
            status_code=403, detail="Deleting requires an org admin/owner role."
        )
    org_id = _garden_org(principal)
    row = sessions_store.get_garden_model(
        org_id=org_id, name=name, version=int(version)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Garden model not found.")
    try:
        ok = sessions_store.delete_garden_model(row["id"])
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return JSONResponse(content={"deleted": bool(ok)})


class ExperimentOptimizeRequest(BaseModel):
    channel: str
    margin: float | None = None
    price: float | None = None
    kpi_kind: str = "revenue"
    # design-space ranges (the optimizer auto-samples a few points within each)
    duration_min: int = 4
    duration_max: int = 12
    intensity_min: float = 50.0
    intensity_max: float = 100.0
    include_holdout: bool = True
    # explicit overrides for the auto-sampling (optional)
    durations: list[int] | None = None
    scaling_intensities: list[float] | None = None
    max_draws: int = 80
    seed: int = 42


@app.post(
    "/projects/{project_id}/experiment-design/optimize",
    dependencies=[_proj_write, _rl_heavy],
)
async def start_experiment_optimization(
    project_id: str, body: ExperimentOptimizeRequest
):
    """Start a NON-BLOCKING experiment-setup optimization: explores the design
    grid and returns the Pareto front (MDE × short-term cost × duration) + a
    recommended setup with cool-down. Poll the returned job_id."""
    from mmm_framework.api.history import latest_model_run_payload

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    dataset_path, kpi = _design_inputs(project_id, body.channel)
    run = latest_model_run_payload(project_id)
    margin, price = _resolve_project_margin(project_id, body.margin, body.price)

    op_kwargs: dict = {
        "dataset_path": dataset_path,
        "kpi": kpi,
        "channel": body.channel,
        "margin": margin,
        "price": price,
        "kpi_kind": body.kpi_kind,
        "duration_min": int(body.duration_min),
        "duration_max": int(body.duration_max),
        "intensity_min": float(body.intensity_min),
        "intensity_max": float(body.intensity_max),
        "include_holdout": bool(body.include_holdout),
        "max_draws": int(body.max_draws),
        "random_seed": int(body.seed),
    }
    if body.durations:
        op_kwargs["durations"] = [int(d) for d in body.durations]
    if body.scaling_intensities:
        op_kwargs["scaling_intensities"] = [float(x) for x in body.scaling_intensities]

    synthetic_tid = f"__simjobs__{project_id}"
    job = sessions_store.add_artifact(
        synthetic_tid,
        "experiment_optimization",
        {
            "status": "pending",
            "project_id": project_id,
            "channel": body.channel,
            "result": None,
            "error": None,
        },
    )
    job_id = job["id"]
    _spawn_job_task(
        _run_model_op_job(
            job_id,
            synthetic_tid,
            run,
            "experiment_optimizer",
            op_kwargs,
            "experiment_optimization",
        )
    )
    return JSONResponse(
        status_code=202, content={"job_id": job_id, "status": "pending"}
    )


@app.get(
    "/projects/{project_id}/experiment-design/optimize/{job_id}",
    dependencies=[_proj_read],
)
async def get_experiment_optimization(project_id: str, job_id: str):
    """Poll an experiment-optimization job: {status, result|null, error|null}."""
    art = sessions_store.get_artifact(job_id)
    if art is None or (art.get("payload") or {}).get("project_id") != project_id:
        raise HTTPException(status_code=404, detail="Optimization job not found.")
    return JSONResponse(content=safe_json_dumps_load(art["payload"]))


# ── Artifacts ─────────────────────────────────────────────────────────────────


@app.get("/artifacts/{thread_id}", dependencies=[_sess_read])
async def list_artifacts_endpoint(thread_id: str):
    return JSONResponse(content=sessions_store.list_artifacts(thread_id))


@app.delete("/artifacts/{artifact_id}")
async def delete_artifact_endpoint(
    artifact_id: str, principal: PrincipalDep = _DEV_PRINCIPAL
):
    art = sessions_store.get_artifact(artifact_id)
    if art is not None:
        _ensure_session_access(principal, art.get("thread_id"), Role.ANALYST)
    sessions_store.delete_artifact(artifact_id)
    return JSONResponse(content={"status": "ok"})


@app.get("/sessions/{thread_id}/export", dependencies=[_sess_read])
async def export_session_endpoint(thread_id: str, format: str = "py"):
    """Download the session's Python work as a standalone, runnable script.

    Synthesizes a preamble that reconstitutes tool-injected state (dataset,
    fitted model, helpers), then the execute_python cells in order — so the
    download is a real, portable reproduction of the session, not a code dump
    that NameErrors. See ``agents/session_export.build_session_script``.
    """
    import re

    from mmm_framework.agents.session_export import build_session_script

    script = build_session_script(thread_id)
    sess = sessions_store.get_session(thread_id) or {}
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sess.get("name") or thread_id)).strip(
        "_"
    )
    filename = f"{slug or 'session'}.py"
    return Response(
        content=script,
        media_type="text/x-python; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Assumptions log ──────────────────────────────────────────────────────────


class RecordAssumptionBody(BaseModel):
    key: str
    value: Any
    rationale: str
    category: str = "other"
    change_note: str | None = None


@app.get("/assumptions/{thread_id}", dependencies=[_sess_read])
async def list_assumptions_endpoint(thread_id: str, history: bool = False):
    return JSONResponse(
        content=sessions_store.list_assumptions(thread_id, include_history=history)
    )


@app.get("/assumption_history/{thread_id}/{key:path}", dependencies=[_sess_read])
async def get_assumption_history_endpoint(thread_id: str, key: str):
    """Distinct route prefix so {key:path} can't shadow the list endpoint."""
    history = sessions_store.get_assumption_history(thread_id, key)
    if not history:
        raise HTTPException(status_code=404, detail="assumption not found")
    return JSONResponse(content=history)


@app.post("/assumptions/{thread_id}", dependencies=[_sess_write])
async def record_assumption_endpoint(thread_id: str, body: RecordAssumptionBody):
    rec = sessions_store.record_assumption(
        thread_id=thread_id,
        key=body.key,
        value=body.value,
        rationale=body.rationale,
        category=body.category,
        change_note=body.change_note,
    )
    return JSONResponse(content=rec)


class RetractAssumptionBody(BaseModel):
    reason: str


@app.delete("/assumption/{thread_id}/{key:path}", dependencies=[_sess_write])
async def retract_assumption_endpoint(
    thread_id: str, key: str, body: RetractAssumptionBody
):
    rec = sessions_store.retract_assumption(thread_id, key, body.reason)
    if not rec:
        raise HTTPException(status_code=404, detail="assumption not found")
    return JSONResponse(content=rec)


# ── Workflow status ──────────────────────────────────────────────────────────


async def _infer_workflow_status(thread_id: str) -> list[dict[str, Any]]:
    """Derive 9-step workflow status from session state + overrides.

    The agent shouldn't be the source of truth — we infer from concrete state
    changes:
      1: research_question assumption exists
      2: dag_structure assumption exists OR dataset inspected
      3: model_spec set in dashboard
      4: prior_predictive_check assumption exists
      5: model_status == 'completed'
      6: diagnostics dashboard key exists
      7: posterior predictive evidence (decomposition computed)
      8: any sensitivity::* assumption OR override
      9: roi_metrics computed
    """
    state_values: dict[str, Any] = {}
    try:
        g = _admin_graph()
        snap = await g.aget_state({"configurable": {"thread_id": thread_id}})
        if snap and snap.values:
            state_values = snap.values
    except Exception:
        state_values = {}
    dashboard = state_values.get("dashboard_data") or {}
    assumptions = {a["key"]: a for a in sessions_store.list_assumptions(thread_id)}
    overrides = sessions_store.get_workflow_overrides(thread_id)

    has = lambda k: k in assumptions
    inferred: dict[int, str] = {i: "pending" for i in range(1, 10)}
    inferred[1] = "done" if has("research_question") else "pending"
    inferred[2] = (
        "done"
        if has("dag_structure")
        else (
            "in_progress"
            if (state_values.get("dataset_path") or dashboard.get("dataset"))
            else "pending"
        )
    )
    inferred[3] = (
        "done"
        if (state_values.get("model_spec") or dashboard.get("model_spec"))
        else "pending"
    )
    inferred[4] = "done" if has("prior_predictive_check") else "pending"
    inferred[5] = (
        "done" if state_values.get("model_status") == "completed" else "pending"
    )
    inferred[6] = "done" if dashboard.get("diagnostics") else "pending"
    inferred[7] = "done" if dashboard.get("decomposition") else "pending"
    sens_keys = [k for k in assumptions if k.startswith("sensitivity::")]
    inferred[8] = "done" if sens_keys else "pending"
    inferred[9] = "done" if dashboard.get("roi_metrics") else "pending"

    out = []
    for step, title, desc in sessions_store.WORKFLOW_STEPS:
        override = overrides.get(step)
        # Treat 'pending' override as "no override" so a manually-cleared step
        # falls back to the inferred status (e.g. 'done' if model_spec exists).
        active_override = (
            override if (override and override["status"] != "pending") else None
        )
        status = active_override["status"] if active_override else inferred[step]
        notes = active_override["notes"] if active_override else None
        out.append(
            {
                "step": step,
                "title": title,
                "description": desc,
                "status": status,
                "notes": notes,
                "inferred_status": inferred[step],
                "overridden": active_override is not None,
                "updated_at": (
                    active_override["updated_at"] if active_override else None
                ),
            }
        )
    return out


@app.get("/workflow/{thread_id}", dependencies=[_sess_read])
async def workflow_status_endpoint(thread_id: str):
    return JSONResponse(content=await _infer_workflow_status(thread_id))


class WorkflowStepBody(BaseModel):
    status: str
    notes: str | None = None


@app.patch("/workflow/{thread_id}/{step}", dependencies=[_sess_write])
async def update_workflow_step_endpoint(
    thread_id: str, step: int, body: WorkflowStepBody
):
    if step < 1 or step > 9:
        raise HTTPException(status_code=400, detail="step must be 1..9")
    rec = sessions_store.set_workflow_step(thread_id, step, body.status, body.notes)
    return JSONResponse(content=rec)


# ── Files registry ───────────────────────────────────────────────────────────


@app.get("/files/{thread_id}", dependencies=[_sess_read])
async def list_files_endpoint(thread_id: str):
    return JSONResponse(content=sessions_store.list_files(thread_id))


@app.delete("/files/{file_id}")
async def delete_file_endpoint(file_id: str, principal: PrincipalDep = _DEV_PRINCIPAL):
    rec = sessions_store.get_file(file_id)
    if rec is not None:
        _ensure_session_access(principal, rec.get("thread_id"), Role.ANALYST)
    sessions_store.delete_file(file_id)
    return JSONResponse(content={"status": "ok"})


# ── Generated-file downloads (req 4) ──────────────────────────────────────────


def _safe_open_within(path: str) -> "tuple[int, int]":
    """Open ``path`` read-only for serving, TOCTOU-safe (Phase 3 PR-E.2).

    The kernel can write into the workspace, so a file we resolved a moment ago
    could be swapped for a symlink to ``/etc/passwd`` before we open it. Guard by
    (1) validating the *resolved* path is inside an allowed root, (2) opening the
    realpath with ``O_NOFOLLOW`` so a symlinked final component is rejected, and
    (3) confirming the opened fd is a regular file. Returns ``(fd, size)``;
    raises ``HTTPException`` (403 outside roots, 404 missing/non-regular). The
    caller owns the fd. (The narrow parent-dir-swap race is closed for real by
    the Tier-2 read-only mount namespace; this is defense-in-depth today.)"""
    import stat as _stat

    from mmm_framework.agents import workspace as _ws

    if not path:
        raise HTTPException(status_code=404, detail="File not found")
    # is_within() resolves symlinks, so an out-of-roots target is refused here.
    if not _ws.is_within(path):
        raise HTTPException(status_code=403, detail="File is outside the allowed roots")
    realpath = os.path.realpath(path)
    if not _ws.is_within(realpath):
        raise HTTPException(status_code=403, detail="File is outside the allowed roots")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(realpath, flags)
    except OSError:  # symlinked final component (O_NOFOLLOW) or vanished
        raise HTTPException(status_code=404, detail="File not found")
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        os.close(fd)
        raise
    except OSError:
        os.close(fd)
        raise HTTPException(status_code=404, detail="File not found")
    return fd, st.st_size


def _iter_fd(fd: int, chunk: int = 64 * 1024):
    """Stream an open fd in chunks, closing it when exhausted — so the response
    is served from the exact fd we validated, never re-opened."""
    f = os.fdopen(fd, "rb")
    try:
        while True:
            data = f.read(chunk)
            if not data:
                break
            yield data
    finally:
        f.close()


def _safe_serve(
    path: str,
    media_type: str,
    *,
    download_name: str | None = None,
    headers: "dict[str, str] | None" = None,
) -> StreamingResponse:
    """TOCTOU-safe serve of a file inside an allowed root. ``download_name`` sets
    an attachment Content-Disposition (else the body renders inline)."""
    fd, size = _safe_open_within(path)
    hdrs = {"Content-Length": str(size), **(headers or {})}
    if download_name:
        safe_name = os.path.basename(download_name).replace('"', "")
        hdrs["Content-Disposition"] = f'attachment; filename="{safe_name}"'
    return StreamingResponse(_iter_fd(fd), media_type=media_type, headers=hdrs)


def _safe_upload_name(filename: str | None, default: str) -> str:
    """A traversal-safe filename for an upload: the basename only (no path
    separators) and never ``.``/``..`` — so a crafted ``filename`` can't steer
    where the bytes land (Phase 3 PR-E.2)."""
    name = os.path.basename(filename or "")
    return default if (not name or name in (".", "..")) else name


def _guarded_file_response(path: str, filename: str | None = None) -> StreamingResponse:
    """Attachment download of ``path``, only if it sits inside an allowed root
    (workspace / uploads / mmm_models / mmm_configs / CWD) — blocks traversal and
    symlink-swap (TOCTOU)."""
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return _safe_serve(
        path,
        "application/octet-stream",
        download_name=filename or os.path.basename(path),
    )


@app.get("/plots/{plot_id}")
async def get_plot_endpoint(plot_id: str):
    """Serve a content-addressed Plotly figure JSON. Because the id is a content
    hash, the response is immutable — the browser caches it permanently, so each
    plot crosses the wire at most once per client (instead of the full plot list
    being re-streamed every turn)."""
    from mmm_framework.agents import workspace as _ws

    path = _ws.plot_path(plot_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Plot not found: {plot_id}")
    return FileResponse(
        str(path),
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/tables/{table_id}")
async def get_table_endpoint(table_id: str):
    """Serve a content-addressed structured table payload (same immutable-cache
    contract as /plots/{id} — refs stream, rows are fetched once)."""
    from mmm_framework.agents import workspace as _ws

    path = _ws.table_path(table_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Table not found: {table_id}")
    return FileResponse(
        str(path),
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/workspace/{thread_id}/files", dependencies=[_sess_read])
async def workspace_files_endpoint(thread_id: str):
    """List the session's registered files (uploads + generated), each with a
    download id."""
    files = sessions_store.list_files(thread_id)
    return JSONResponse(content={"files": files, "total": len(files)})


@app.get("/files/{file_id}/download")
async def download_file_endpoint(
    file_id: str, principal: PrincipalDep = _DEV_PRINCIPAL
):
    rec = sessions_store.get_file(file_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    _ensure_session_access(principal, rec.get("thread_id"), Role.VIEWER)
    return _guarded_file_response(rec["path"], rec.get("name"))


@app.get("/artifacts/{artifact_id}/download")
async def download_artifact_endpoint(
    artifact_id: str, principal: PrincipalDep = _DEV_PRINCIPAL
):
    art = sessions_store.get_artifact(artifact_id)
    if art is None:
        raise HTTPException(
            status_code=404, detail=f"Artifact not found: {artifact_id}"
        )
    _ensure_session_access(principal, art.get("thread_id"), Role.VIEWER)
    p = art.get("payload", {})
    # Resolve a downloadable path from whatever kind of artifact this is
    path = p.get("path") or p.get("report_path") or p.get("model_path")
    if not path:
        raise HTTPException(
            status_code=404, detail="This artifact has no downloadable file"
        )
    if os.path.isdir(path):  # model_run model_path is a directory — zip it
        import tempfile
        from mmm_framework.agents import workspace as _ws

        if not _ws.is_within(path):
            raise HTTPException(status_code=403, detail="Path outside allowed roots")
        base = os.path.join(tempfile.gettempdir(), f"artifact_{artifact_id}")
        archive = shutil.make_archive(base, "zip", path)
        return FileResponse(
            archive,
            filename=f"{os.path.basename(path)}.zip",
            media_type="application/zip",
        )
    return _guarded_file_response(path)


# ── Knowledge base (req 2/3 — project-level RAG) ─────────────────────────────


@app.post("/projects/{project_id}/kb", dependencies=[_proj_write, _rl_heavy])
async def kb_upload_endpoint(
    project_id: str,
    file: UploadFile = File(...),
    template: bool = Form(False),
):
    """Add a document to a project's knowledge base: store it, then chunk +
    embed it (in a threadpool) so it becomes searchable. Pass template=true to
    tag it as a template document (surfaced by the agent's list_templates)."""
    from fastapi.concurrency import run_in_threadpool
    from mmm_framework.agents import workspace as _ws
    from mmm_framework.agents import knowledge_base as kb

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    kb_dir = _ws.project_kb_dir(project_id)
    # Flatten to a safe basename + safe_join so a crafted filename ("../../x")
    # can't escape the project KB dir (Phase 3 PR-E.2).
    name = _safe_upload_name(file.filename, "document.txt")
    dest = str(_ws.safe_join(kb_dir, name))
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    size_bytes = os.path.getsize(dest)
    kind = kb.kind_for(name)

    doc = sessions_store.add_kb_document(
        project_id=project_id,
        name=name,
        path=dest,
        kind=kind,
        size_bytes=size_bytes,
        status="pending",
        meta={"content_type": file.content_type, "template": bool(template)},
    )
    # Ingest synchronously-in-threadpool so the response reflects final status.
    doc = await run_in_threadpool(kb.ingest_document, doc["id"])
    return JSONResponse(content=doc)


@app.get("/projects/{project_id}/kb", dependencies=[_proj_read])
async def kb_list_endpoint(project_id: str):
    docs = sessions_store.list_kb_documents(project_id)
    return JSONResponse(content={"documents": docs, "total": len(docs)})


@app.get("/projects/{project_id}/kb/search", dependencies=[_proj_read])
async def kb_search_endpoint(project_id: str, q: str, k: int = 6):
    from fastapi.concurrency import run_in_threadpool
    from mmm_framework.agents import knowledge_base as kb

    if not q or not q.strip():
        return JSONResponse(content={"results": []})
    try:
        results = await run_in_threadpool(kb.search, project_id, q, k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")
    return JSONResponse(content={"results": results})


@app.delete("/kb/{document_id}")
async def kb_delete_endpoint(
    document_id: str, principal: PrincipalDep = _DEV_PRINCIPAL
):
    doc = sessions_store.get_kb_document(document_id)
    if doc is None:
        raise HTTPException(
            status_code=404, detail=f"Document not found: {document_id}"
        )
    if not principal.is_dev:
        ensure_project_access(principal, doc.get("project_id"), Role.ANALYST)
    sessions_store.delete_kb_document(document_id)
    # best-effort remove the file on disk
    try:
        if doc.get("path") and os.path.isfile(doc["path"]):
            os.remove(doc["path"])
    except OSError:
        pass
    return JSONResponse(content={"success": True})


# ── DAG endpoint (read current DAG from state, return React Flow JSON) ───────


@app.get("/dag/{thread_id}", dependencies=[_sess_read])
async def dag_endpoint(thread_id: str):
    """Return the current DAG (if any) plus a fresh identifiability report.
    Pulls DAG from dashboard_data['dag']['spec'] which `propose_dag` writes.
    """
    try:
        agent_graph = _admin_graph()
        snap = await agent_graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snap or not snap.values:
            return JSONResponse(content={"dag": None})
        dashboard = snap.values.get("dashboard_data") or {}
        dag_payload = dashboard.get("dag")
        if not dag_payload:
            return JSONResponse(content={"dag": None})
        return JSONResponse(content=dag_payload)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class DAGUpdateRequest(BaseModel):
    nodes: list[dict]
    edges: list[dict]


@app.put("/dag/{thread_id}", dependencies=[_sess_write])
async def update_dag(thread_id: str, body: DAGUpdateRequest):
    """Manually update the DAG for a session.

    Accepts React Flow node/edge format, converts to DAGSpec, validates,
    and persists into the agent state dashboard_data.dag.
    """
    from mmm_framework.dag_model_builder.frontend_adapter import (
        react_flow_to_dag_spec,
        dag_spec_to_react_flow,
    )
    from mmm_framework.dag_model_builder.validation import validate_dag
    from mmm_framework.agents.causal_tools import validate_causal_identification

    try:
        spec = react_flow_to_dag_spec(body.nodes, body.edges)
        validation = validate_dag(spec)
        react_flow = dag_spec_to_react_flow(spec)

        dag_payload: dict = {
            "spec": spec.model_dump(mode="json"),
            "react_flow": react_flow,
            "validation": {
                "valid": validation.valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            },
        }

        config = {"configurable": {"thread_id": thread_id}}
        agent_graph = _admin_graph()
        snap = await agent_graph.aget_state(config)
        dashboard = (
            (snap.values.get("dashboard_data") or {}) if snap and snap.values else {}
        )
        dashboard["dag"] = dag_payload
        await agent_graph.aupdate_state(config, {"dashboard_data": dashboard})

        return JSONResponse(content=dag_payload)
    except Exception as e:
        logger.exception("DAG update failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


class OutlierApplyRequest(BaseModel):
    action_ids: list[str]
    reason: str | None = None


@app.post("/outliers/{thread_id}/apply", dependencies=[_sess_write])
async def apply_outliers_endpoint(thread_id: str, body: OutlierApplyRequest):
    """Apply confirmed outlier-treatment actions from the UI (EDA tab confirm
    buttons), without a chat round-trip.

    Applies a STATE-ONLY update via aupdate_state (same pattern as PUT /dag):
    no ToolMessage/AIMessage is appended — an orphan tool message injected
    outside a real tool call corrupts Anthropic message threads."""
    from fastapi.concurrency import run_in_threadpool

    from mmm_framework.agents.eda_tools import _apply_outlier_treatment_core

    try:
        config = {"configurable": {"thread_id": thread_id}}
        agent_graph = _admin_graph()
        snap = await agent_graph.aget_state(config)
        values = (snap.values or {}) if snap else {}
        error, summary, update = await run_in_threadpool(
            _apply_outlier_treatment_core,
            values,
            thread_id,
            list(body.action_ids or []),
            body.reason,
        )
        if error:
            return JSONResponse(status_code=400, content={"error": error})
        if update:
            await agent_graph.aupdate_state(config, update)
        return JSONResponse(
            content={
                "summary": summary,
                "applied": list(body.action_ids or []),
                "dataset_path": update.get("dataset_path"),
                "eda": (update.get("dashboard_data") or {}).get("eda"),
            }
        )
    except Exception as e:
        logger.exception("Outlier apply failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Model spec: manual edits + lock confirmation ──────────────────────────────


class SpecUpdateRequest(BaseModel):
    model_spec: dict
    # Explicit leaf paths the user changed (computed client-side against the
    # defaulted baseline the editor showed). When provided these are locked
    # verbatim; otherwise the server falls back to diffing the full spec, which
    # over-locks materialized defaults — so the client always sends them.
    lock_paths: list[str] | None = None
    unlock_paths: list[str] | None = None


class SpecResolveRequest(BaseModel):
    path: str
    action: str  # "approve" | "reject"


def _mirror_spec_dashboard(
    dashboard: dict, spec: dict, locked: list[str], pending: list[dict]
) -> dict:
    dashboard = dict(dashboard or {})
    dashboard["model_spec"] = spec
    dashboard["locked_fields"] = locked
    dashboard["pending_spec_changes"] = pending
    return dashboard


@app.patch("/spec/{thread_id}", dependencies=[_sess_write])
async def update_spec(thread_id: str, body: SpecUpdateRequest):
    """Server-authoritative manual edit of the model configuration.

    Writes the edited ``model_spec`` directly into the agent state and locks the
    leaf fields the user actually changed (diffed server-side) so the LLM can no
    longer silently overwrite them. ``unlock_paths`` hands fields back to the LLM.
    """
    from mmm_framework.agents.spec_locks import diff_locked

    config = {"configurable": {"thread_id": thread_id}}
    try:
        agent_graph = _admin_graph()
        snap = await agent_graph.aget_state(config)
        values = (snap.values or {}) if snap else {}

        current_spec = values.get("model_spec") or {}
        new_spec = body.model_spec or {}

        # Prefer the client's precise touched-path list; fall back to a server
        # diff only when it isn't supplied (e.g. a programmatic caller).
        if body.lock_paths is not None:
            newly_locked = list(body.lock_paths)
        else:
            newly_locked = diff_locked(current_spec, new_spec)
        unlock = set(body.unlock_paths or [])
        locked = [p for p in (values.get("locked_fields") or []) if p not in unlock]
        for p in newly_locked:
            if p not in locked:
                locked.append(p)

        # Drop any stale pending proposals for fields the user just decided.
        decided = set(newly_locked) | unlock
        pending = [
            p
            for p in (values.get("pending_spec_changes") or [])
            if p.get("path") not in decided
        ]

        dashboard = _mirror_spec_dashboard(
            values.get("dashboard_data") or {}, new_spec, locked, pending
        )

        update = {
            "model_spec": new_spec,
            "locked_fields": locked,
            "pending_spec_changes": pending,
            "dashboard_data": dashboard,
        }
        status = values.get("model_status")
        if new_spec.get("kpi") and status in (None, "", "unconfigured"):
            update["model_status"] = "configured"

        # Attribute the write to the agent node so the update is unambiguous on
        # the two-node graph; with no tool-call message appended it routes to END.
        await agent_graph.aupdate_state(config, update, as_node="agent")
        return JSONResponse(
            content=safe_json_dumps_load(
                {
                    "model_spec": new_spec,
                    "locked_fields": locked,
                    "pending_spec_changes": pending,
                }
            )
        )
    except Exception as e:
        logger.exception("Spec update failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/spec/{thread_id}/resolve", dependencies=[_sess_write])
async def resolve_spec_change(thread_id: str, body: SpecResolveRequest):
    """Confirm or decline an LLM-proposed change to a user-locked field.

    ``approve`` applies the proposed value (and keeps the field locked at the new
    value). ``reject`` keeps the user's value and writes a note into the thread so
    the LLM has decline-memory and won't re-propose the same change next turn.
    """
    from mmm_framework.agents.spec_locks import set_at, get_at

    config = {"configurable": {"thread_id": thread_id}}
    action = (body.action or "").lower()
    if action not in ("approve", "reject"):
        return JSONResponse(
            status_code=400, content={"error": "action must be approve or reject"}
        )
    try:
        agent_graph = _admin_graph()
        snap = await agent_graph.aget_state(config)
        values = (snap.values or {}) if snap else {}

        pending = list(values.get("pending_spec_changes") or [])
        entry = next((p for p in pending if p.get("path") == body.path), None)
        if entry is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"no pending change for '{body.path}'"},
            )

        spec = copy.deepcopy(values.get("model_spec") or {})
        locked = list(values.get("locked_fields") or [])
        remaining = [p for p in pending if p.get("path") != body.path]
        path = entry["path"]

        if action == "approve":
            set_at(spec, path, entry["proposed"])
            if path not in locked:
                locked.append(path)
            note = (
                f"[system] The user APPROVED changing `{path}` "
                f"from `{entry.get('current')}` to `{entry.get('proposed')}`. "
                "It is now applied and remains user-locked at the new value."
            )
        else:  # reject
            note = (
                f"[system] The user DECLINED changing `{path}` "
                f"(it stays `{get_at(spec, path)}`; you had proposed "
                f"`{entry.get('proposed')}`). Do not propose this change again "
                "unless the user explicitly asks."
            )

        dashboard = _mirror_spec_dashboard(
            values.get("dashboard_data") or {}, spec, locked, remaining
        )
        await agent_graph.aupdate_state(
            config,
            {
                "model_spec": spec,
                "locked_fields": locked,
                "pending_spec_changes": remaining,
                "dashboard_data": dashboard,
                "messages": [HumanMessage(content=note)],
            },
            # Append the decline/approve note as the agent node; the trailing
            # HumanMessage has no tool_calls so the graph settles at END.
            as_node="agent",
        )
        return JSONResponse(
            content=safe_json_dumps_load(
                {
                    "action": action,
                    "model_spec": spec,
                    "locked_fields": locked,
                    "pending_spec_changes": remaining,
                }
            )
        )
    except Exception as e:
        logger.exception("Spec resolve failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Dataset preview ─────────────────────────────────────────────────────────


@app.get("/dataset/preview/{thread_id}", dependencies=[_sess_read])
async def dataset_preview(
    thread_id: str,
    variable: str,
    dim: str | None = None,
    value: str | None = None,
):
    """Return a time-series for one MFF variable, optionally filtered by one dimension.

    Query params:
      variable  — value of VariableName (e.g. 'TV', 'Sales')
      dim       — dimension column to filter on (e.g. 'Geography')
      value     — the dimension value; pass '(national)' to filter NaN rows
    """
    try:
        import pandas as pd

        agent_graph = _admin_graph()
        snap = await agent_graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snap or not snap.values:
            raise HTTPException(status_code=404, detail="no state found for thread")

        ds_path = snap.values.get("dataset_path")
        if not ds_path or not os.path.exists(ds_path):
            raise HTTPException(status_code=404, detail="dataset not found")

        df = pd.read_csv(ds_path)

        if "VariableName" not in df.columns:
            raise HTTPException(
                status_code=400, detail="not an MFF dataset (no VariableName column)"
            )

        sub = df[df["VariableName"] == variable].copy()
        if sub.empty:
            raise HTTPException(
                status_code=404, detail=f"variable '{variable}' not found"
            )

        if dim and value:
            if dim not in df.columns:
                raise HTTPException(
                    status_code=400, detail=f"dimension '{dim}' not found"
                )
            if value == "(national)":
                sub = sub[sub[dim].isna()]
            else:
                sub = sub[sub[dim] == value]

        date_cols = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ("date", "week", "period", "time"))
        ]
        if not date_cols:
            raise HTTPException(
                status_code=400, detail="no date column found in dataset"
            )
        date_col = date_cols[0]

        series = (
            sub.groupby(date_col)["VariableValue"]
            .sum()
            .reset_index()
            .rename(columns={date_col: "date", "VariableValue": "value"})
            .sort_values("date")
        )

        return JSONResponse(
            content={
                "variable": variable,
                "dim": dim,
                "value": value,
                "series": series.to_dict(orient="records"),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("dataset_preview failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Misc ──────────────────────────────────────────────────────────────────────


@app.post("/upload", dependencies=[_sess_write, _rl_heavy])
async def upload_file(file: UploadFile = File(...), thread_id: str | None = None):
    """Upload a file scoped to a session. Files land in the session WORKSPACE
    directory (so execute_python and the Files tab share one location), are
    registered in data_files, and an ABSOLUTE path is returned so the agent can
    read it from any tool (execute_python runs in the workspace; fit/inspect run
    in the server cwd)."""
    from mmm_framework.agents import workspace as _ws

    tid = thread_id or "_unscoped"
    if thread_id:
        upload_dir = str(_ws.thread_dir(thread_id))
    else:
        upload_dir = os.path.join("uploads", tid)
        os.makedirs(upload_dir, exist_ok=True)
    # Flatten to a safe basename + safe_join so a crafted filename can't escape
    # the upload dir (Phase 3 PR-E.2). safe_join resolves + returns an abs path.
    name = _safe_upload_name(file.filename, "upload.bin")
    file_location = str(_ws.safe_join(Path(upload_dir), name))
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    size_bytes = os.path.getsize(file_location)
    preview: str | None = None
    kind = "upload"
    if name.lower().endswith((".csv", ".tsv", ".txt")):
        try:
            with open(file_location, "r", errors="ignore") as f:
                head = [next(f) for _ in range(6)]
            preview = "".join(head)
            kind = "dataset"
        except Exception:
            preview = None
    elif name.lower().endswith((".xlsx", ".xls")):
        kind = "dataset"

    if thread_id:
        sessions_store.register_file(
            thread_id=thread_id,
            path=file_location,
            name=name,
            kind=kind,
            size_bytes=size_bytes,
            preview=preview,
            meta={"content_type": file.content_type},
        )

    return {
        "filename": name,
        "path": file_location,
        "size_bytes": size_bytes,
        "kind": kind,
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/observability")
async def observability_endpoint():
    """Reliability signals for operators: audit-chain integrity, off-host ship
    backlog, and recent fit activity. No tenant data."""
    from mmm_framework.api.observability import system_health

    return JSONResponse(content=system_health())


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics (Phase 4d): per-event audit counters, live kernel count,
    and the active-fit gauge the autoscaler scales on (§5.1). Sourced from the
    mmm_audit events so the audit log is the single source of truth."""
    from mmm_framework.agents.audit_sink import event_counts
    from mmm_framework.agents.tools import _KERNELS

    counts = event_counts()
    live = len(getattr(_KERNELS, "_instances", {}))
    active_fits = max(
        0, counts.get("kernel_fit_start", 0) - counts.get("kernel_fit_done", 0)
    )
    lines = [
        "# HELP mmm_audit_events_total Count of mmm_audit events by type.",
        "# TYPE mmm_audit_events_total counter",
    ]
    for ev, c in sorted(counts.items()):
        ev_safe = ev.replace("\\", "").replace('"', "")
        lines.append(f'mmm_audit_events_total{{event="{ev_safe}"}} {c}')
    lines += [
        "# HELP mmm_kernels_live Currently live per-session kernels.",
        "# TYPE mmm_kernels_live gauge",
        f"mmm_kernels_live {live}",
        "# HELP mmm_active_fits In-flight model fits (autoscaling signal).",
        "# TYPE mmm_active_fits gauge",
        f"mmm_active_fits {active_fits}",
    ]
    return Response(
        "\n".join(lines) + "\n", media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.get("/integrations/catalog")
async def integrations_catalog_endpoint(principal: PrincipalDep = _DEV_PRINCIPAL):
    """List available data-source + ad-platform integrations (non-secret).

    Powers the Settings "Data connections" section: which connectors exist,
    whether their optional SDK is installed, the auth story, and (for ad
    platforms) the recommended ingestion path. No credentials are returned.

    Requires an authenticated principal: the installed-SDK flags reveal
    deployment topology, so the endpoint is gated like its neighbours rather
    than left open for reconnaissance.
    """
    from mmm_framework.integrations import list_data_sources
    from mmm_framework.integrations.ad_platforms import list_ad_platforms

    return {
        "data_sources": list_data_sources(),
        "ad_platforms": list_ad_platforms(),
    }


@app.get("/model-config")
async def model_config_endpoint():
    """Report the active LLM provider/model (non-secret).

    Lets the frontend decide whether to prompt for an API key: when the server
    authenticates via Vertex AI / ADC (or a server-side env key),
    ``requires_api_key`` is False and the login key prompt can be skipped. Never
    returns the API key or credentials contents.
    """
    from mmm_framework.agents.llm import describe_active_config

    try:
        return describe_active_config()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to describe model config")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# Small in-process TTL cache: Vertex model discovery is a network call.
_vertex_models_cache: dict[tuple, tuple[float, list]] = {}
_VERTEX_MODELS_TTL = 300.0  # seconds


@app.get("/vertex-models")
async def vertex_models_endpoint(
    project: str | None = None, location: str | None = None
):
    """List selectable Vertex models for the configured (or given) project/region.

    Combines live-discovered Gemini models, a best-effort Claude catalog, and any
    configured ``extra_models``. Results are cached briefly. The frontend should
    still offer a free-text field for ids not in the list (e.g. versioned Claude
    ids pasted from the Vertex console).
    """
    import time

    from mmm_framework.agents.llm import list_vertex_models, load_model_config

    try:
        cfg = load_model_config()
        proj = project or cfg.project
        loc = location or cfg.location
        key = (proj, loc)
        now = time.monotonic()
        cached = _vertex_models_cache.get(key)
        if cached and (now - cached[0]) < _VERTEX_MODELS_TTL:
            models = cached[1]
        else:
            # Run the (blocking) discovery off the event loop.
            models = await asyncio.to_thread(
                list_vertex_models, cfg, project=proj, location=loc
            )
            _vertex_models_cache[key] = (now, models)
        return {
            "project": proj,
            "location": loc,
            "active_model": cfg.model,
            "models": models,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to list Vertex models")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/lmstudio-models")
async def lmstudio_models_endpoint(base_url: str | None = None):
    """List models currently loaded in LM Studio (its OpenAI-compatible
    ``/v1/models``). Returns an empty list if LM Studio isn't running, so the
    frontend can still offer free-text model entry."""
    from mmm_framework.agents.llm import (
        list_lmstudio_models,
        lmstudio_base_url,
        load_model_config,
    )

    try:
        cfg = load_model_config()
        url = base_url or lmstudio_base_url(cfg)
        models = await asyncio.to_thread(list_lmstudio_models, cfg, base_url=url)
        return {
            "base_url": url,
            "active_model": cfg.model if cfg.provider == "lmstudio" else None,
            "models": models,
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to list LM Studio models")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# Report serving. In the hosted profile the agent writes reports per-session under
# the workspace (an allowed root; CWD is dropped) — so these endpoints take an
# optional `thread_id` and resolve via workspace.report_path(); in dev they serve
# the legacy CWD files unchanged. Either way _safe_serve is TOCTOU-safe (PR-E.2).


def _serve_report(
    name: str,
    *,
    thread_id: str | None,
    missing: str,
    media_type: str = "text/html",
    download_name: str | None = None,
):
    from mmm_framework.agents import workspace as _ws

    p = str(_ws.report_path(name, thread_id))
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": missing})
    return _safe_serve(p, media_type, download_name=download_name)


@app.get("/report", dependencies=[_rep_read])
async def view_report(thread_id: str | None = None):
    """Serve the generated HTML report inline for embedding."""
    return _serve_report(
        "agent_mmm_report.html",
        thread_id=thread_id,
        missing="No report generated yet. Fit a model first.",
    )


@app.get("/report/download", dependencies=[_rep_read])
async def download_report(thread_id: str | None = None):
    """Download the generated HTML report."""
    return _serve_report(
        "agent_mmm_report.html",
        thread_id=thread_id,
        missing="No report generated yet.",
        media_type="application/octet-stream",
        download_name="mmm_report.html",
    )


@app.get("/project-report", dependencies=[_rep_read])
async def view_project_report(thread_id: str | None = None):
    """Serve the project findings HTML report."""
    return _serve_report(
        "agent_project_report.html",
        thread_id=thread_id,
        missing="No project report yet. Ask the agent to generate_project_report.",
    )


@app.get("/project-report/download", dependencies=[_rep_read])
async def download_project_report(thread_id: str | None = None):
    return _serve_report(
        "agent_project_report.html",
        thread_id=thread_id,
        missing="No project report yet.",
        media_type="application/octet-stream",
        download_name="mmm_project_report.html",
    )


@app.get("/project-slides", dependencies=[_rep_read])
async def view_project_slides(thread_id: str | None = None):
    """Serve the Reveal.js project slideshow."""
    return _serve_report(
        "agent_project_slides.html",
        thread_id=thread_id,
        missing="No slideshow yet. Ask the agent to generate_project_report.",
    )


@app.get("/project-slides/download", dependencies=[_rep_read])
async def download_project_slides(thread_id: str | None = None):
    return _serve_report(
        "agent_project_slides.html",
        thread_id=thread_id,
        missing="No slideshow yet.",
        media_type="application/octet-stream",
        download_name="mmm_project_slides.html",
    )


@app.get("/client-report", dependencies=[_rep_read])
async def view_client_report(thread_id: str | None = None):
    """Serve the client-ready HTML report (no diagnostics, with nav + confidentiality notice)."""
    return _serve_report(
        "agent_client_report.html",
        thread_id=thread_id,
        missing="No client report yet. Ask the agent to generate_client_report.",
    )


@app.get("/model-defense", dependencies=[_rep_read])
async def view_model_defense(thread_id: str | None = None):
    """Serve the model-defense (causal-rigor) report."""
    return _serve_report(
        "agent_model_defense.html",
        thread_id=thread_id,
        missing="No model-defense report yet. Ask the agent to generate_model_defense_report.",
    )


@app.get("/client-report/download", dependencies=[_rep_read])
async def download_client_report(thread_id: str | None = None):
    return _serve_report(
        "agent_client_report.html",
        thread_id=thread_id,
        missing="No client report yet.",
        media_type="application/octet-stream",
        download_name="mmm_client_report.html",
    )


@app.get("/client-slides", dependencies=[_rep_read])
async def view_client_slides(thread_id: str | None = None):
    """Serve the client-ready Reveal.js slideshow (no MCMC stats, with confidentiality footer)."""
    return _serve_report(
        "agent_client_slides.html",
        thread_id=thread_id,
        missing="No client slides yet. Ask the agent to generate_client_slides.",
    )


@app.get("/client-slides/download", dependencies=[_rep_read])
async def download_client_slides(thread_id: str | None = None):
    return _serve_report(
        "agent_client_slides.html",
        thread_id=thread_id,
        missing="No client slides yet.",
        media_type="application/octet-stream",
        download_name="mmm_client_slides.html",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mmm_framework.api.main:app", host="0.0.0.0", port=8000, reload=True)
