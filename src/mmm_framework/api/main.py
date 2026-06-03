import json
import math
import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import FastAPI, Request, UploadFile, File, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mmm_framework.agents.graph import create_agent_graph
from mmm_framework.api import sessions as sessions_store

logger = logging.getLogger("mmm_api")
logging.basicConfig(level=logging.INFO)


def safe_json_dumps(obj: dict) -> str:
    """JSON serializer that handles NaN/Inf, numpy scalars, and numpy arrays."""
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

    return json.dumps(obj, default=_default)


# ── Persistent checkpointer ───────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "sessions.db"
memory: AsyncSqliteSaver | None = None
_aiosqlite_conn: aiosqlite.Connection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, _aiosqlite_conn
    _aiosqlite_conn = await aiosqlite.connect(str(DB_PATH))
    memory = AsyncSqliteSaver(_aiosqlite_conn)
    await memory.setup()
    sessions_store.init_db()
    yield
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


@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: str | None = Header(None),
    x_model_name: str | None = Header(None),
    x_base_url: str | None = Header(None),
    x_provider: str | None = Header(None),
):
    config = {"configurable": {"thread_id": request.thread_id}}
    # Mark this session active so the thread-scoped model cache + workspace dir
    # resolve correctly for tools run during this request.
    from mmm_framework.agents.runtime import set_current_thread

    set_current_thread(request.thread_id)
    llm = get_llm(x_model_name, x_api_key, x_base_url, x_provider)
    agent_graph = create_agent_graph(llm, checkpointer=memory)

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

        initial_message = HumanMessage(content=request.message)

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
            elif _k == "text_output":
                captured_artifact_keys.add(f"text_output::{_p.get('call_id', '')}")

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
                            combined_dashboard.update(dd)

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

                        msg_dashboard = {
                            k: v for k, v in combined_dashboard.items() if k != "plots"
                        }
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

                    if combined_dashboard.get("plots"):
                        plots_event = {
                            "type": "dashboard_update",
                            "dashboard_data": {"plots": combined_dashboard["plots"]},
                        }
                        try:
                            yield f"data: {safe_json_dumps(plots_event)}\n\n"
                            logger.info(
                                f"Sent {len(combined_dashboard['plots'])} plot(s) to frontend"
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


@app.get("/state/{thread_id}")
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


@app.delete("/state/{thread_id}")
async def clear_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        agent_graph = _admin_graph()
        await agent_graph.aupdate_state(config, {"messages": [], "dashboard_data": {}})
        return JSONResponse(content={"status": "cleared"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── History / Rewind (back & retry) ───────────────────────────────────────────


@app.get("/history/{thread_id}")
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


@app.post("/rewind/{thread_id}")
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


class RenameSessionRequest(BaseModel):
    name: str


@app.get("/sessions")
async def list_sessions_endpoint(
    project_id: str | None = None,
    skip: int = 0,
    limit: int = 50,
):
    rows = sessions_store.list_sessions(project_id=project_id)
    enriched = []
    for row in rows:
        detail = sessions_store.get_session(row["thread_id"])
        enriched.append(detail if detail else row)
    total = len(enriched)
    page = enriched[skip : skip + limit]
    return JSONResponse(content={"sessions": page, "total": total})


@app.post("/sessions")
async def create_session_endpoint(body: CreateSessionRequest):
    return JSONResponse(
        content=sessions_store.create_session(body.name, project_id=body.project_id)
    )


@app.get("/sessions/{thread_id}")
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


@app.patch("/sessions/{thread_id}")
async def rename_session_endpoint(thread_id: str, body: RenameSessionRequest):
    if not sessions_store.rename_session(thread_id, body.name):
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(content={"status": "ok"})


@app.delete("/sessions/{thread_id}")
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
):
    plans = sessions_store.list_analysis_plans(thread_id=thread_id)
    page = plans[:limit]
    return JSONResponse(content={"plans": page, "total": len(plans)})


@app.get("/analysis-plans/{plan_id}")
async def get_analysis_plan_endpoint(plan_id: str):
    plan = sessions_store.get_analysis_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="plan not found")
    return JSONResponse(content=plan)


@app.post("/analysis-plans")
async def lock_analysis_plan_endpoint(body: LockPlanRequest):
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
async def delete_analysis_plan_endpoint(plan_id: str):
    if not sessions_store.delete_analysis_plan(plan_id):
        raise HTTPException(status_code=404, detail="plan not found")
    return JSONResponse(content={"status": "ok"})


# ── Models (stubs derived from session model_run artifacts) ───────────────────


@app.get("/models")
async def list_models_endpoint(
    limit: int = 8,
    status: str | None = None,
    project_id: str | None = None,
):
    """Return model_run artifacts from all sessions as lightweight ModelInfo stubs."""
    import time as _time

    all_sessions = sessions_store.list_sessions()
    models = []
    for s in all_sessions:
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
async def get_model_endpoint(model_id: str):
    import time as _time

    art, tid = _find_model_artifact(model_id)
    if art is None:
        raise HTTPException(status_code=404, detail="model not found")
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
async def get_model_dashboard_endpoint(model_id: str):
    """Return roi_metrics + decomposition + summary from the model's LangGraph thread."""
    art, tid = _find_model_artifact(model_id)
    if art is None:
        raise HTTPException(status_code=404, detail="model not found")
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
async def list_projects_endpoint():
    projects = sessions_store.list_projects()
    return JSONResponse(content={"projects": projects, "total": len(projects)})


@app.post("/projects")
async def create_project_endpoint(body: ProjectCreateRequest):
    return JSONResponse(
        content=sessions_store.create_project(body.name, body.description)
    )


@app.get("/projects/{project_id}")
async def get_project_endpoint(project_id: str):
    proj = sessions_store.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(content=proj)


@app.patch("/projects/{project_id}")
async def update_project_endpoint(project_id: str, body: ProjectUpdateRequest):
    if not sessions_store.update_project(project_id, body.name, body.description):
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return JSONResponse(content=sessions_store.get_project(project_id))


@app.delete("/projects/{project_id}")
async def delete_project_endpoint(project_id: str):
    if not sessions_store.delete_project(project_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete (not found or is the Default Project)",
        )
    return JSONResponse(content={"success": True})


# ── Budget Plans (stub — agent API has no budget plan management) ──────────────


@app.get("/budget-plans")
async def list_budget_plans_endpoint():
    return JSONResponse(content={"plans": [], "total": 0})


# ── Artifacts ─────────────────────────────────────────────────────────────────


@app.get("/artifacts/{thread_id}")
async def list_artifacts_endpoint(thread_id: str):
    return JSONResponse(content=sessions_store.list_artifacts(thread_id))


@app.delete("/artifacts/{artifact_id}")
async def delete_artifact_endpoint(artifact_id: str):
    sessions_store.delete_artifact(artifact_id)
    return JSONResponse(content={"status": "ok"})


@app.get("/sessions/{thread_id}/export")
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


@app.get("/assumptions/{thread_id}")
async def list_assumptions_endpoint(thread_id: str, history: bool = False):
    return JSONResponse(
        content=sessions_store.list_assumptions(thread_id, include_history=history)
    )


@app.get("/assumption_history/{thread_id}/{key:path}")
async def get_assumption_history_endpoint(thread_id: str, key: str):
    """Distinct route prefix so {key:path} can't shadow the list endpoint."""
    history = sessions_store.get_assumption_history(thread_id, key)
    if not history:
        raise HTTPException(status_code=404, detail="assumption not found")
    return JSONResponse(content=history)


@app.post("/assumptions/{thread_id}")
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


@app.delete("/assumption/{thread_id}/{key:path}")
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


@app.get("/workflow/{thread_id}")
async def workflow_status_endpoint(thread_id: str):
    return JSONResponse(content=await _infer_workflow_status(thread_id))


class WorkflowStepBody(BaseModel):
    status: str
    notes: str | None = None


@app.patch("/workflow/{thread_id}/{step}")
async def update_workflow_step_endpoint(
    thread_id: str, step: int, body: WorkflowStepBody
):
    if step < 1 or step > 9:
        raise HTTPException(status_code=400, detail="step must be 1..9")
    rec = sessions_store.set_workflow_step(thread_id, step, body.status, body.notes)
    return JSONResponse(content=rec)


# ── Files registry ───────────────────────────────────────────────────────────


@app.get("/files/{thread_id}")
async def list_files_endpoint(thread_id: str):
    return JSONResponse(content=sessions_store.list_files(thread_id))


@app.delete("/files/{file_id}")
async def delete_file_endpoint(file_id: str):
    sessions_store.delete_file(file_id)
    return JSONResponse(content={"status": "ok"})


# ── Generated-file downloads (req 4) ──────────────────────────────────────────


def _guarded_file_response(path: str, filename: str | None = None) -> FileResponse:
    """FileResponse over `path`, but only if it sits inside an allowed root
    (workspace / uploads / mmm_models / mmm_configs / CWD) — blocks traversal."""
    from mmm_framework.agents import workspace as _ws

    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    if not _ws.is_within(path):
        raise HTTPException(status_code=403, detail="File is outside the allowed roots")
    return FileResponse(
        path,
        filename=filename or os.path.basename(path),
        media_type="application/octet-stream",
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


@app.get("/workspace/{thread_id}/files")
async def workspace_files_endpoint(thread_id: str):
    """List the session's registered files (uploads + generated), each with a
    download id."""
    files = sessions_store.list_files(thread_id)
    return JSONResponse(content={"files": files, "total": len(files)})


@app.get("/files/{file_id}/download")
async def download_file_endpoint(file_id: str):
    rec = sessions_store.get_file(file_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return _guarded_file_response(rec["path"], rec.get("name"))


@app.get("/artifacts/{artifact_id}/download")
async def download_artifact_endpoint(artifact_id: str):
    art = sessions_store.get_artifact(artifact_id)
    if art is None:
        raise HTTPException(
            status_code=404, detail=f"Artifact not found: {artifact_id}"
        )
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


@app.post("/projects/{project_id}/kb")
async def kb_upload_endpoint(project_id: str, file: UploadFile = File(...)):
    """Add a document to a project's knowledge base: store it, then chunk +
    embed it (in a threadpool) so it becomes searchable."""
    from fastapi.concurrency import run_in_threadpool
    from mmm_framework.agents import workspace as _ws
    from mmm_framework.agents import knowledge_base as kb

    if sessions_store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    kb_dir = _ws.project_kb_dir(project_id)
    name = file.filename or "document.txt"
    dest = os.path.join(str(kb_dir), name)
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
        meta={"content_type": file.content_type},
    )
    # Ingest synchronously-in-threadpool so the response reflects final status.
    doc = await run_in_threadpool(kb.ingest_document, doc["id"])
    return JSONResponse(content=doc)


@app.get("/projects/{project_id}/kb")
async def kb_list_endpoint(project_id: str):
    docs = sessions_store.list_kb_documents(project_id)
    return JSONResponse(content={"documents": docs, "total": len(docs)})


@app.get("/projects/{project_id}/kb/search")
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
async def kb_delete_endpoint(document_id: str):
    doc = sessions_store.get_kb_document(document_id)
    if doc is None:
        raise HTTPException(
            status_code=404, detail=f"Document not found: {document_id}"
        )
    sessions_store.delete_kb_document(document_id)
    # best-effort remove the file on disk
    try:
        if doc.get("path") and os.path.isfile(doc["path"]):
            os.remove(doc["path"])
    except OSError:
        pass
    return JSONResponse(content={"success": True})


# ── DAG endpoint (read current DAG from state, return React Flow JSON) ───────


@app.get("/dag/{thread_id}")
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


@app.put("/dag/{thread_id}")
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


# ── Dataset preview ─────────────────────────────────────────────────────────


@app.get("/dataset/preview/{thread_id}")
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


@app.post("/upload")
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
    file_location = os.path.abspath(
        os.path.join(upload_dir, file.filename or "upload.bin")
    )
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    size_bytes = os.path.getsize(file_location)
    preview: str | None = None
    name = file.filename or "upload.bin"
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


@app.get("/report")
async def view_report():
    """Serve the generated HTML report inline for embedding."""
    report_path = "agent_mmm_report.html"
    if not os.path.exists(report_path):
        return JSONResponse(
            status_code=404,
            content={"error": "No report generated yet. Fit a model first."},
        )
    return FileResponse(report_path, media_type="text/html")


@app.get("/report/download")
async def download_report():
    """Download the generated HTML report."""
    report_path = "agent_mmm_report.html"
    if not os.path.exists(report_path):
        return JSONResponse(
            status_code=404, content={"error": "No report generated yet."}
        )
    return FileResponse(
        report_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=mmm_report.html"},
    )


@app.get("/project-report")
async def view_project_report():
    """Serve the project findings HTML report."""
    p = "agent_project_report.html"
    if not os.path.exists(p):
        return JSONResponse(
            status_code=404,
            content={
                "error": "No project report yet. Ask the agent to generate_project_report."
            },
        )
    return FileResponse(p, media_type="text/html")


@app.get("/project-report/download")
async def download_project_report():
    p = "agent_project_report.html"
    if not os.path.exists(p):
        return JSONResponse(
            status_code=404, content={"error": "No project report yet."}
        )
    return FileResponse(
        p,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=mmm_project_report.html"},
    )


@app.get("/project-slides")
async def view_project_slides():
    """Serve the Reveal.js project slideshow."""
    p = "agent_project_slides.html"
    if not os.path.exists(p):
        return JSONResponse(
            status_code=404,
            content={
                "error": "No slideshow yet. Ask the agent to generate_project_report."
            },
        )
    return FileResponse(p, media_type="text/html")


@app.get("/project-slides/download")
async def download_project_slides():
    p = "agent_project_slides.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No slideshow yet."})
    return FileResponse(
        p,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=mmm_project_slides.html"},
    )


@app.get("/client-report")
async def view_client_report():
    """Serve the client-ready HTML report (no diagnostics, with nav + confidentiality notice)."""
    p = "agent_client_report.html"
    if not os.path.exists(p):
        return JSONResponse(
            status_code=404,
            content={
                "error": "No client report yet. Ask the agent to generate_client_report."
            },
        )
    return FileResponse(p, media_type="text/html")


@app.get("/client-report/download")
async def download_client_report():
    p = "agent_client_report.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No client report yet."})
    return FileResponse(
        p,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=mmm_client_report.html"},
    )


@app.get("/client-slides")
async def view_client_slides():
    """Serve the client-ready Reveal.js slideshow (no MCMC stats, with confidentiality footer)."""
    p = "agent_client_slides.html"
    if not os.path.exists(p):
        return JSONResponse(
            status_code=404,
            content={
                "error": "No client slides yet. Ask the agent to generate_client_slides."
            },
        )
    return FileResponse(p, media_type="text/html")


@app.get("/client-slides/download")
async def download_client_slides():
    p = "agent_client_slides.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No client slides yet."})
    return FileResponse(
        p,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=mmm_client_slides.html"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mmm_framework.api.main:app", host="0.0.0.0", port=8000, reload=True)
