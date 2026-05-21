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
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
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


def get_llm(model_name: str | None, api_key: str | None):
    if not model_name:
        model_name = "claude-sonnet-4-6"

    if "gpt" in model_name.lower():
        from langchain_openai import ChatOpenAI
        kwargs: dict = {"model": model_name, "temperature": 0}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)
    elif "claude" in model_name.lower():
        from langchain_anthropic import ChatAnthropic
        kwargs = {"model": model_name, "temperature": 0}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": model_name, "temperature": 0}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)


# ── Chat ──────────────────────────────────────────────────────────────────────

async def _hard_reset_thread(thread_id: str) -> None:
    """Delete all LangGraph checkpoints for thread_id directly from SQLite.

    Nuclear recovery: preserves session metadata (artifacts, assumptions, files)
    but wipes the LangGraph conversation state so the next request starts fresh.
    """
    if _aiosqlite_conn is None:
        return
    try:
        await _aiosqlite_conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        await _aiosqlite_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
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
                for tc in (m.tool_calls or []):
                    tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
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
        logger.exception("Failed to repair orphan tool_calls for %s; hard-resetting", thread_id)
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
):
    config = {"configurable": {"thread_id": request.thread_id}}
    llm = get_llm(x_model_name, x_api_key)
    agent_graph = create_agent_graph(llm, checkpointer=memory)

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
            elif _k == "code_snippet":
                captured_artifact_keys.add(f"code::{_p.get('call_id', '')}")

        try:
            stream = agent_graph.astream(
                {"messages": [initial_message]}, config, stream_mode="updates"
            )
            async for event in stream:
                if await raw_request.is_disconnected():
                    logger.info("Client disconnected; aborting stream for %s", request.thread_id)
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    break

                for node_name, state_update in event.items():
                    updates = state_update if isinstance(state_update, list) else [state_update]

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
                                if isinstance(block, dict) and block.get('type') == 'text':
                                    text_parts.append(block.get('text', ''))
                                elif isinstance(block, str):
                                    text_parts.append(block)
                            content = '\n'.join(text_parts)

                        tool_calls = getattr(msg, 'tool_calls', []) or []
                        tool_call_id = getattr(msg, 'tool_call_id', None)

                        # Persist execute_python code as an artifact (once per call_id)
                        for tc in tool_calls:
                            if tc.get("name") == "execute_python":
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

                        msg_dashboard = {k: v for k, v in combined_dashboard.items() if k != 'plots'}
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
                                "content": f"[Result could not be serialized: {str(e)[:120]}]" if msg_type == "tool" else "",
                                "tool_call_id": tool_call_id,
                                "tool_calls": [],
                                "dashboard_data": {},
                            }
                            try:
                                yield f"data: {json.dumps(fallback)}\n\n"
                            except Exception:
                                pass
                        await asyncio.sleep(0.01)

                    if combined_dashboard.get('plots'):
                        plots_event = {
                            "type": "dashboard_update",
                            "dashboard_data": {"plots": combined_dashboard["plots"]},
                        }
                        try:
                            yield f"data: {safe_json_dumps(plots_event)}\n\n"
                            logger.info(f"Sent {len(combined_dashboard['plots'])} plot(s) to frontend")
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
            err_lower = str(e).lower()
            is_api_400 = any(
                kw in err_lower
                for kw in ("400", "bad request", "invalid_request_error", "badrequest")
            )
            if is_api_400:
                logger.warning("Anthropic 400 for %s; hard-resetting thread", request.thread_id)
                await _hard_reset_thread(request.thread_id)
                recovery = (
                    "The conversation history was corrupted and has been automatically reset. "
                    "Your session is now fresh — please resend your message."
                )
                yield f"data: {json.dumps({'type': 'error', 'content': recovery})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
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
        return JSONResponse(content=safe_json_dumps_load(_serialize_state(state.values)))
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
                (i for i in range(len(visible) - 1, -1, -1)
                 if getattr(visible[i], "type", None) == "human"),
                None,
            )
            timeline.append(
                {
                    "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
                    "parent_checkpoint_id": (snap.parent_config or {}).get("configurable", {}).get("checkpoint_id"),
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
        async for s in agent_graph.aget_state_history({"configurable": {"thread_id": thread_id}}):
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
        return JSONResponse(content=safe_json_dumps_load(_serialize_state(live.values or {})))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rewind failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Sessions ──────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    name: str | None = None


class RenameSessionRequest(BaseModel):
    name: str


@app.get("/sessions")
async def list_sessions_endpoint():
    return JSONResponse(content=sessions_store.list_sessions())


@app.post("/sessions")
async def create_session_endpoint(body: CreateSessionRequest):
    return JSONResponse(content=sessions_store.create_session(body.name))


@app.patch("/sessions/{thread_id}")
async def rename_session_endpoint(thread_id: str, body: RenameSessionRequest):
    if not sessions_store.rename_session(thread_id, body.name):
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(content={"status": "ok"})


@app.delete("/sessions/{thread_id}")
async def delete_session_endpoint(thread_id: str):
    sessions_store.delete_session(thread_id)
    return JSONResponse(content={"status": "ok"})


# ── Artifacts ─────────────────────────────────────────────────────────────────

@app.get("/artifacts/{thread_id}")
async def list_artifacts_endpoint(thread_id: str):
    return JSONResponse(content=sessions_store.list_artifacts(thread_id))


@app.delete("/artifacts/{artifact_id}")
async def delete_artifact_endpoint(artifact_id: str):
    sessions_store.delete_artifact(artifact_id)
    return JSONResponse(content={"status": "ok"})


# ── Assumptions log ──────────────────────────────────────────────────────────

class RecordAssumptionBody(BaseModel):
    key: str
    value: Any
    rationale: str
    category: str = "other"
    change_note: str | None = None


@app.get("/assumptions/{thread_id}")
async def list_assumptions_endpoint(thread_id: str, history: bool = False):
    return JSONResponse(content=sessions_store.list_assumptions(thread_id, include_history=history))


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
        thread_id=thread_id, key=body.key, value=body.value,
        rationale=body.rationale, category=body.category, change_note=body.change_note,
    )
    return JSONResponse(content=rec)


class RetractAssumptionBody(BaseModel):
    reason: str


@app.delete("/assumption/{thread_id}/{key:path}")
async def retract_assumption_endpoint(thread_id: str, key: str, body: RetractAssumptionBody):
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
        "done" if has("dag_structure")
        else "in_progress" if (state_values.get("dataset_path") or dashboard.get("dataset"))
        else "pending"
    )
    inferred[3] = "done" if (state_values.get("model_spec") or dashboard.get("model_spec")) else "pending"
    inferred[4] = "done" if has("prior_predictive_check") else "pending"
    inferred[5] = "done" if state_values.get("model_status") == "completed" else "pending"
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
        active_override = override if (override and override["status"] != "pending") else None
        status = active_override["status"] if active_override else inferred[step]
        notes = active_override["notes"] if active_override else None
        out.append({
            "step": step, "title": title, "description": desc,
            "status": status, "notes": notes,
            "inferred_status": inferred[step],
            "overridden": active_override is not None,
            "updated_at": active_override["updated_at"] if active_override else None,
        })
    return out


@app.get("/workflow/{thread_id}")
async def workflow_status_endpoint(thread_id: str):
    return JSONResponse(content=await _infer_workflow_status(thread_id))


class WorkflowStepBody(BaseModel):
    status: str
    notes: str | None = None


@app.patch("/workflow/{thread_id}/{step}")
async def update_workflow_step_endpoint(thread_id: str, step: int, body: WorkflowStepBody):
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
            raise HTTPException(status_code=400, detail="not an MFF dataset (no VariableName column)")

        sub = df[df["VariableName"] == variable].copy()
        if sub.empty:
            raise HTTPException(status_code=404, detail=f"variable '{variable}' not found")

        if dim and value:
            if dim not in df.columns:
                raise HTTPException(status_code=400, detail=f"dimension '{dim}' not found")
            if value == "(national)":
                sub = sub[sub[dim].isna()]
            else:
                sub = sub[sub[dim] == value]

        date_cols = [c for c in df.columns if any(k in c.lower() for k in ("date", "week", "period", "time"))]
        if not date_cols:
            raise HTTPException(status_code=400, detail="no date column found in dataset")
        date_col = date_cols[0]

        series = (
            sub.groupby(date_col)["VariableValue"]
            .sum()
            .reset_index()
            .rename(columns={date_col: "date", "VariableValue": "value"})
            .sort_values("date")
        )

        return JSONResponse(content={
            "variable": variable,
            "dim": dim,
            "value": value,
            "series": series.to_dict(orient="records"),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("dataset_preview failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Misc ──────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), thread_id: str | None = None):
    """Upload a file scoped to a session. Files land in `uploads/{thread_id}/`
    so concurrent sessions can't clobber each other's filenames, and the file
    is registered in the session's data_files table for the Files panel.
    """
    tid = thread_id or "_unscoped"
    upload_dir = os.path.join("uploads", tid)
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename or "upload.bin")
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

    return {"filename": name, "path": file_location, "size_bytes": size_bytes, "kind": kind}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/report")
async def view_report():
    """Serve the generated HTML report inline for embedding."""
    report_path = "agent_mmm_report.html"
    if not os.path.exists(report_path):
        return JSONResponse(status_code=404, content={"error": "No report generated yet. Fit a model first."})
    return FileResponse(report_path, media_type="text/html")


@app.get("/report/download")
async def download_report():
    """Download the generated HTML report."""
    report_path = "agent_mmm_report.html"
    if not os.path.exists(report_path):
        return JSONResponse(status_code=404, content={"error": "No report generated yet."})
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
        return JSONResponse(status_code=404, content={"error": "No project report yet. Ask the agent to generate_project_report."})
    return FileResponse(p, media_type="text/html")


@app.get("/project-report/download")
async def download_project_report():
    p = "agent_project_report.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No project report yet."})
    return FileResponse(p, media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=mmm_project_report.html"})


@app.get("/project-slides")
async def view_project_slides():
    """Serve the Reveal.js project slideshow."""
    p = "agent_project_slides.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No slideshow yet. Ask the agent to generate_project_report."})
    return FileResponse(p, media_type="text/html")


@app.get("/project-slides/download")
async def download_project_slides():
    p = "agent_project_slides.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No slideshow yet."})
    return FileResponse(p, media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=mmm_project_slides.html"})


@app.get("/client-report")
async def view_client_report():
    """Serve the client-ready HTML report (no diagnostics, with nav + confidentiality notice)."""
    p = "agent_client_report.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No client report yet. Ask the agent to generate_client_report."})
    return FileResponse(p, media_type="text/html")


@app.get("/client-report/download")
async def download_client_report():
    p = "agent_client_report.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No client report yet."})
    return FileResponse(p, media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=mmm_client_report.html"})


@app.get("/client-slides")
async def view_client_slides():
    """Serve the client-ready Reveal.js slideshow (no MCMC stats, with confidentiality footer)."""
    p = "agent_client_slides.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No client slides yet. Ask the agent to generate_client_slides."})
    return FileResponse(p, media_type="text/html")


@app.get("/client-slides/download")
async def download_client_slides():
    p = "agent_client_slides.html"
    if not os.path.exists(p):
        return JSONResponse(status_code=404, content={"error": "No client slides yet."})
    return FileResponse(p, media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=mmm_client_slides.html"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mmm_framework.api.main:app", host="0.0.0.0", port=8000, reload=True)
