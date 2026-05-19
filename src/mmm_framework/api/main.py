import json
import math
import asyncio
import logging
from fastapi import FastAPI, Request, UploadFile, File, Header
import os
import shutil
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from mmm_framework.agents.graph import create_agent_graph

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
        # Try common pandas types without importing pandas
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

app = FastAPI(title="MMM Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = MemorySaver()

def get_llm(model_name: str | None, api_key: str | None):
    # Default to a safe fallback if not provided
    if not model_name:
        model_name = "claude-sonnet-4-6"
        
    if "gpt" in model_name.lower():
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    elif "claude" in model_name.lower():
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"

@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    x_api_key: str | None = Header(None),
    x_model_name: str | None = Header(None)
):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    llm = get_llm(x_model_name, x_api_key)
    agent_graph = create_agent_graph(llm, checkpointer=memory)
    
    async def event_generator():
        initial_message = HumanMessage(content=request.message)
        try:
            # Note: We use the synchronous stream() in an async wrapper to avoid missing async implementations in tools if any.
            # But astream is preferred if tools support it. Let's try stream() in a thread or just astream if safe.
            # LangGraph handles async automatically if possible.
            async for event in agent_graph.astream({"messages": [initial_message]}, config, stream_mode="updates"):
                # event maps node_name -> state_update
                # state_update is a dict for single-tool calls, or a list of dicts for parallel tool calls
                for node_name, state_update in event.items():
                    # Normalise: wrap single dict in a list so we can handle both cases uniformly
                    updates = state_update if isinstance(state_update, list) else [state_update]

                    # Collect all messages and merge dashboard_data across all parallel updates
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

                    # Emit each message
                    for msg in all_messages:
                        msg_type = msg.type
                        # Normalize content: may be string or list of content blocks
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

                        # Emit the message event (without potentially-huge plots inline)
                        msg_dashboard = {k: v for k, v in combined_dashboard.items() if k != 'plots'}
                        data = {
                            "node": node_name,
                            "type": msg_type,
                            "content": content,
                            "tool_calls": tool_calls,
                            "tool_call_id": tool_call_id,
                            "dashboard_data": msg_dashboard
                        }
                        try:
                            yield f"data: {safe_json_dumps(data)}\n\n"
                        except Exception as e:
                            logger.error(f"Failed to serialize message: {e}")
                            # Emit a stripped fallback so the frontend can still mark the tool done
                            fallback = {
                                "node": node_name,
                                "type": msg_type,
                                "content": f"[Result could not be serialized: {str(e)[:120]}]" if msg_type == "tool" else "",
                                "tool_call_id": tool_call_id,
                                "tool_calls": [],
                                "dashboard_data": {}
                            }
                            try:
                                yield f"data: {json.dumps(fallback)}\n\n"
                            except Exception:
                                pass
                        await asyncio.sleep(0.01)

                    # Emit plots as a separate event to avoid size issues
                    if combined_dashboard.get('plots'):
                        plots_event = {
                            "type": "dashboard_update",
                            "dashboard_data": {"plots": combined_dashboard["plots"]}
                        }
                        try:
                            yield f"data: {safe_json_dumps(plots_event)}\n\n"
                            logger.info(f"Sent {len(combined_dashboard['plots'])} plot(s) to frontend")
                        except Exception as e:
                            logger.error(f"Failed to serialize plots: {e}")
                        await asyncio.sleep(0.01)
                        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/state/{thread_id}")
async def get_state(
    thread_id: str,
    x_api_key: str | None = Header(None),
    x_model_name: str | None = Header(None)
):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # We need an agent graph to get state, but getting state doesn't execute the LLM.
        # So we can just instantiate with the passed LLM
        llm = get_llm(x_model_name, x_api_key)
        agent_graph = create_agent_graph(llm, checkpointer=memory)
        
        state = agent_graph.get_state(config)
        if not state or not state.values:
            return JSONResponse(content={"messages": [], "dashboard_data": {}})
            
        values = state.values
        messages = []
        for msg in values.get("messages", []):
            messages.append({
                "type": msg.type,
                "content": msg.content,
                "tool_calls": getattr(msg, "tool_calls", [])
            })
            
        return JSONResponse(content={
            "messages": messages,
            "dashboard_data": values.get("dashboard_data", {})
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/state/{thread_id}")
async def clear_state(
    thread_id: str,
    x_api_key: str | None = Header(None),
    x_model_name: str | None = Header(None)
):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        llm = get_llm(x_model_name, x_api_key)
        agent_graph = create_agent_graph(llm, checkpointer=memory)
        # Overwrite the thread state with an empty messages list
        agent_graph.update_state(config, {"messages": [], "dashboard_data": {}})
        return JSONResponse(content={"status": "cleared"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_location}

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
        headers={"Content-Disposition": "attachment; filename=mmm_report.html"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mmm_framework.api.main:app", host="0.0.0.0", port=8000, reload=True)
