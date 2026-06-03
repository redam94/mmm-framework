# Agent Knowledge Base & Workspace — Design Contract

This document is the frozen interface contract for the "agent superpowers" upgrade.
It is the single source of truth shared by the backend (`src/mmm_framework/api`,
`src/mmm_framework/agents`) and the React frontend (`frontend/src/pages/AgentPage.tsx`).

The upgrade delivers seven capabilities:

1. The agent can reach **all** library features (curated power tools + discovery + `execute_python` escape hatch).
2. Users **add files** to a **project-level** knowledge base (KB).
3. The agent can **look up** the KB (retrieval / RAG).
4. Generated files are **downloadable** from the frontend.
5. Python **text outputs** are persisted and rendered nicely in the Artifacts tab.
6. **Previous results** are reusable / queryable by the agent.
7. The agent can **see and grep** files in its output directory.

The unifying primitive is a **scoped workspace directory** plus **project identity**.

---

## 1. Scope model (projects, sessions, threads)

* A **project** groups sessions and owns a knowledge base. New `projects` table in the
  agent store (`src/mmm_framework/api/sessions.db`).
* A **session** (LangGraph `thread_id`) belongs to exactly one project via the existing
  `sessions.project_id` column. On first run a `Default Project` is auto-created and all
  pre-existing/orphan sessions resolve to it.
* The **KB is scoped to `project_id`**. Output files are scoped to `thread_id`.

### Project resolution
`resolve_project_id(thread_id) -> str` (in `sessions.py`): the session's `project_id`,
falling back to the Default Project id. Never returns `None`.

---

## 2. Workspace directory layout

Root: `MMM_AGENT_WORKSPACE` env var, else `<cwd>/agent_workspace`. Resolved absolute.

```
<workspace_root>/
  threads/<thread_id>/        # per-session OUTPUT dir. execute_python chdir's here.
                             #   reports, generated CSV/PNG, etc. land here. greppable.
  projects/<project_id>/kb/   # per-project KB source files (uploaded context docs)
uploads/<thread_id>/          # EXISTING dataset-upload dir (unchanged, back-compat)
```

Helpers in `src/mmm_framework/agents/workspace.py`:
* `workspace_root() -> Path`
* `thread_dir(thread_id) -> Path`  (mkdir -p)
* `project_kb_dir(project_id) -> Path`  (mkdir -p)
* `safe_join(root: Path, relpath: str) -> Path` — resolves and **guards against traversal** (raises `ValueError` if the result escapes `root`).
* `is_within(path, *roots) -> bool` — download guard: path must be inside one of the allowed roots (`workspace_root`, `uploads/`, `mmm_models/`, `mmm_configs/`).
* `register_generated_files(thread_id, before: set, kind="export")` — diff a dir snapshot and `register_file` new/changed files into `data_files`.

---

## 3. Active-thread context + model cache  (`src/mmm_framework/agents/runtime.py`)

* `current_thread_id: ContextVar[str|None]`.
* `set_current_thread(tid)`, `get_current_thread() -> str` (returns `"__default__"` if unset).
* `MODEL_CACHE` — thread-scoped, bounded **LRU (maxsize 2)** dict-proxy. `.get(key)`,
  `[key]`, `__setitem__`, `__contains__` operate on the **current thread's** bucket
  (selected by `current_thread_id`). Disk persistence (`mmm_models/run_*`) is the durable
  fallback when a bucket is evicted. Exported from `tools.py` as `_MODEL_CACHE` for
  back-compat with `causal_tools.py`.
* The contextvar is set in two places (belt + suspenders, because LangGraph may run sync
  tools in an executor where context does not auto-propagate):
  1. top of the `/chat` handler, and
  2. top of every model-/workspace-using tool, from `InjectedConfig`'s `thread_id`.

---

## 4. Database tables  (added to `src/mmm_framework/api/sessions.py`, idempotent `init_db`)

### `projects`
| col | type | notes |
|---|---|---|
| project_id | TEXT PK | uuid4 hex |
| name | TEXT NOT NULL | |
| description | TEXT | |
| created_at | REAL NOT NULL | |
| updated_at | REAL NOT NULL | |

### `kb_documents`
| col | type | notes |
|---|---|---|
| id | TEXT PK | uuid4 hex |
| project_id | TEXT NOT NULL | |
| name | TEXT NOT NULL | original filename |
| path | TEXT NOT NULL | on-disk path under projects/<pid>/kb/ |
| kind | TEXT NOT NULL | text/markdown/csv/pdf/docx/xlsx |
| size_bytes | INTEGER | |
| n_chunks | INTEGER DEFAULT 0 | |
| status | TEXT NOT NULL | `pending`/`ready`/`error` |
| error | TEXT | |
| meta_json | TEXT | |
| created_at | REAL NOT NULL | |

### `kb_chunks`
| col | type | notes |
|---|---|---|
| id | TEXT PK | |
| document_id | TEXT NOT NULL | FK-by-convention |
| project_id | TEXT NOT NULL | denormalized for fast project search |
| chunk_index | INTEGER NOT NULL | |
| text | TEXT NOT NULL | |
| embedding | BLOB NOT NULL | float32 little-endian, dim from model |
| dim | INTEGER NOT NULL | |
| created_at | REAL NOT NULL | |

Functions (mirroring existing `register_file`/`list_files` style):
`create_project`, `list_projects`, `get_project`, `update_project`, `delete_project`,
`ensure_default_project`, `resolve_project_id`;
`add_kb_document`, `set_kb_document_status`, `list_kb_documents`, `get_kb_document`,
`delete_kb_document`; `add_kb_chunks(document_id, project_id, chunks: list[(idx,text,vec)])`,
`iter_kb_chunks(project_id) -> list[dict]` (id, document_id, text, embedding bytes, dim).

---

## 5. Embeddings  (`src/mmm_framework/agents/embeddings.py`)

`build_embeddings(cfg: ModelConfig | None = None) -> Embeddings` resolves an embedder
**independently of the chat LLM** (Anthropic/`vertex_anthropic` have no embedder):

| chat provider | embedder | default model |
|---|---|---|
| vertex_anthropic, vertex_gemini | `VertexAIEmbeddings` (ADC, same `project`) | `text-embedding-005` |
| google_genai | `GoogleGenerativeAIEmbeddings` | `models/text-embedding-004` |
| openai | `OpenAIEmbeddings` | `text-embedding-3-small` |
| anthropic (direct) | Vertex if `GOOGLE_CLOUD_PROJECT` set, else OpenAI if key, else error | |

Env overrides: `MMM_EMBED_PROVIDER`, `MMM_EMBED_MODEL`, `MMM_EMBED_LOCATION`
(Vertex embed location default `us-central1`, independent of the Claude region `us-east5`).
`embed_documents(texts) -> list[list[float]]`, `embed_query(text) -> list[float]`.

---

## 6. Knowledge base pipeline  (`src/mmm_framework/agents/knowledge_base.py`)

* `extract_text(path) -> str` — txt/md/csv (utf-8), pdf (`pypdf`), docx (`python-docx`),
  xlsx (`openpyxl`). Missing optional dep → clear error string, status=`error`.
* `chunk_text(text, size=1200, overlap=200) -> list[str]` (char-based, paragraph-aware).
* `ingest_document(project_id, path, name, kind) -> kb_document` — extract → chunk → embed
  → `add_kb_chunks`; sets status. Runs in a threadpool from the endpoint.
* `search(project_id, query, top_k=6) -> list[{document, chunk_index, text, score}]` —
  brute-force cosine over `iter_kb_chunks` (numpy). No vector-store dependency.

---

## 7. Agent tools  (added to `TOOLS` in `src/mmm_framework/agents/tools.py`)

**Knowledge base**
* `search_knowledge_base(query, top_k=6)` → formatted top-k snippets for the active project.
* `list_knowledge_base()` → documents in the active project's KB.

**Filesystem (workspace-scoped, req 7)**
* `list_workspace_files(subdir="")` → tree of the thread workspace.
* `read_workspace_file(path, max_bytes=20000)` → file contents (guarded).
* `grep_workspace(pattern, glob="*", max_results=100)` → regex matches with file:line.

**Results reuse (req 6)**
* `list_artifacts_tool()` / `query_past_results(kind=None)` → prior artifacts (model_run,
  report, text_output, code_snippet) for the session, with download ids.

**Library power tools (req 1)**
* `library_reference(topic=None)` → curated menu of reachable capabilities + import paths +
  the input-shape/ordering traps (extensions take raw arrays; calibration before fit; the
  DAG→model-type bridge).
* `fit_extended_model(model_type, ...)` → mediation/multivariate/combined via the
  `DAGModelBuilder` model-type bridge (or direct constructors), respecting input shapes.
* `run_budget_scenario(spend_changes)` / `run_counterfactual(...)` → `MMMAnalyzer`.
* `add_lift_test_calibration(...)` → `ExperimentMeasurement` + re-fit ordering note.
* register the previously-omitted `define_analysis_plan`, `check_spec_divergence`.

`execute_python` is enhanced: chdir into `thread_dir`, inject `OUTPUT_DIR`, pre-import the
framework surface (`BayesianMMM`, builders, `mmm_framework.analysis`, `mmm_extensions`,
`reporting`), and **register newly written files** into `data_files` (download + grep).

---

## 8. Artifacts  (persisted in the `/chat` capture loop, `src/mmm_framework/api/main.py`)

Existing kinds: `code_snippet`, `report`, `project_report`, `project_slides`,
`client_report`, `client_slides`, `model_run`.

**New kind `text_output`** (req 5/6) — one per `execute_python` tool result:
```
{ "call_id": <tool_call_id>, "stdout": <str>, "plot_count": <int>, "is_error": <bool> }
```
Dedup key `text_output::{call_id}`. Frontend rehydrates these into the Python REPL widget
on session load (instead of resetting to []).

**New kind `kb_ingest`** (optional, informational) is NOT used; KB docs live in `kb_documents`.

---

## 9. HTTP API  (all on the **agent app** `src/mmm_framework/api/main.py`, port 8000)

### Projects
* `GET /projects` → `{projects:[{project_id,name,description,session_count,doc_count,created_at,updated_at}], total}`
* `POST /projects` `{name, description?}` → project  (replaces the old 501 stub)
* `GET /projects/{project_id}` → project
* `PATCH /projects/{project_id}` `{name?, description?}` → project
* `DELETE /projects/{project_id}` → `{success}` (sessions become unassigned → Default)

### Knowledge base
* `POST /projects/{project_id}/kb` (multipart `file`) → kb_document (ingest in threadpool)
* `GET  /projects/{project_id}/kb` → `{documents:[...], total}`
* `GET  /projects/{project_id}/kb/search?q=...&k=6` → `{results:[{document,chunk_index,text,score}]}`
* `DELETE /kb/{document_id}` → `{success}` (removes chunks + file)

### Downloads (req 4)
* `GET /files/{file_id}/download` → FileResponse over `data_files.path` (guarded)
* `GET /artifacts/{artifact_id}/download` → FileResponse over the artifact's `path`/`model_path`/`report_path` (guarded)
* `GET /workspace/{thread_id}/files` → `{files:[{id,name,path,kind,size_bytes,created_at}]}` (data_files for the thread)

`/sessions` POST gains optional `project_id`. `/upload` unchanged but also accepts
`project_id` (ignored for datasets). All new routes require the existing `X-API-Key` dep.

---

## 10. Frontend  (`frontend/src/pages/AgentPage.tsx`, `CausalWidgets.tsx`, `package.json`)

* **Project picker** in `SessionSidebar`: a dropdown above the session list to select / create
  a project; sessions list filters by the selected project; new sessions inherit it. Persist
  the selection in `localStorage('mmm.projectId')`.
* **Knowledge tab** (new tab `knowledge` in the tab array): drag-drop upload to
  `POST /projects/{pid}/kb`, document list (name/kind/size/status, delete), and a search box
  hitting `/kb/search`. Mirrors the `DataFilesWidget` pattern.
* **Downloads**: every code-snippet / report / generated-file card and a new
  `WorkspaceFilesWidget` (Files tab) gets a download link to `/files/{id}/download` or
  `/artifacts/{id}/download`, reusing the `DocCard` download pattern.
* **Persisted python outputs**: `loadThreadState` rehydrates `text_output` artifacts into
  `pythonOutputs` instead of resetting to `[]`.
* **Syntax highlighting**: add `react-syntax-highlighter`; wire into `MD_COMPONENTS.code`,
  `PythonCodeBlock`, and the `ArtifactsPanel` code `<pre>`.

---

## 11b. Plot caching (browser-side)

Captured Plotly figures used to be accumulated as full JSON in `dashboard_data["plots"]`
and re-streamed in their entirety on every turn (O(N²) wire traffic over a session, and
checkpoint bloat). Now each figure is **content-addressed**: `execute_python` writes it
once to `<workspace_root>/plots/<sha256>.json` (`workspace.store_plot`) and stores only a
lightweight `{id, title}` ref in state. `GET /plots/{id}` serves the JSON with
`Cache-Control: public, max-age=31536000, immutable` — since the id is a content hash, the
browser caches it permanently and identical figures dedup. `PlotCard` fetches by id once
(in-memory `Map` + browser HTTP cache) and renders inline figures too (back-compat for
plots saved before this change).

## 11c. Stateful `execute_python` ("warm kernel")

`execute_python` keeps a **persistent per-thread namespace** so the agent can build an
analysis incrementally — variables defined in one call are visible in the next, exactly
like notebook cells. This matches the LLM's universal prior about code tools (Jupyter /
code-interpreter), which previously broke because each call `exec`'d a fresh `env`.

* **Store:** `NAMESPACE_CACHE` in `runtime.py` — a `_ThreadScopedNamespace` (subclass of
  the model cache): thread-scoped, **LRU(2)-over-threads**, dropped on eviction / process
  restart. Same out-of-checkpoint rationale as `MODEL_CACHE` (the namespace holds live,
  non-msgpack-serializable objects).
* **Precedence contract:** each call re-layers the *reserved system bindings*
  (`pd/np/plt/mmf`, builders, `OUTPUT_DIR`, `dataset_path`, `save_result`/`load_result`,
  and `mmm`/`results` from `MODEL_CACHE`) on top of the persistent namespace via
  `ns.update(env)` — so a refit refreshes `mmm`/`results` and system names can't be
  permanently shadowed. **User-defined names persist untouched.** `exec(code, ns)` uses a
  **single dict** (splitting globals/locals would break top-level `def`/`class` scoping).
* **Dataset auto-bind:** `df` is auto-loaded from `dataset_path` **once** (only if not
  already in the namespace; .csv/.parquet, ≤250 MB, best-effort) so the most common
  cross-cell reference works even on a cold kernel, while still letting the analyst
  reassign `df` and have it persist.
* **Durability (req: survive restart):** `save_result(name, obj)` / `load_result(name)`
  persist named objects to `<thread_dir>/results/` — parquet for DataFrames/Series
  (fallback pickle), cloudpickle otherwise. `list_saved_results()` lists them. The
  deliverable files an analyst writes to the workspace are durable regardless; this is for
  *intermediate* objects worth keeping.
* **Self-healing:** a `NameError` appends a hint that the kernel may have been reset and to
  rebuild / `load_result` — turning the one failure mode into a recoverable one.
* **`reset_namespace` tool:** clears the current thread's namespace for a fresh kernel
  (saved results on disk are untouched).

**Kernel abstraction (Phase 1 of `agent-session-kernels.md`).** The execution above is now
behind a `KernelManager` seam (`agents/kernels.py`); `execute_python` builds a `KernelContext`
and dispatches to `MANAGER.get_or_spawn(thread_id).execute(code, ctx)`, then does the unchanged
API-side post-processing (content-address plots, register files, build the `Command`). Two
implementations, selected by **`MMM_AGENT_KERNEL`** (default `inprocess`):
* **`InProcessKernel`** (default) — exactly the in-process warm namespace described above
  (delegates to `NAMESPACE_CACHE`); zero behavior change.
* **`SubprocessKernel`** (opt-in, `MMM_AGENT_KERNEL=subprocess`) — one `ipykernel` process per
  session via the sync `jupyter_client` API; the multi-user-ready evolution toward
  `agent-session-kernels.md`. Faithful to the in-process contract (same plot normalization,
  the `Error executing code` + NameError hint via the shared `format_execution_error`, echo
  suppression, df auto-bind). **Phase-1 boundary:** fits still run in the API process, so
  `mmm`/`results` are NOT bound in the subprocess (referencing them raises `NameError`) until
  Phase 2 relocates fits. Not sandboxed yet (Phase 3). Bounded by an LRU cap
  (`MMM_MAX_KERNELS`, default 8) + a per-cell wall-clock cap (`MMM_CELL_TIMEOUT`, default 600s);
  kernels are reaped on app shutdown (`atexit` + the FastAPI lifespan).

## 11. Concurrency / safety assumptions

Single-user / low-concurrency local tool. `execute_python` is **unsandboxed in-process
`exec`** — the workspace dir is an *organizational* boundary, not a security one. `chdir`
is wrapped in `try/finally` to always restore. The thread-scoped model cache **and the
warm-kernel namespace** are bounded (LRU 2) to cap memory. The warm namespace is
**process-local**: it persists only while successive `/chat` calls hit the same live
process (true for the documented single-process `uvicorn --reload`); under `--workers N`
or after a restart it goes cold and the agent falls back to `load_result` / rebuilding.
Two `execute_python` calls emitted in the *same* turn share one namespace dict (GIL-safe,
but interleaved) — acceptable under this single-user posture. Download routes are id-based
+ path-guarded to prevent traversal.
