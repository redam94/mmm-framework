# Atelier Notebook — demo/test space for bespoke models

A Jupyter-like space inside the **Atelier** (Model Garden authoring IDE) where an
author uploads a dataset and runs free-form Python cells to **demonstrate and
test the bespoke model they're writing** — fit it, inspect ROI, plot a
decomposition — without leaving the editor or registering a version. Outputs
(plots, tables, stdout) and markdown notes are tracked per model and persisted.

It is almost entirely a **composition of primitives that already existed** (the
session kernel, the content-addressed plot/table stores, dataset upload, garden
class loading, the non-blocking job machinery, the React output components). The
notebook is the surface that wires them together.

## How it works

```
Atelier center panel (frontend/src/pages/ModelGarden/index.tsx)
  editorTab: 'code' | 'docs' | 'notebook'
        'notebook' → <AtelierNotebook name liveSource={editorValue} />

<AtelierNotebook>  (frontend/src/components/modelGarden/AtelierNotebook.tsx)
  • Upload data  → POST /model-garden/notebook/dataset      (stages CSV into the kernel workspace)
  • Run cell     → POST /model-garden/notebook/cell  → poll GET .../cell/{job_id}
  • Cells        → <NotebookCell> (code | markdown), add/delete/reorder, run-cell/run-all
  • Outputs      → PlotCard / TableCard / stdout box / ReactMarkdown   (reused from Agent)
  • Persist      → GET / PUT /model-garden/notebook   (one artifact per notebook; autosaved)
```

### Source: the live editor buffer
Cells run against **whatever source is in the Monaco editor** — `source_code` +
a `source_rev` hash. The kernel re-imports the model (binding it to `GardenModel`)
only when `source_rev` changes, so editing the model and re-running a cell shows
the new behaviour with **no register step**. When viewing a registered version
the editor shows that version's source (read-only), so the notebook runs it too.

### One warm kernel per notebook
Each `(org, model, source)` maps to a deterministic synthetic thread
(`__atelier_nb__{org}__{name}__draft`, `_notebook_tid`). The kernel for that
thread holds the warm namespace, so user variables (`mmm`, `results`, `df`, …)
persist across cells. The uploaded dataset sits in `thread_dir(tid)` and
auto-binds as `df`; the live model source is staged into
`garden_loaded_dir(name, "draft", tid)/model.py` (under the thread dir, so the
sandboxed container kernel can import it). This mirrors `_garden_test_sync`.

### Output capture
The cell worker (`_notebook_cell_sync` in `api/main.py`) runs
`KernelManager.execute(code, ctx)` and maps the `ExecuteResult` to JSON-safe
refs using the **same content-addressing `execute_python` uses**:
`workspace.store_plot` for `fig.show()` figures and `tables.publish_tables` for
`show_table(df)` tables. The frontend renders those refs with the existing
`PlotCard` / `TableCard` (which lazily fetch `/plots/{id}` & `/tables/{id}`).

### Starter notebook
When no doc exists, `GET /model-garden/notebook` returns a seeded starter
(`_notebook_starter`): a markdown intro, a data-load cell (uploaded `df` if
present, else `synth.generate_mff('realistic')`), a build+fit cell
(`build_model(spec, src, model_cls=GardenModel)` → `mmm.fit(method='map')`), and
an ROI table + bar-chart cell. It runs **out of the box** on a synthetic world;
the author edits the spec to map their own columns.

### Notebook copilot — diagnose cell errors, get tips + rewrites
The notebook has a built-in **copilot** (toolbar **Copilot** toggle; a right rail
on ≥md screens) — the same Bayesian-modeling + PyMC expert as the editor copilot,
but wired to **debug cell execution**. When a code cell errors, its output shows a
**Diagnose with copilot** button: clicking it opens the rail and auto-asks the
assistant to fix *that* cell. Any code block the assistant returns gets an **Apply
to cell [n]** action (writes back into the originating cell, clearing its stale
error output) — or **Insert as new cell** / **Copy** for free-form answers. When
the fix is to the **model class** (the diagnosis pack often roots a cell error in
`_build_model`), the action becomes **Apply to editor (model source)** instead,
because the kernel imports the model from the editor buffer, not a cell (wired via
`onApplyToEditor` → the page's `applyCode`). Diagnosing a second cell while one is
still streaming queues it (fires when the stream finishes) rather than dropping it.

It streams from the **existing `POST /model-garden/copilot`** SSE endpoint (no new
endpoint, no agent thread — a grounded LLM, same as the editor copilot). The turn
carries an optional `notebook` context (`NotebookCopilotContext`): the failing
cell's `cell_code` + `traceback`, the `dataset_preview` (binds as `df`), the
sibling `other_cells` (variables flow between cells in one kernel), and `is_error`.
`build_copilot_system_prompt(source_code, notebook=…)` then appends a
**cell-diagnosis knowledge pack** (`NOTEBOOK_DIAGNOSIS_KNOWLEDGE` — the real
PyMC/MMM failure modes: `-inf` logp at the start point / support+scaling
mismatches, `NameError` from cell order, `pytensor.scan` slowness, dim/broadcast
errors, missing `beta_*`/`sat_*` registrations, MFF binding) plus a "fix this
cell" instruction grounding the model source + cell + traceback + siblings.

- Frontend: `components/modelGarden/NotebookCopilotPanel.tsx` (the chat),
  `copilotMarkdown.tsx` (shared markdown renderer + `lastCodeBlock`, extracted so
  the component files only export components), `AtelierNotebook.tsx` (toggle +
  `DiagnoseRequest` + apply-to-cell), `NotebookCell.tsx` (the Diagnose button),
  `copilotService.ts` (`streamCopilot(…, notebook?)` + shared `readCopilotStream`).
- Backend: `GardenCopilotRequest.notebook` + `NotebookCopilotContext` in
  `api/main.py`; `build_copilot_system_prompt(notebook=…)` +
  `NOTEBOOK_DIAGNOSIS_KNOWLEDGE` in `agents/garden_authoring.py`. The chat is
  ephemeral (not persisted with the notebook doc).

## Endpoints (api/main.py — registered before the parametric `/model-garden/{name}` routes)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/model-garden/notebook?name=&version=` | persisted doc or seeded starter |
| PUT  | `/model-garden/notebook` | upsert the doc (one `atelier_notebook` artifact) |
| POST | `/model-garden/notebook/dataset?name=&version=` | stage a dataset (multipart) |
| POST | `/model-garden/notebook/cell` | start a cell run → `{job_id}` |
| GET  | `/model-garden/notebook/cell/{job_id}` | poll → `{status, result, error}` |

All are org-scoped (`_garden_org`) and gated at `Role.ANALYST`. Cell code and the
model source execute **only in the session kernel** (the same trust boundary the
compatibility suite uses), so untrusted author source never imports in the host;
in the hosted profile that kernel is the scrubbed-env container sandbox.

## Key files
- Backend: `src/mmm_framework/api/main.py` (the `# Atelier notebook` block + the
  `/model-garden/copilot` endpoint); `agents/garden_authoring.py`
  (`build_copilot_system_prompt`, `NOTEBOOK_DIAGNOSIS_KNOWLEDGE`);
  `agents/tools.py::_garden_copy_source_to_session`.
- Frontend: `pages/ModelGarden/index.tsx` (3rd tab),
  `components/modelGarden/AtelierNotebook.tsx`, `.../NotebookCell.tsx`,
  `.../NotebookCopilotPanel.tsx`, `.../copilotMarkdown.tsx`,
  `api/services/atelierNotebookService.ts`, `api/services/copilotService.ts`,
  `api/hooks/useAtelierNotebook.ts`.
- Reused: `agents/kernels.py` (KernelManager), `agents/workspace.py`
  (`store_plot`/`garden_loaded_dir`), `agents/tables.py` (`publish_tables`),
  `garden/contract.py::find_garden_class`, `agents/fitting.py::build_model`,
  `pages/Agent/components/{plots/PlotCard,tables/TableCard}`.

## Tests
`tests/test_atelier_notebook.py`: the cell worker (GardenModel binding,
plot/table/error mapping, setup-error path, cross-cell var persistence), the
cell job + poll endpoint with org-scoping + the analyst gate, the notebook-doc
save/reload/upsert round-trip, and the **copilot diagnosis prompt**
(`TestDiagnosisPrompt`: grounds on cell+traceback+dataset+siblings, error vs
assistive framing, context truncation, `GardenCopilotRequest.notebook` parsing).
Uses the default in-process kernel (no extra service) and avoids a real PyMC fit,
so the suite is fast.
