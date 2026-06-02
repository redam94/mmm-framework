# Agent LLM / Model Configuration

The MMM **agent** (the LangGraph chat assistant served by
`mmm_framework.api.main` and the React UI) talks to an LLM that you choose via a
**model configuration file**. The same file works for direct API-key providers
and for **Google Vertex AI**, where authentication uses **Application Default
Credentials (ADC)** — so on a GCP VM the attached service account is used with
no API key anywhere.

The configuration layer lives in
[`src/mmm_framework/agents/llm.py`](../src/mmm_framework/agents/llm.py):
`ModelConfig`, `load_model_config()`, `build_llm()`, and `describe_active_config()`.

## Quick start

```bash
# 1. Copy the annotated template (the real file is git-ignored)
cp config/model_config.example.yaml config/model_config.yaml

# 2. Edit config/model_config.yaml — set provider / model / project / location

# 3. Run the agent server (it reads the config at startup)
cd src/mmm_framework && uvicorn mmm_framework.api.main:app --reload
# or the standalone CLI example:
uv run python examples/ex_agent_workflow.py
# or the Vertex/ADC smoke test:
uv run python examples/ex_vertex_agent.py
```

## Providers

| `provider`         | Backend                              | Auth                              |
|--------------------|--------------------------------------|-----------------------------------|
| `vertex_anthropic` | Claude on Vertex AI (Model Garden)   | ADC (or `credentials_path`)       |
| `vertex_gemini`    | Gemini on Vertex AI                  | ADC (or `credentials_path`)       |
| `anthropic`        | Anthropic API (direct)               | `api_key` / `ANTHROPIC_API_KEY`   |
| `openai`           | OpenAI API (direct)                  | `api_key` / `OPENAI_API_KEY`      |
| `google_genai`     | Gemini Developer API (direct)        | `api_key` / `GOOGLE_API_KEY`      |

> Vertex Gemini uses the non-deprecated `ChatGoogleGenerativeAI(vertexai=True)`
> path; Vertex Claude uses `ChatAnthropicVertex` from `langchain-google-vertexai`.

## Example: Vertex AI on a GCP VM (ADC)

```yaml
# config/model_config.yaml
provider: vertex_anthropic
model:    claude-sonnet-4-5@20250929   # copy the EXACT id from Model Garden
project:                               # blank => use the VM's metadata project
location: us-east5                     # a region that actually serves the model
credentials_path:                      # blank => Application Default Credentials
temperature: 0
max_tokens: 8192
```

On the VM the service account needs the **`roles/aiplatform.user`** role. No
API key is required: `google.auth.default()` resolves the VM's credentials.
To reproduce locally, run `gcloud auth application-default login`.

> **Set `project` explicitly when testing locally.** A blank `project:` relies
> on `google.auth.default()` returning a default project — which works from GCE
> VM metadata, but `gcloud auth application-default login` often yields *no*
> default project. If you see a "project was not passed" error locally, set
> `project:` (or `MMM_LLM_PROJECT`) to your GCP project id.

> **Model ids & regions:** Vertex Model Garden ids may differ from the direct
> Anthropic names and can carry an `@version` suffix. Claude is served only in a
> subset of regions (e.g. `us-east5`, `europe-west1`). Confirm both in your
> Vertex AI Model Garden console — the value above is only a placeholder.

## Configuration precedence

For each field: **`MMM_LLM_*` env var → config file → built-in default.**

- Config search order: `$MMM_MODEL_CONFIG` → `./model_config.yaml` →
  `./config/model_config.yaml`. If none is found, the agent falls back to a
  direct Anthropic Claude default (so existing deployments keep working).
- Per-field env overrides (handy on a VM): `MMM_LLM_PROVIDER`, `MMM_LLM_MODEL`,
  `MMM_LLM_PROJECT`, `MMM_LLM_LOCATION`, `MMM_LLM_MAX_TOKENS`, `MMM_LLM_TEMPERATURE`,
  `MMM_LLM_CREDENTIALS_PATH`, `MMM_LLM_API_KEY`.

## Per-request headers vs. server config

The React UI sends `X-Model-Name` / `X-API-Key` headers. How `build_llm`
resolves them depends on the **server's** configured provider:

- **Vertex provider** → server ADC config is authoritative. A client `X-API-Key`
  is ignored (a VM user may type a dummy key just to pass the login gate). An
  `X-Model-Name` is honored only if it's the same model family (e.g. swapping
  one Claude id for another).
- **Direct provider** → a client `X-API-Key` / `X-Model-Name` selects a direct
  provider inferred from the model name (the multi-provider key-entry flow).

A blank or sentinel key (`""`, `server-managed`) is always normalised to "no
client key", so it can never clobber the server's own credential.

## Selecting a Vertex model

You don't have to hard-code a single `model:` — the agent can discover the
models available to your project/region and let you pick one.

- **`GET /vertex-models`** returns the selectable models for the configured (or
  `?project=&location=`-overridden) project/region:
  `{ project, location, active_model, models: [{id, provider, family, display_name, source, location}] }`.
  Results are cached for ~5 minutes (discovery is a network call).
- **Gemini** models are discovered **live** (`source: "live"`), scoped to the
  configured region, so you're never offered a model the region can't serve.
- **Claude** is best-effort (`source: "catalog"`, the global Model Garden
  catalog). Because Vertex Claude ids carry rotating `@version` suffixes and the
  catalog ignores per-project enablement, **a free-text field is always
  available** — paste the exact id from your Vertex console.
- Add your own convenience ids with **`extra_models`** in the config
  (`source: "config"`); these are shown as-is, never guessed for you.

In the React **Login** page (server-managed/Vertex mode) this surfaces as a
**Model** dropdown plus an "Or enter a model id" field. The chosen id rides as
`X-Model-Name`; `build_llm` routes it to the matching Vertex backend by family —
a Gemini id → `vertex_gemini`, a Claude id → `vertex_anthropic` — keeping the
same project/region/ADC and **never** falling back to a key-based provider.

> Cross-family selection keeps the configured `location`. If you pick a Claude
> id while configured for a Gemini-only region (e.g. `us-central1`), set a
> Claude-serving region (e.g. `us-east5`) in the config or `MMM_LLM_LOCATION`.

Programmatic use:

```python
from mmm_framework.agents import list_vertex_models, load_model_config
for m in list_vertex_models(load_model_config()):
    print(m["provider"], m["id"], f"({m['source']})")
```

## Frontend behaviour

`GET /model-config` returns a non-secret summary (provider, model, region,
`uses_adc`, `requires_api_key`). When `requires_api_key` is `false` (Vertex/ADC
or a server-side env key), the **Login** page shows a "Server-managed
credentials" panel and a **Continue** button — no key entry — plus the Vertex
**Model** picker described above. Otherwise it shows the usual provider selector
+ API-key field.
