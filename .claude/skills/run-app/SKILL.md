---
name: run-app
description: Launch and drive the MMM Framework app — the FastAPI agent API (src/mmm_framework/api, uvicorn :8000) plus the React/Vite frontend (frontend/, :5173) — and walk through it in a real browser. The repo's verified launch recipe; use it WHENEVER the user asks to run, start, launch, serve, preview, or open the app / dev server / frontend / backend / UI, or to walk through, drive, screenshot, smoke-test, or verify a change in the running app — especially the Atelier / Model Garden page. Encodes the working ports, the dev auth bypass, the Python-Playwright driver, and the by-port shutdown.
---

# Run the MMM Framework app

Two processes: the **agent API** (FastAPI, sqlite-backed) and the **React/Vite
frontend**, which proxies `/api/*` to the backend so only `:5173` needs to be
reachable. The backend boots clean on sqlite — **no Redis or LLM key needed** for
the Model Garden / Atelier surface (and most read paths).

## 1. Launch both servers (background)

From the repo root:

```bash
# Backend — agent API on :8000 (isolated workspace; in-process kernel)
MMM_AGENT_WORKSPACE=/tmp/mmm_ws MMM_AGENT_KERNEL=inprocess \
  uv run uvicorn mmm_framework.api.main:app --port 8000 --log-level warning
# Frontend — Vite dev server on :5173 (proxies /api -> :8000)
cd frontend && npm run dev
```

Run each in the background (the harness's `run_in_background`, or append `&`).
`npm install` in `frontend/` first if `node_modules` is missing.

## 2. Wait until ready

Poll all three before driving — the Vite proxy path (`/api/health`) confirms the
frontend can actually reach the backend:

```bash
for i in $(seq 1 40); do
  be=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
  fe=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5173/)
  px=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5173/api/health)
  [ "$be$fe$px" = "200200200" ] && { echo READY; break; }
  sleep 2
done
```

A quick API smoke (dev principal, no auth): `curl -s http://localhost:8000/model-garden` → `{"models":[...]}`.

## 3. Drive it — don't just launch it

Launching only proves the entrypoint resolves. Drive the page a user would see
and **look at the screenshot**. A bundled Playwright driver walks the Atelier
through its full lifecycle (author → register → compatibility test → publish):

```bash
/opt/homebrew/Caskroom/mambaforge/base/bin/python \
  .claude/skills/run-app/scripts/atelier_walkthrough.py
# screenshots land in /tmp/atelier_shots/ (override with ATELIER_SHOTS)
```

Then **Read the screenshots** (`/tmp/atelier_shots/01-atelier.png` … `05-published.png`).

### Browser-driver setup (important)

- **Node Playwright is NOT installed.** The `playwright` on PATH is mambaforge's
  **Python** Playwright with cached Chromium — use
  `/opt/homebrew/Caskroom/mambaforge/base/bin/python` and
  `from playwright.sync_api import sync_playwright`. `p.chromium.launch(headless=True)`
  works out of the box. (System `Google Chrome.app` exists too — fall back via
  `channel="chrome"` if the cached Chromium is missing.)
- **Auth bypass (dev):** the frontend `ProtectedRoute` treats a stored
  `mmm_api_key` as authenticated. Seed it before navigating; the garden endpoints
  ignore the value (dev principal):
  ```python
  ctx.add_init_script(
      "window.localStorage.setItem('mmm_api_key','demo-key');"
      "window.localStorage.setItem('mmm_model_name','claude-sonnet-4-6');")
  ```

To drive a different page, adapt the script — the launch + readiness + auth
bypass above are the reusable parts.

## 4. Stop the servers

```bash
bash .claude/skills/run-app/scripts/stop.sh   # kills by PORT (8000, 5173)
```

## Gotchas (learned the hard way)

- **Shutdown by PORT, not name.** `uv run uvicorn` shows up as `python3.1`, so
  `pkill -f uvicorn` misses it. Use `lsof -ti tcp:8000` / `tcp:5173` (that's what
  `stop.sh` does).
- **Compatibility test latency.** "Run compatibility test" fits a model
  in-process (~60–120s, more under load). Wait up to ~240s for the
  `Compatible`/`Not compatible` verdict; don't treat a slow test as a failure —
  check `GET /model-garden/<name>/<version>` for `status: tested`.
- **Playwright `name="Publish"` is ambiguous** once any registry row shows a
  "Published" chip (the row's accessible name contains "Published"). Use
  `get_by_role("button", name="Publish", exact=True)`.
- **Monaco loads from a CDN at runtime by default** — fine for online dev; an
  offline/CSP-restricted environment needs `monaco-editor` self-hosted via
  `loader.config`.
- **Override the proxy target** with `MMM_API_PROXY_TARGET` if the backend isn't
  on `localhost:8000`.
