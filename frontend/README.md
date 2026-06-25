# MMM Framework — React UI

The supported web frontend for the MMM Framework (Vite + React + TypeScript +
Tailwind 4 + Zustand). It talks to the **MMM Agent API**
(`mmm_framework.api.main:app`) and surfaces the full measurement loop: Program,
Experiments, Performance, the Agent workspace, the Atelier / Model Garden,
Knowledge, Team, and Admin.

## Develop

```bash
npm install            # first run only
npm run dev            # Vite dev server on http://localhost:5173
```

In dev, the Vite server proxies `/api/*` to the backend (default
`http://localhost:8000`). Start the agent API first:

```bash
# from the repository root
uv run uvicorn mmm_framework.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Configuration

- `MMM_API_PROXY_TARGET` — backend the dev-server proxy forwards `/api/*` to
  (default `http://localhost:8000`). Useful when the API runs on another host/port.
- `VITE_API_URL` — set an absolute API URL to bypass the dev proxy (e.g. when
  tunnelling). Leave unset for normal local development. See `.env.example`.

## Build

```bash
npm run build          # type-check + production build to dist/
npm run preview        # preview the production build locally
```

## Notes

- Design tokens live in `src/theme/tokens.css` (Tailwind 4 `@theme`); the
  `tailwind.config.js` is inert under Tailwind 4.
- The legacy Streamlit UI in `../app/` is deprecated; this is the canonical UI.
- See the repository root [`README.md`](../README.md) for the full Quick Start.
