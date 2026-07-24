"""MMM Framework API server (FastAPI).

The web application layer for the MMM Framework — separated from the core
``mmm-framework`` package so that library users (notebooks, analysis scripts)
never install the web/LLM stack. Run it with:

    uv run uvicorn mmm_framework_server.main:app --host 0.0.0.0 --port 8000

Depends on ``mmm-framework[agents]`` plus fastapi/uvicorn; the shared
persistence layer lives in ``mmm_framework.platform`` (core package) so the
agents and seed scripts work without this server installed.
"""

__version__ = "1.0.0"
