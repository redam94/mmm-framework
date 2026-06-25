# Container image for the MMM Agent API (mmm_framework.api.main:app).
#
#   docker build -f deploy/api/Dockerfile -t mmm-api .
#   docker run -p 8000:8000 \
#       -e MMM_AUTH_ENABLED=1 -e MMM_AUTH_SECRET=... \
#       -e MMM_AGENT_WORKSPACE=/data/workspace \
#       -v mmm-data:/data mmm-api
#
# NOTE: the agent API runs model fits in its session kernel — no Redis/worker is
# required for the canonical path. For the container kernel sandbox (hosted),
# also build deploy/kernel/Containerfile and provide a container runtime.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# System libs commonly needed by the scientific stack (numpy/pymc/pytensor).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency manifests first for layer caching.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

# Install locked runtime deps (no dev group). Editable install of the package.
RUN uv sync --frozen --no-dev

# Persisted state (DB + agent workspace) lives on a mounted volume.
RUN useradd -m appuser && mkdir -p /data && chown -R appuser /app /data
USER appuser
ENV MMM_AGENT_WORKSPACE=/data/workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "uvicorn", "mmm_framework.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
