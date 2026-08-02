.PHONY: tests fast_tests slow_tests format format_check lint hooks run-api run-ui run-app \
        kernel-lock kernel-image kernel-verify kernel-push

format:                          ## black, via the LOCKED dev dependency
	uv run black src server/src tests examples

format_check:                    ## the black gate CI runs (no writes)
	uv run black --check --diff src server/src tests examples

lint:                            ## the same ruff gate CI runs
	uv run ruff check src server/src

hooks:                           ## install the git pre-commit hook (ruff + black)
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-commit
	@echo "✓ pre-commit hook installed (ruff check + black --check)"

tests:
	uv run pytest tests/ --cov=mmm_framework -n logical

fast_tests:
	uv run pytest tests/ --cov=mmm_framework -n logical -m 'not slow'

slow_tests:
	uv run pytest tests/ --cov=mmm_framework -n logical -m slow

run-api:
	uv run uvicorn mmm_framework_server.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	cd frontend && npm run dev

run-app:
	@echo "Starting both the FastAPI backend and React frontend..."
	$(MAKE) -j2 run-api run-ui

# ── Hardened agent kernel image (enables MMM_AGENT_HOSTED=1) ──────────────────
# See deploy/kernel/README.md for the full ship runbook.
KERNEL_RUNTIME  ?= podman
KERNEL_IMAGE    ?= mmm-kernel:latest
KERNEL_REGISTRY ?=

# Kernel closure = lean core + pinned ipykernel (the `kernel` dependency
# group). --no-dev keeps the web/LLM stack out of the sandbox image;
# --no-emit-workspace keeps the server workspace member (a path dep pip can't
# resolve inside the container) out of the lock.
kernel-lock:                     ## refresh the pinned dependency closure
	uv export --frozen --no-emit-project --no-emit-workspace --no-dev --group kernel > deploy/kernel/requirements.lock

kernel-image:                    ## build the per-session sandbox image
	$(KERNEL_RUNTIME) build -t $(KERNEL_IMAGE) -f deploy/kernel/Containerfile .

kernel-verify:                   ## smoke-test the image under the run-time sandbox flags
	@command -v $(KERNEL_RUNTIME) >/dev/null || { echo "✗ $(KERNEL_RUNTIME) not found"; exit 1; }
	@test -f deploy/kernel/requirements.lock || { echo "✗ deploy/kernel/requirements.lock missing — run 'make kernel-lock'"; exit 1; }
	$(KERNEL_RUNTIME) run --rm --read-only --tmpfs /tmp --network none \
		--cap-drop ALL --user 10001 $(KERNEL_IMAGE) \
		python -c "import mmm_framework, ipykernel; print('kernel image OK:', mmm_framework.__name__)"

kernel-push:                     ## tag + push to KERNEL_REGISTRY
	@test -n "$(KERNEL_REGISTRY)" || { echo "✗ set KERNEL_REGISTRY=registry.example.com/yourorg"; exit 1; }
	$(KERNEL_RUNTIME) tag $(KERNEL_IMAGE) $(KERNEL_REGISTRY)/$(KERNEL_IMAGE)
	$(KERNEL_RUNTIME) push $(KERNEL_REGISTRY)/$(KERNEL_IMAGE)