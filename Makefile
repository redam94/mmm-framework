.PHONY: tests fast_tests slow_tests format run-api run-ui run-app \
        kernel-lock kernel-image kernel-verify kernel-push

format:
	uvx black src tests examples api app

tests:
	uv run pytest tests/ --cov=mmm_framework -n logical

fast_tests:
	uv run pytest tests/ --cov=mmm_framework -n logical -m 'not slow'

slow_tests:
	uv run pytest tests/ --cov=mmm_framework -n logical -m slow

run-api:
	uv run uvicorn src.mmm_framework.api.main:app --host 0.0.0.0 --port 8000 --reload

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

kernel-lock:                     ## refresh the pinned dependency closure
	uv export --frozen --no-emit-project > deploy/kernel/requirements.lock

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