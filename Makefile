.PHONY: tests fast_tests slow_tests format run-api run-ui run-app

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