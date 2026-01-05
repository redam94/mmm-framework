.PHONY: tests fast_tests slow_tests format

format:
	uvx black src tests examples api app

tests:
	uv run pytest tests/ --cov=mmm_framework -n logical

fast_tests:
	uv run pytest tests/ --cov=mmm_framework -n logical -m 'not slow'

slow_tests:
	uv run pytest tests/ --cov=mmm_framework -n logical -m slow