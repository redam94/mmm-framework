"""Platform services: persistent workbench state shared by the agents and the API server.

This subpackage is the dependency-light persistence/service layer that used to
live in ``mmm_framework.api`` next to the FastAPI app. The web app itself moved
to the separate ``mmm-framework-server`` package (``server/`` in the repo);
what remains here is business state, not web routing:

- :mod:`~mmm_framework.platform.sessions` — the sessions store (SQLite):
  orgs/users, projects, chat sessions, model runs, artifacts, the experiment
  lifecycle registry, preferences/branding.
- :mod:`~mmm_framework.platform.history` — per-run metric persistence.
- :mod:`~mmm_framework.platform.runs`, :mod:`~mmm_framework.platform.pacing`,
  :mod:`~mmm_framework.platform.scorecard`,
  :mod:`~mmm_framework.platform.triangulation`,
  :mod:`~mmm_framework.platform.estimands`,
  :mod:`~mmm_framework.platform.portfolio_benchmark` — read-side services over
  that state.
- :mod:`~mmm_framework.platform.backfill`, :mod:`~mmm_framework.platform.backup`,
  :mod:`~mmm_framework.platform.connection_sync`,
  :mod:`~mmm_framework.platform.onboarding`,
  :mod:`~mmm_framework.platform.observability` — maintenance / integration
  helpers.

Nothing in this subpackage imports FastAPI or the LLM stack at module level;
it must stay importable in a lean ``pip install mmm-framework`` environment.

Note: ``platform`` shadows nothing for external code — absolute imports mean
``import platform`` anywhere still resolves to the stdlib module.
"""
