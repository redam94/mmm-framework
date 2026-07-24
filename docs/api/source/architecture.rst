Architecture
============

At v1.0.0 the framework is a uv workspace with a **lean modeling core** and
optional application layers:

* ``pip install mmm-framework`` — business logic only: the full Bayesian MMM
  (build / fit / analyze / report / validate) with **no web framework and no
  LLM stack**.
* ``pip install "mmm-framework[agents]"`` — adds the LangGraph oracle agent
  (langgraph, langchain-*, httpx, pypdf, python-docx). The langchain-free
  agents service modules (``fitting``, ``workspace``, ``model_ops``, …) import
  without the extra via the lazy ``agents/__init__``.
* ``mmm-framework-server`` — the FastAPI application, a separate workspace
  package in the repository's ``server/`` directory. Run it with
  ``uvicorn mmm_framework_server.main:app``.
* The former ``mmm_framework.api`` service modules (sessions store, run
  history, pacing, scorecards, …) live in :mod:`mmm_framework.platform` —
  dependency-light, shared by the agents and the server.

The lean-core invariant is pinned in CI by ``tests/test_lean_imports.py``.

Package & dependency structure
------------------------------

Dependencies point strictly inward — nothing in the core depends on the agent
stack or the server:

.. mermaid::

   graph TD
       FE["frontend/<br/><small>React + Vite UI</small>"]
       SRV["mmm-framework-server<br/><small>FastAPI app · repo server/ · package mmm_framework_server</small>"]
       AGENTS["mmm-framework with the agents extra<br/><small>LangGraph oracle · langgraph, langchain, httpx, pypdf, python-docx</small>"]
       CORE["mmm-framework — the core<br/><small>model · transforms · config + builders · reporting · validation<br/>planning · estimands · diagnostics · eda · synth · platform · auth core</small>"]
       KERNEL["session kernel image<br/><small>lean core + ipykernel + python-pptx + matplotlib<br/>no LLM or web stack</small>"]
       GCP["gcp extra<br/><small>GCS + BigQuery</small>"]
       S3["s3 extra<br/><small>S3 object store</small>"]

       FE -->|"HTTP + SSE"| SRV
       SRV -->|"depends on"| AGENTS
       AGENTS -->|"depends on"| CORE
       KERNEL -->|"built from"| CORE
       GCP -.->|"optional extra of"| CORE
       S3 -.->|"optional extra of"| CORE

Model-building data flow
------------------------

.. mermaid::

   graph LR
       subgraph IN["Data in"]
           MFF["MFF CSV<br/><small>long format</small>"]
           WIDE["wide CSV"]
           EX["load_example()"]
       end
       LOAD["MFFLoader<br/><small>validation · cadence checks</small>"]
       PANEL["PanelDataset"]
       MODEL["BayesianMMM<br/><small>adstock → saturation → trend, seasonality, controls → likelihood<br/>fit: NUTS / numpyro · approximate MAP, ADVI, pathfinder, Laplace · SMC exact</small>"]
       RES["MMMResults<br/><small>posterior · diagnostics · estimands</small>"]
       subgraph OUT["Fan-out"]
           REP["MMMReportGenerator<br/><small>Augur HTML readout</small>"]
           IREP["InteractiveReportGenerator<br/><small>recompute-in-browser report</small>"]
           SER["MMMSerializer<br/><small>persisted model directory</small>"]
           ANA["analysis + estimands<br/><small>ROI · marginal ROAS · counterfactuals</small>"]
           VAL["validation<br/><small>backtests · SBC · refutation</small>"]
       end

       MFF --> LOAD
       WIDE --> LOAD
       EX --> LOAD
       LOAD --> PANEL
       PANEL --> MODEL
       MODEL --> RES
       RES --> REP
       RES --> IREP
       RES --> SER
       RES --> ANA
       RES --> VAL

Application request flow
------------------------

How a question in the web UI becomes a fitted model and rendered artifacts
(all server-side components are in the separate ``mmm-framework-server``
package; the persistence layer is core :mod:`mmm_framework.platform`):

.. mermaid::

   sequenceDiagram
       autonumber
       actor User
       participant UI as React UI (port 5173)
       participant API as FastAPI server (port 8000)
       participant Agent as LangGraph agent
       participant Tools as Tools layer
       participant Kernel as Session kernel
       participant Store as platform.sessions (SQLite)

       User->>UI: Ask a question in the Oracle
       UI->>API: POST /chat — SSE stream opens
       API->>Agent: Resume thread from checkpoint
       Note over Agent: Workflow-step guard —<br/>one pipeline milestone per turn
       Agent->>Tools: Tool call (EDA, build, fit, report, ...)
       Tools->>Kernel: execute_python — fits run in-kernel
       Kernel-->>Tools: stdout, plots, tables
       Tools->>Store: Persist artifacts, experiments, run metrics
       Store-->>API: Session state and artifact refs
       API-->>UI: SSE events — messages, plots, tables, artifacts
       UI-->>User: Rendered in the workspace tabs

The measurement loop
--------------------

The product's core cycle — fit, prioritize experiments by information value,
run them, fold the results back in as calibration, allocate, and re-test when
information decays:

.. mermaid::

   graph LR
       T0["T₀ · Fit<br/><small>Bayesian MMM with calibration</small>"]
       T1["T₁ · Prioritize<br/><small>EIG / EVOI per channel</small>"]
       T2["T₂ · Design + pre-register<br/><small>geo lift · DiD · flighting<br/>draft → planned → running → completed → calibrated</small>"]
       T3["T₃ · Calibrated refit<br/><small>experiment likelihoods in-graph</small>"]
       T4["T₄ · Allocate<br/><small>budget optimizer</small>"]
       T5["T₅ · Re-evaluate<br/><small>information decay triggers re-tests</small>"]

       T0 --> T1
       T1 --> T2
       T2 --> T3
       T3 --> T4
       T4 --> T5
       T5 -->|"new priorities"| T1

Where things live
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Subsystem
     - Import path
   * - Core modeling
     - :mod:`mmm_framework.model`, :mod:`mmm_framework.transforms`,
       ``mmm_framework.builders``, ``mmm_framework.config``
   * - Data loading
     - :mod:`mmm_framework.data_loader` (MFF),
       ``mmm_framework.dataset`` (role-tagged wide tables),
       :mod:`mmm_framework.datasets` (bundled examples)
   * - Results & prediction
     - ``mmm_framework.model.results``
   * - Estimands
     - :mod:`mmm_framework.estimands`
   * - Reporting
     - :mod:`mmm_framework.reporting`
   * - Validation & diagnostics
     - :mod:`mmm_framework.validation`, :mod:`mmm_framework.diagnostics`,
       :mod:`mmm_framework.calibration`
   * - Experiment planning
     - :mod:`mmm_framework.planning`,
       :mod:`mmm_framework.continuous_learning`, :mod:`mmm_framework.ltv`
   * - Platform services (sessions store, run history, …)
     - :mod:`mmm_framework.platform`
   * - Agents (optional ``[agents]`` extra)
     - ``mmm_framework.agents``
   * - API server (separate package)
     - ``mmm_framework_server.main``

Further reading
---------------

* `Architecture & flow diagrams (docs site)
  <https://redam94.github.io/mmm-framework/architecture.html>`_ — the same
  diagrams with the full narrative.
* `REST API reference <https://redam94.github.io/mmm-framework/rest-api.html>`_
  — all server endpoints, generated from the live OpenAPI schema.
* :doc:`contracts` — what the v1.0 stability promise covers.
