API Contracts & Stability
=========================

From **v1.0.0** (2026-07-24) the project follows `semantic versioning
<https://semver.org>`_ strictly. The frozen surfaces:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Surface
     - What is frozen
   * - Python API
     - The documented import surface of the core package (this reference):
       ``BayesianMMM``, ``MFFLoader``/``PanelDataset``, ``Dataset``,
       builders, ``MMMSerializer``, estimands, reporting entry points,
       ``mmm_framework.platform`` services, ``mmm_framework.agents.fitting``.
   * - Config enum values
     - Serialized into specs and saved configs — e.g. ``FitMethod`` (``nuts``,
       ``map``, ``advi``, ``fullrank_advi``, ``pathfinder``, ``laplace``,
       ``smc``), ``SaturationType``, ``AdstockType``. Values are never
       removed or renamed outside a major version.
   * - Model spec keys
     - The consumed-paths registry in ``mmm_framework.agents.fitting``
       (``unconsumed_spec_path`` / ``unconsumed_prior_path``) is the
       authoritative write-side contract: every consumed key is registered,
       every registered key is consumed.
   * - Data formats
     - The MFF long format (8 required columns, forgiving value parsing,
       cadence rules) and the role-tagged ``Dataset`` schema
       (``schema_version 1.0``).
   * - Persisted models
     - The ``MMMSerializer`` directory layout (``format_version 1.1``):
       ``metadata.json`` self-description, ``configs.json``, scaling params,
       ArviZ trace. Loads are gated on the format major version.
   * - Estimand schema
     - ``Estimand`` (schema 1.0) and the built-in names
       (``contribution_roi``, ``counterfactual_roi``, ``marginal_roas``,
       ``contribution``, …) — referenced by string in specs and saved models.
   * - REST API
     - The (method, path) set of ``mmm-framework-server`` (202 operations),
       snapshot-pinned; removals/renames are major-version events.

Modules flagged **Experimental** (``mmm_extensions``, ``continuous_learning``,
``garden``/Atelier, ``dag_model_builder``, non-MMM families) may still change
in minor releases — see the `stability grid
<https://redam94.github.io/mmm-framework/changelog.html#stability>`_.

The CI gates
------------

Every contract is enforced by a test — a breaking change fails the build
rather than shipping silently:

* ``tests/test_api_contracts.py`` — REST route snapshot
  (``tests/contracts/rest_routes.json``), public Python symbols, consumed
  spec/prior paths, frozen enum values, built-in estimand names, and the
  user-roster projection (auth columns never leak).
* ``tests/test_lean_imports.py`` — the lean-core invariant: the whole core
  import surface works with the web/LLM stacks import-blocked.
* ``tests/test_estimands.py`` — bit-stability of estimand arithmetic against
  the pre-1.0 legacy implementations.
* ``tests/test_docs_snippets.py`` — every Python snippet on the docs site
  references real APIs.

Full contract detail
--------------------

The complete, field-by-field contract documentation lives on the docs site:

* `API Contracts (v1.0) <https://redam94.github.io/mmm-framework/api-contracts.html>`_
  — spec keys, prior paths, MFF/Dataset fields, persisted formats, sessions
  store, estimand schema.
* `REST API reference <https://redam94.github.io/mmm-framework/rest-api.html>`_
  — every endpoint with parameters and response shapes (machine-readable
  `openapi.json <https://redam94.github.io/mmm-framework/shared/openapi.json>`_).
* `Versioning & changelog <https://redam94.github.io/mmm-framework/changelog.html>`_.
