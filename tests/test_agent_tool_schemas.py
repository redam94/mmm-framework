"""Guard: every bound agent tool must produce a Google-Gemini-valid schema.

Gemini's ``generateContent`` validates each tool's ``function_declaration``
schema *server-side* with rules that the local
``convert_to_genai_function_declarations`` does NOT fully enforce, so a bad
parameter type only blows up at runtime (a 400 that breaks the whole agent,
since one invalid tool rejects the entire request). Two rules bite in practice:

1. **No untyped / ``Any`` parameters** — a bare ``Any`` annotation yields an
   empty ``{}`` schema, which Gemini rejects as a null property schema.
2. **Every ``array`` must declare ``items``** — a bare ``list`` annotation
   yields ``{"type": "array"}`` with no element type, which Gemini rejects with
   "items: missing field".

This test replays both rules over the actual bound tool sets using only each
tool's LLM-facing JSON schema (no Gemini dependency, no network), so a newly
added ``param: Any`` or ``param: list`` fails here instead of in production.
Fix by giving the parameter a concrete type, e.g. ``list[float]`` /
``Union[str, int, float, bool]`` (see ``update_model_setting`` /
``record_assumption`` for the pattern).
"""

from __future__ import annotations

import pytest

from mmm_framework.agents.tools import EXPERT_TOOLS, ORCHESTRATOR_TOOLS, TOOLS


def _all_bound_tools():
    """Every tool that can be bound to the LLM, de-duplicated by name."""
    seen: dict[str, object] = {}
    for lst in (TOOLS, EXPERT_TOOLS, ORCHESTRATOR_TOOLS):
        for t in lst:
            seen.setdefault(t.name, t)
    return list(seen.values())


def _gemini_schema_issues(schema, path: str) -> list[str]:
    """Return Gemini-incompatibility issues in a JSON schema fragment.

    ``path`` carries a ``.`` once we are inside a tool's parameters, which is how
    we tell a real (LLM-supplied) parameter from the root object.
    """
    issues: list[str] = []
    if not isinstance(schema, dict):
        return issues

    stype = schema.get("type")
    has_typing = stype is not None or any(
        k in schema for k in ("anyOf", "allOf", "oneOf", "$ref", "enum", "const")
    )
    is_parameter = "." in path

    # Rule 2: an array must declare its element type.
    if stype == "array" and not schema.get("items"):
        issues.append(f"{path}: array parameter missing 'items' (element type)")

    # Rule 1: a parameter with no type at all (bare ``Any``) is rejected.
    if is_parameter and not has_typing and "properties" not in schema:
        issues.append(f"{path}: untyped (Any) parameter — give it a concrete type")

    for key, sub in (schema.get("properties") or {}).items():
        issues += _gemini_schema_issues(sub, f"{path}.{key}")
    if isinstance(schema.get("items"), dict):
        issues += _gemini_schema_issues(schema["items"], f"{path}[item]")
    for i, member in enumerate(schema.get("anyOf") or schema.get("oneOf") or []):
        issues += _gemini_schema_issues(member, f"{path}|union{i}")
    return issues


def _tool_issues(tool) -> list[str]:
    schema = tool.tool_call_schema.model_json_schema()
    issues = _gemini_schema_issues(schema, tool.name)
    # Nested pydantic models surface as $defs; scan those too.
    for name, sub in (schema.get("$defs") or {}).items():
        issues += _gemini_schema_issues(sub, f"{tool.name}#defs.{name}")
    return issues


@pytest.mark.parametrize("tool", _all_bound_tools(), ids=lambda t: t.name)
def test_tool_schema_is_gemini_valid(tool):
    issues = _tool_issues(tool)
    assert not issues, "Gemini-invalid tool schema:\n- " + "\n- ".join(issues)


def test_no_bound_tool_has_a_gemini_invalid_schema():
    """Aggregate view: lists every offender at once (clearer than the first
    per-tool failure when several regress together)."""
    all_issues = [msg for tool in _all_bound_tools() for msg in _tool_issues(tool)]
    assert not all_issues, "Gemini-invalid tool schemas:\n- " + "\n- ".join(all_issues)


def test_converter_accepts_full_tool_list():
    """Secondary check via the real converter (catches the null-schema class the
    converter itself raises on). Skipped if the Gemini integration isn't
    installed; the schema rules above are the dependency-free source of truth."""
    fu = pytest.importorskip("langchain_google_genai._function_utils")
    # Raises if any single tool can't be converted (e.g. an ``Any`` property).
    fu.convert_to_genai_function_declarations(list(TOOLS))
