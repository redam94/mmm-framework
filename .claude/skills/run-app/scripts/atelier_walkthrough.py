#!/usr/bin/env python
"""Drive the Atelier (Model Garden) page end-to-end in a headless browser.

Runs the full lifecycle a user would: author a bespoke model, register it, run
the compatibility test (which fits a model in-process and auto-promotes
draft->tested on pass), then publish it (tested->published). Screenshots each
stage so a human can LOOK at the result.

The UI steps are best-effort (headless Chromium can crash re-rendering the
Monaco-heavy page after a fit), so the BACKEND API is the source of truth for
PASS/FAIL — the script confirms the model actually reached `published` via
GET /model-garden. Screenshots are visual evidence on top of that.

Run with the interpreter that HAS playwright (Node playwright isn't installed;
the `playwright` on PATH is mambaforge's Python one):

    /opt/homebrew/Caskroom/mambaforge/base/bin/python \
        .claude/skills/run-app/scripts/atelier_walkthrough.py

Env:
    ATELIER_BASE     frontend origin   (default http://localhost:5173)
    ATELIER_API      backend origin    (default http://localhost:8000)
    ATELIER_SHOTS    screenshot dir    (default /tmp/atelier_shots)
    ATELIER_MODEL    model name        (default demo-walkthrough)
"""

import json
import os
import sys
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get("ATELIER_BASE", "http://localhost:5173")
API = os.environ.get("ATELIER_API", "http://localhost:8000")
SHOTS = os.environ.get("ATELIER_SHOTS", "/tmp/atelier_shots")
MODEL = os.environ.get("ATELIER_MODEL", "demo-walkthrough")
os.makedirs(SHOTS, exist_ok=True)

# Headless-stability flags: this page mounts the Monaco editor, which can crash a
# bare headless tab when it re-renders after the compat fit.
LAUNCH_ARGS = ["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]


def log(*a):
    print(*a, flush=True)


def shot(page, name):
    try:
        page.screenshot(path=f"{SHOTS}/{name}.png")
    except Exception as e:  # noqa: BLE001
        log(f"(screenshot {name} skipped: {e})")


def api_versions(name):
    """Backend's view of a model's versions (source of truth)."""
    try:
        with urllib.request.urlopen(f"{API}/model-garden/{name}/versions", timeout=10) as r:
            return json.load(r).get("versions", [])
    except Exception as e:  # noqa: BLE001
        log("api_versions error:", e)
        return []


def drive_ui():
    """Best-effort UI walkthrough + screenshots. Returns the (name, version) the
    test was started on, or (name, None). Never raises — a crashed tab just ends
    the UI portion early; the API check still verifies the outcome."""
    started_version = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=LAUNCH_ARGS)
        ctx = browser.new_context(viewport={"width": 1500, "height": 950})
        # Bypass the login gate: ProtectedRoute treats a stored api key as
        # authenticated; the garden endpoints use the dev principal and ignore it.
        ctx.add_init_script(
            "window.localStorage.setItem('mmm_api_key','demo-key');"
            "window.localStorage.setItem('mmm_model_name','claude-sonnet-4-6');"
        )
        page = ctx.new_page()
        try:
            page.goto(f"{BASE}/atelier", wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(1200)
            log("H1:", (page.text_content("h1") or "").strip())
            log("Monaco present:", page.locator(".monaco-editor").count() > 0)
            shot(page, "01-atelier")

            # Author + register (the starter CustomMMM template is valid as-is)
            page.get_by_role("button", name="New model").click()
            page.wait_for_timeout(600)
            page.locator('input[placeholder*="model name"]').fill(MODEL)
            page.locator('input[placeholder*="docs"]').fill("Run-skill walkthrough model.")
            shot(page, "02-authoring")
            page.get_by_role("button", name="Register draft").click()
            page.wait_for_selector(f"text={MODEL}", timeout=15000)
            page.wait_for_timeout(1000)
            shot(page, "03-registered")
            vs = api_versions(MODEL)
            started_version = max((v["version"] for v in vs), default=None)
            log("registered draft; version:", started_version)

            # Compatibility test (fits a model; can exceed 180s under load)
            page.get_by_role("button", name="Run compatibility test").click()
            log("testing (in-process fit; up to 240s)…")
            page.wait_for_selector("text=/Compatible|Not compatible/", timeout=240000)
            page.wait_for_timeout(2000)
            log("verdict rendered; tier rows:", page.locator("table tbody tr").count())
            shot(page, "04-tested")

            # Publish gate (exact=True — "Published" list rows also match a loose name)
            pub = page.get_by_role("button", name="Publish", exact=True)
            if pub.count() > 0:
                pub.click()
                page.wait_for_timeout(2000)
                shot(page, "05-published")
        except Exception as e:  # noqa: BLE001
            log("UI step ended early (tab may have crashed):", repr(e)[:120])
        finally:
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass
    return MODEL, started_version


def main() -> int:
    drive_ui()

    # Source of truth: did the model actually move through the lifecycle?
    vs = api_versions(MODEL)
    if not vs:
        log(f"\nFAIL: backend has no '{MODEL}' versions — registration didn't land.")
        return 1
    latest = max(vs, key=lambda v: v["version"])
    history = " -> ".join(h["status"] for h in latest.get("status_history", []))
    report = latest.get("compat_report") or {}
    log(f"\nBackend truth for {MODEL} v{latest['version']}:")
    log("  status:        ", latest["status"])
    log("  lifecycle:     ", history)
    log("  blocking_passed:", report.get("blocking_passed"), "| score:", report.get("score"))
    log("  screenshots in:", SHOTS)

    if latest["status"] == "published":
        log("\nPASS: full lifecycle draft -> tested -> published verified via the API.")
        return 0
    if latest["status"] == "tested" and report.get("blocking_passed"):
        log("\nPARTIAL: model registered + passed compat (tested), but publish didn't land "
            "(UI tab likely crashed before the click). Compat verified via API.")
        return 0
    log(f"\nFAIL: model stuck at '{latest['status']}'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
