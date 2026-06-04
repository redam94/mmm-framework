#!/usr/bin/env python
"""Phase 4b load-test harness for the hosted agent API.

Opens N concurrent ``/chat`` SSE sessions against a deployed API, drives a small
fit in each, and reports first-event / completion latency percentiles against a
stated concurrency SLO. This is the §4b "load test to a stated concurrency SLO"
driver — it is **runnable code**, but needs a deployed target to produce numbers
(there is no cluster on the dev box).

Usage:
    MMM_LOADTEST_URL=https://api.example.com \\
    MMM_LOADTEST_CONCURRENCY=20 \\
    uv run python deploy/loadtest/chat_load.py

Each virtual user POSTs to ``/chat`` and consumes the SSE stream, timing:
  - time-to-first-event (first SSE chunk — proxies first-cell / kernel-warm), and
  - time-to-done (the ``[DONE]`` sentinel).
A session must own a server-minted thread (hosted mode), so each user first
POSTs ``/sessions``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time


async def _one_user(client, base: str, project_id: str, idx: int) -> dict:
    import httpx  # local import so the file imports without httpx at lint time

    t0 = time.monotonic()
    result = {"ok": False, "first_event_s": None, "done_s": None, "error": None}
    try:
        # server-minted session (hosted mode requires it)
        r = await client.post(
            f"{base}/sessions", json={"name": f"load-{idx}", "project_id": project_id}
        )
        r.raise_for_status()
        thread_id = r.json()["thread_id"]

        msg = (
            "Generate 30 weeks of synthetic MMM data, then fit a quick model "
            "(chains=1, draws=50, tune=50) and report ROI."
        )
        first_event = None
        async with client.stream(
            "POST", f"{base}/chat", json={"message": msg, "thread_id": thread_id}
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if first_event is None:
                    first_event = time.monotonic()
                    result["first_event_s"] = first_event - t0
                if line.strip() == "data: [DONE]":
                    break
        result["done_s"] = time.monotonic() - t0
        result["ok"] = True
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    return result


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url", default=os.environ.get("MMM_LOADTEST_URL", "http://localhost:8000")
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("MMM_LOADTEST_CONCURRENCY", "10")),
    )
    ap.add_argument(
        "--slo-first-event-s",
        type=float,
        default=float(os.environ.get("MMM_LOADTEST_SLO_FIRST_EVENT_S", "3.0")),
    )
    args = ap.parse_args()

    try:
        import httpx
    except ImportError:
        print("httpx required: uv run python deploy/loadtest/chat_load.py")
        return 2

    base = args.url.rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        proj = await client.post(f"{base}/projects", json={"name": "loadtest"})
        proj.raise_for_status()
        pid = proj.json()["project_id"]
        print(f"driving {args.concurrency} concurrent /chat sessions at {base} ...")
        results = await asyncio.gather(
            *[_one_user(client, base, pid, i) for i in range(args.concurrency)]
        )

    ok = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    print(f"\n{len(ok)}/{len(results)} sessions ok; {len(fails)} failed")
    for r in fails[:5]:
        print("  fail:", r["error"])
    if ok:
        fe = sorted(r["first_event_s"] for r in ok if r["first_event_s"] is not None)
        dn = sorted(r["done_s"] for r in ok if r["done_s"] is not None)

        def pct(xs, p):
            return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")

        print(
            f"first-event  p50={pct(fe, 0.5):.2f}s  p95={pct(fe, 0.95):.2f}s  "
            f"max={max(fe):.2f}s"
        )
        print(
            f"completion   p50={pct(dn, 0.5):.2f}s  p95={pct(dn, 0.95):.2f}s  "
            f"max={max(dn):.2f}s"
        )
        p95_fe = pct(fe, 0.95)
        slo_ok = p95_fe <= args.slo_first_event_s and not fails
        print(
            f"SLO p95 first-event <= {args.slo_first_event_s}s: "
            f"{'PASS' if slo_ok else 'FAIL'} (p95={p95_fe:.2f}s)"
        )
        return 0 if slo_ok else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
