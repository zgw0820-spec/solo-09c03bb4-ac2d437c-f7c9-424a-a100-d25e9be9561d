#!/usr/bin/env python3
"""End-to-end self-check against a running service.

    python3 scripts/selfcheck.py [base-url]

Verifies, over real HTTP:
1. health + two distinct models
2. switching the active model
3. version pinning: requests accepted before a switch keep the old version,
   requests accepted after it use the new one, and no batch mixes versions
4. batching: 8 accepted emails -> batches of at most 4
5. admission cap: 8 global slots shared by sync/async, per-email verdicts
6. deadlines: a slow email times out, its batch-mates still succeed, and the
   late result never flips the timeout back to success
7. early dispatch when the earliest deadline is near
"""

from __future__ import annotations

import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SPAM = "Congratulations! You are a winner. Claim your free prize now."
HAM = "Hi team, attached is the quarterly report for our meeting tomorrow."
CRYPTO = "Limited offer! Invest in crypto today and double your money fast."

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def wait_job(client: httpx.Client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.02)
    return state


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=30.0)

    print("1. health & models")
    health = client.get("/healthz").json()
    check(health["status"] == "ok", "service healthy")
    check({"v1", "v2"} <= set(health["models"]), "both v1 and v2 shipped")
    models = client.get("/api/models").json()["models"]
    sizes = {m["version"]: m["vocabulary_size"] for m in models}
    check(sizes["v1"] != sizes["v2"], f"models differ (v1={sizes['v1']}, v2={sizes['v2']} terms)")

    print("2. model switch")
    switched = client.post("/api/model", json={"version": "v2"}).json()
    check(switched["active"] == "v2", "active model switched to v2")
    check(client.get("/healthz").json()["active_model"] == "v2", "health reflects the switch")
    check(client.post("/api/model", json={"version": "nope"}).status_code == 404, "unknown version rejected")

    print("3. version pinning across a live switch")
    client.post("/api/model", json={"version": "v1"})
    job = client.post(
        "/api/jobs",
        json={
            "label": "pin",
            "emails": [
                {"text": SPAM, "deadline_ms": 900},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 260},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 420},
            ],
        },
    ).json()
    time.sleep(0.12)
    switch = client.post("/api/model", json={"version": "v2"}).json()
    check(switch["event"]["from"] == "v1" and switch["event"]["to"] == "v2", "switched v1 -> v2 mid-queue")
    pinned = wait_job(client, job["id"])
    versions = [e["model_version"] for e in pinned["entries"]]
    check(versions[0] == "v1", "email accepted before the switch kept v1")
    check(all(v == "v2" for v in versions[1:]), "emails accepted after the switch used v2")
    batches = client.get("/api/batches?limit=50").json()["batches"]
    for entry in pinned["entries"]:
        batch = next((b for b in batches if b["batch_id"] == entry["batch_id"]), None)
        if batch:
            expected = entry["model_version"]
            check(batch["model_version"] == expected, f"batch {batch['batch_id']} is pure {expected}")

    print("4. batching limits (8 accepted = the global admission cap)")
    client.post("/api/model", json={"version": "v1"})
    bulk = client.post(
        "/api/jobs",
        json={
            "label": "bulk8",
            "emails": [
                {"text": f"{SPAM} ref {i}", "deadline_ms": 1000}
                for i in range(8)
            ],
        },
    ).json()
    bulk_state = wait_job(client, bulk["id"])
    check(bulk_state["accepted"] == 8, f"all 8 emails admitted (accepted={bulk_state['accepted']})")
    check(bulk_state["rejected"] == 0, "no rejection below the cap")
    check(bulk_state["ok"] == 8, f"all 8 classified (ok={bulk_state['ok']})")
    bulk_ids = {e["batch_id"] for e in bulk_state["entries"]}
    check(len(bulk_ids) == 2, f"8 emails formed 2 batches (got {len(bulk_ids)})")
    sizes = [b["size"] for b in client.get("/api/batches?limit=50").json()["batches"] if b["batch_id"] in bulk_ids]
    check(max(sizes) <= 4 and sorted(sizes) == [4, 4], f"batch sizes {sorted(sizes)}")

    print("5. global admission cap (8) shared by async and sync")
    client.post("/api/model", json={"version": "v1"})
    over = client.post(
        "/api/jobs",
        json={
            "label": "over-cap",
            "emails": [
                {"text": f"{SPAM} over {i}", "deadline_ms": 1000,
                 "simulate_latency_ms": 700}
                for i in range(12)
            ],
        },
    ).json()
    for _ in range(300):
        probed = client.get(f"/api/jobs/{over['id']}").json()
        if probed["pending_admission"] == 0:
            break
        time.sleep(0.01)
    check(probed["accepted"] == 8, f"exactly 8 of 12 admitted (got {probed['accepted']})")
    check(probed["rejected"] == 4, f"4 refused at intake (got {probed['rejected']})")
    rejected_entries = [e for e in probed["entries"] if e["status"] == "rejected"]
    check(
        all(e["reason"] == "admission_overloaded" for e in rejected_entries),
        "refusals report admission_overloaded",
    )
    check(
        all(e["batch_id"] is None and e["score"] is None for e in rejected_entries),
        "refused emails never got a batch or score",
    )

    sync = client.post(
        "/api/classify",
        json={"label": "sync while full", "emails": [{"text": SPAM, "deadline_ms": 1000}]},
    ).json()
    check(
        sync["entries"][0]["status"] == "rejected",
        f"sync submission shares the pool and is refused (got {sync['entries'][0]['status']})",
    )
    admission = client.get("/api/state").json()["admission"]
    check(admission["limit"] == 8, f"limit reported as 8 (got {admission['limit']})")
    check(admission["used"] <= 8, f"pool never oversold (used={admission['used']})")
    check(
        admission["rejected_total"] >= 5,
        f"rejections counted in state (got {admission['rejected_total']})",
    )

    # Wait until every scoring thread has really finished.
    for _ in range(600):
        if client.get("/api/state").json()["admission"]["used"] == 0:
            break
        time.sleep(0.05)
    check(
        client.get("/api/state").json()["admission"]["used"] == 0,
        "all slots released after scoring finished",
    )
    rejected_ids = {e["request_id"] for e in rejected_entries}
    rejected_ids.add(sync["entries"][0]["request_id"])
    leaked = {
        rid
        for batch in client.get("/api/batches?limit=100").json()["batches"]
        for rid in batch["request_ids"]
        if rid in rejected_ids
    }
    check(not leaked, f"no rejected email entered a batch (leaked={sorted(leaked)})")

    print("6. deadline timeout + late result cannot flip it")
    client.post("/api/model", json={"version": "v2"})
    timeout_job = client.post(
        "/api/classify",
        json={
            "label": "batch timeout",
            "emails": [
                {"text": SPAM, "deadline_ms": 100, "simulate_latency_ms": 600},
                {"text": CRYPTO, "deadline_ms": 1000},
            ],
        },
    ).json()
    late = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 100)
    fast = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 1000)
    check(late["status"] == "timeout", "slow email timed out")
    check(late["score"] is None, "timed-out email has no score")
    check(late["late_result_discarded"] is True, "late result explicitly discarded")
    check(fast["status"] == "ok", "batch-mate still succeeded")
    check(late["batch_id"] == fast["batch_id"], "they were in the same batch")
    time.sleep(0.7)
    reread = client.get(f"/api/jobs/{timeout_job['id']}").json()
    again = next(e for e in reread["entries"] if e["deadline_ms"] == 100)
    check(again["status"] == "timeout", "timeout still a timeout after the late thread finished")

    print("7. early dispatch when the earliest deadline is near")
    near = client.post(
        "/api/jobs",
        json={"label": "near", "emails": [{"text": SPAM, "deadline_ms": 150}]},
    ).json()
    near_state = wait_job(client, near["id"])
    entry = near_state["entries"][0]
    check(entry["status"] == "ok", "near-deadline request was dispatched early and scored")
    check(entry["latency_ms"] < 200, f"latency {entry['latency_ms']}ms < 200ms window")

    print("8. interleaved demo")
    demo = client.post("/api/demo/interleaved").json()
    demo_state = wait_job(client, demo["id"], timeout=10)
    old = [e for e in demo_state["entries"] if e["index"] <= 2]
    new = [e for e in demo_state["entries"] if e["index"] > 2]
    check({e["model_version"] for e in old} == {"v1"}, "demo old requests pinned to v1")
    check({e["model_version"] for e in new} == {"v2"}, "demo new requests used v2")
    slow = next(e for e in new if e["simulate_latency_ms"] > 0)
    check(slow["status"] == "timeout", "demo slow email timed out")
    check(client.get("/healthz").json()["active_model"] == "v2", "demo left v2 active")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
