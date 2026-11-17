"""Gateway-level tests: model switching while requests are queued."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import main as main_module  # noqa: E402


@pytest.fixture()
def client():
    # Reset process-global state per test.
    main_module.jobs.clear()
    main_module.switches.clear()
    main_module.active_version = main_module.registry.default_version
    main_module.worker.metrics.update(
        {key: 0 for key in main_module.worker.metrics}
    )
    main_module.worker.reset_admission()
    with TestClient(main_module.app) as test_client:
        yield test_client


def _wait(client, job_id: str, timeout: float = 8.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.01)
    return state


def test_health_and_models(client):
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert {"v1", "v2"} <= set(health["models"])

    models = client.get("/api/models").json()
    assert models["active"] in ("v1", "v2")
    assert len(models["models"]) == 2
    v1, v2 = {m["version"]: m for m in models["models"]}.values()
    assert v1["vocabulary_size"] != v2["vocabulary_size"]


# ---------------------------------------------------------------------------
# global admission cap (8 emails, shared by sync and async submissions)
# ---------------------------------------------------------------------------
def test_admission_cap_shared_by_async_and_sync(client):
    """Eight slots total; later emails are rejected per email, never queued."""
    client.post("/api/model", json={"version": "v1"})
    # Long batching window keeps these eight in the queue while we probe.
    blocker = client.post(
        "/api/jobs",
        json={
            "label": "fills the pool",
            "emails": [
                {"text": f"free spam {i}", "deadline_ms": 1000,
                 "simulate_latency_ms": 900}
                for i in range(8)
            ],
        },
    ).json()
    assert blocker["size"] == 8

    # The synchronous path shares the same pool and must be refused too.
    sync = client.post(
        "/api/classify",
        json={"label": "sync", "emails": [{"text": "free spam", "deadline_ms": 1000}]},
    ).json()
    assert sync["entries"][0]["status"] == "rejected"
    assert sync["entries"][0]["reason"] == "admission_overloaded"
    assert sync["entries"][0]["model_version"] is None
    assert sync["entries"][0]["batch_id"] is None
    assert sync["rejected"] == 1 and sync["accepted"] == 0

    # The async path reports rejection per email as well.
    async_probe = client.post(
        "/api/jobs",
        json={
            "label": "async",
            "emails": [{"text": "free spam", "deadline_ms": 1000}],
        },
    ).json()
    state = _wait(client, async_probe["id"], timeout=3)
    entry = state["entries"][0]
    assert entry["admission"] == "rejected"
    assert entry["status"] == "rejected"

    snapshot = client.get("/api/state").json()["admission"]
    assert snapshot["limit"] == 8
    assert snapshot["rejected_total"] >= 2
    assert snapshot["used"] <= 8

    # No batch may contain a rejected request id.
    rejected_ids = {
        e["request_id"] for e in sync["entries"] if e["status"] == "rejected"
    } | {entry["request_id"]}
    for batch in client.get("/api/batches?limit=50").json()["batches"]:
        assert not (rejected_ids & set(batch["request_ids"]))


def test_admission_state_exposed(client):
    state = client.get("/api/state").json()
    assert state["config"]["max_admitted_emails"] == 8
    admission = state["admission"]
    for key in ("limit", "used", "available", "queued", "executing",
                "admitted_total", "rejected_total"):
        assert key in admission
    assert admission["limit"] == 8


def test_admission_slots_return_after_queue_expiry(client):
    """Queued emails that expire while the worker is busy free their slots."""
    client.post("/api/model", json={"version": "v1"})
    # 4 slow-executing emails occupy the worker; 4 more fill the pool.
    client.post(
        "/api/jobs",
        json={
            "label": "slow",
            "emails": [
                {"text": f"free spam {i}", "deadline_ms": 1000,
                 "simulate_latency_ms": 800}
                for i in range(8)
            ],
        },
    )
    probe = client.post(
        "/api/jobs",
        json={"label": "probe", "emails": [{"text": "free spam", "deadline_ms": 200}]},
    ).json()
    probe_state = _wait(client, probe["id"], timeout=4)
    assert probe_state["entries"][0]["admission"] == "rejected"

    # Only short deadlines are left behind: re-probing must eventually be
    # admitted once the earlier emails have settled and released their slots.
    import time

    deadline = time.monotonic() + 8.0
    admitted = False
    while time.monotonic() < deadline:
        again = client.post(
            "/api/classify",
            json={
                "label": "again",
                "emails": [{"text": "free spam", "deadline_ms": 1000}],
            },
        ).json()
        if again["entries"][0]["status"] != "rejected":
            admitted = True
            break
        time.sleep(0.05)
    assert admitted, "slots were never released"
    assert client.get("/api/state").json()["admission"]["rejected_total"] >= 1


def test_switch_model_endpoint(client):
    r = client.post("/api/model", json={"version": "v2"})
    assert r.status_code == 200
    assert r.json()["active"] == "v2"
    assert client.get("/healthz").json()["active_model"] == "v2"

    bad = client.post("/api/model", json={"version": "v9"})
    assert bad.status_code == 404


def test_switch_does_not_retarget_queued_requests(client):
    """Old queued emails keep v1; emails accepted after the switch use v2."""
    # Occupy the single worker with a long-deadline job so later submissions
    # wait in the queue.
    client.post("/api/model", json={"version": "v1"})
    entry = {
        "text": "Limited offer! Invest in crypto today and double your money fast.",
        "deadline_ms": 900,
    }
    job = client.post(
        "/api/jobs",
        json={"label": "pinned", "emails": [entry, dict(entry, deadline_ms=800)]},
    ).json()
    # Let intake accept the first email (pinning v1) before switching.
    import time

    watched = None
    for _ in range(100):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        if watched["entries"][0]["model_version"]:
            break
        time.sleep(0.01)
    assert watched["entries"][0]["model_version"] == "v1"

    client.post("/api/model", json={"version": "v2"})

    for _ in range(300):
        watched = client.get(f"/api/jobs/{job['id']}").json()
        if watched["status"] == "completed":
            break
        time.sleep(0.01)

    assert watched["status"] == "completed"
    assert all(e["model_version"] == "v1" for e in watched["entries"])
    assert all(e["status"] in ("ok", "timeout") for e in watched["entries"])


def test_switch_between_jobs_uses_new_version(client):
    client.post("/api/model", json={"version": "v1"})
    first = client.post(
        "/api/jobs",
        json={"label": "a", "emails": [{"text": "free prize winner", "deadline_ms": 400}]},
    ).json()
    client.post("/api/model", json={"version": "v2"})
    second = client.post(
        "/api/jobs",
        json={"label": "b", "emails": [{"text": "free prize winner", "deadline_ms": 400}]},
    ).json()

    for job in (first, second):
        for _ in range(300):
            state = client.get(f"/api/jobs/{job['id']}").json()
            if state["status"] == "completed":
                break
            import time

            time.sleep(0.01)
        else:
            pytest.fail("job did not complete")
    a = client.get(f"/api/jobs/{first['id']}").json()
    b = client.get(f"/api/jobs/{second['id']}").json()
    assert a["entries"][0]["model_version"] == "v1"
    assert b["entries"][0]["model_version"] == "v2"


def test_submission_limits(client):
    too_many = {
        "label": "big",
        "emails": [{"text": f"mail {i}", "deadline_ms": 300} for i in range(17)],
    }
    assert client.post("/api/jobs", json=too_many).status_code == 422
    bad_deadline = {
        "label": "bad",
        "emails": [{"text": "x", "deadline_ms": 99}],
    }
    assert client.post("/api/jobs", json=bad_deadline).status_code == 422
    ok = {
        "label": "max",
        "emails": [{"text": f"mail {i}", "deadline_ms": 800} for i in range(16)],
    }
    response = client.post("/api/jobs", json=ok)
    assert response.status_code == 202
    assert response.json()["size"] == 16


def test_timeout_never_becomes_success(client):
    result = client.post(
        "/api/classify",
        json={
            "label": "late",
            "emails": [
                {
                    "text": "free spam winner",
                    "deadline_ms": 100,
                    "simulate_latency_ms": 600,
                },
                {"text": "free spam winner", "deadline_ms": 1000},
            ],
        },
    ).json()
    entries = result["entries"]
    late = next(e for e in entries if e["deadline_ms"] == 100)
    fast = next(e for e in entries if e["deadline_ms"] == 1000)
    assert late["status"] == "timeout"
    assert late["score"] is None
    assert late["late_result_discarded"] is True
    assert fast["status"] == "ok"
    assert late["batch_id"] == fast["batch_id"]


def test_batches_never_mix_versions_through_gateway(client):
    client.post("/api/model", json={"version": "v1"})
    client.post(
        "/api/jobs",
        json={
            "label": "v1-batch",
            "emails": [{"text": f"mail {i} free", "deadline_ms": 800} for i in range(4)],
        },
    )
    client.post("/api/model", json={"version": "v2"})
    client.post(
        "/api/jobs",
        json={
            "label": "v2-batch",
            "emails": [{"text": f"mail {i} winner", "deadline_ms": 800} for i in range(4)],
        },
    )
    import time

    time.sleep(0.9)
    batches = client.get("/api/batches").json()["batches"]
    assert batches
    for batch in batches:
        assert batch["size"] <= 4
        assert batch["model_version"] in ("v1", "v2")
    assert {b["model_version"] for b in batches} >= {"v1", "v2"}


def test_demo_interleaved(client):
    result = client.post("/api/demo/interleaved").json()
    import time

    for _ in range(400):
        state = client.get(f"/api/jobs/{result['id']}").json()
        if state["status"] == "completed":
            break
        time.sleep(0.01)
    else:
        pytest.fail("demo job did not complete")
    old = [e for e in state["entries"] if e["index"] <= 2]
    new = [e for e in state["entries"] if e["index"] > 2]
    assert {e["model_version"] for e in old} == {"v1"}
    assert {e["model_version"] for e in new} == {"v2"}
    # The artificially slow new-model email must time out but not take down
    # its batch-mate.
    slow = next(e for e in new if e["simulate_latency_ms"] > 0)
    assert slow["status"] == "timeout"
    assert slow["score"] is None
    assert client.get("/healthz").json()["active_model"] == "v2"
    assert any(ev["type"] == "model_switched" for ev in state["events"])
