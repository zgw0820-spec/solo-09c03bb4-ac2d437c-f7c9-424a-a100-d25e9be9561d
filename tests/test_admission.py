"""Admission-cap tests: a global pool of 8 email slots.

The contract under test:
* synchronous and asynchronous submissions share the same pool;
* a refused email never enters the model queue and is reported per email;
* a queued email releases its slot when it expires;
* an executing email keeps its slot after it has already been reported as a
  timeout, until the non-interruptible scoring call really returns.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.batching import Worker  # noqa: E402

from test_batching import make_registry, wait_for  # noqa: E402


@pytest.mark.asyncio
async def test_eight_admitted_ninth_rejected_and_never_queued():
    worker = Worker(make_registry(), max_batch_size=4, batch_wait_ms=5000)
    await worker.start()
    try:
        admissions = [
            worker.admit(
                worker.new_item(
                    text="free spam",
                    model_version="v1",
                    deadline_ms=1000,
                    request_id=f"r-{i}",
                    request_batch_id="cap",
                )
            )
            for i in range(9)
        ]
        accepted = [a for a in admissions if a.accepted]
        rejected = [a for a in admissions if not a.accepted]
        assert len(accepted) == 8
        assert len(rejected) == 1
        assert rejected[0].rejected["status"] == "rejected"
        assert rejected[0].rejected["reason"] == "admission_overloaded"
        assert rejected[0].rejected["batch_id"] is None

        snapshot = worker.admission_snapshot()
        assert snapshot["limit"] == 8
        assert snapshot["used"] == 8
        assert snapshot["available"] == 0
        assert snapshot["rejected_total"] == 1
        # The refused email was never queued and never counted as submitted.
        assert worker.queued_count() == 8
        assert worker.metrics["submitted"] == 8
        assert worker.metrics["admission_rejected"] == 1
        assert all(
            item.request_id != rejected[0].item.request_id
            for items in worker._queues.values()
            for item in items
        )
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_queued_expiry_returns_slot():
    """A queued email that times out frees its slot for a later submission."""

    def slow_scorer(model, text):
        if "BLOCK" in text:
            time.sleep(0.35)
        return model.predict(text)

    worker = Worker(
        make_registry(),
        max_batch_size=1,
        batch_wait_ms=5,
        scorer=slow_scorer,
        max_admitted=2,
    )
    await worker.start()
    try:
        blocker = asyncio.create_task(
            worker.submit(
                text="BLOCK free spam",
                model_version="v1",
                deadline_ms=900,
                request_id="blocker",
                request_batch_id="slots",
            )
        )
        await wait_for(lambda: worker.admission_snapshot()["executing"] == 1)
        # Second slot taken by a short-lived queued request.
        doomed = asyncio.create_task(
            worker.submit(
                text="free spam",
                model_version="v1",
                deadline_ms=110,
                request_id="doomed",
                request_batch_id="slots",
            )
        )
        await wait_for(lambda: worker.admission_snapshot()["queued"] == 1)
        assert worker.admission_snapshot()["used"] == 2
        doomed_result = await asyncio.wait_for(doomed, timeout=3)
        assert doomed_result["status"] == "timeout"
        assert doomed_result["reason"] == "deadline_exceeded_before_dispatch"

        # Its slot must be back while the blocker still executes.
        assert await wait_for(lambda: worker.admission_snapshot()["used"] == 1)

        replacement = await asyncio.wait_for(
            worker.submit(
                text="free spam",
                model_version="v1",
                deadline_ms=900,
                request_id="replacement",
                request_batch_id="slots",
            ),
            timeout=4,
        )
        assert replacement["status"] == "ok"
        assert (await asyncio.wait_for(blocker, timeout=3))["status"] == "ok"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_executing_timeout_keeps_slot_until_scoring_returns():
    """The reported timeout does not free the slot; actual completion does."""

    def slow_scorer(model, text):
        time.sleep(0.4)
        return model.predict(text)

    worker = Worker(
        make_registry(),
        max_batch_size=1,
        batch_wait_ms=5,
        scorer=slow_scorer,
        max_admitted=1,
    )
    await worker.start()
    try:
        started = time.monotonic()
        result = await asyncio.wait_for(
            worker.submit(
                text="free spam",
                model_version="v1",
                deadline_ms=120,
                request_id="slow-timeout",
                request_batch_id="lease",
            ),
            timeout=3,
        )
        assert result["status"] == "timeout"
        assert result["reason"] == "deadline_exceeded_during_batch"
        assert time.monotonic() - started < 0.35

        # The caller has its timeout result, but the thread is still scoring,
        # so the single admission slot is still held and intake is refused.
        snapshot = worker.admission_snapshot()
        assert snapshot["used"] == 1
        assert snapshot["executing"] == 1
        rejected = await asyncio.wait_for(
            worker.submit(
                text="free spam",
                model_version="v1",
                deadline_ms=500,
                request_id="too-early",
                request_batch_id="lease",
            ),
            timeout=1,
        )
        assert rejected["status"] == "rejected"
        assert rejected["reason"] == "admission_overloaded"

        # Once scoring returns, the slot is released and a new email is taken.
        assert await wait_for(
            lambda: worker.admission_snapshot()["used"] == 0, timeout=2.0
        )
        later = await asyncio.wait_for(
            worker.submit(
                text="free spam",
                model_version="v1",
                deadline_ms=500,
                request_id="after-compute",
                request_batch_id="lease",
            ),
            timeout=3,
        )
        assert later["status"] == "ok"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_concurrent_admissions_never_exceed_limit():
    """Many simultaneous submissions cannot oversell the pool."""
    worker = Worker(make_registry(), max_batch_size=4, batch_wait_ms=5000)
    await worker.start()
    try:
        tasks = [
            worker.submit(
                text="free spam",
                model_version="v1",
                deadline_ms=1000,
                request_id=f"race-{i}",
                request_batch_id="race",
            )
            for i in range(20)
        ]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
        accepted = [r for r in results if r["status"] != "rejected"]
        rejected = [r for r in results if r["status"] == "rejected"]
        assert len(accepted) == 8, results
        assert len(rejected) == 12
        # Those 8 were all still holding slots until their deadlines expired
        # (the batching window is longer than the deadline).
        assert worker.metrics["submitted"] == 8
        assert worker.metrics["admission_rejected"] == 12
        assert worker.admission_snapshot()["used"] == 0
        assert worker.admission_snapshot()["rejected_total"] == 12
    finally:
        await worker.stop()
