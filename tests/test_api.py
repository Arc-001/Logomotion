"""
Tests for src.api.

No live pipeline: generation is stubbed, so these exercise the job lifecycle
— concurrency limiting, timeouts, eviction, and request validation — rather
than anything that renders.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from src import api
from src.api import GenerateRequest, JobStatus


@pytest.fixture(autouse=True)
def clean_jobs():
    api.jobs.clear()
    yield
    api.jobs.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def _queue(job_id):
    api.jobs[job_id] = JobStatus(job_id=job_id, status="pending")


class TestJobConcurrency:
    """A generation job drives manim, ffmpeg and LaTeX; nothing previously
    stopped a client from starting as many as it liked."""

    def _run_jobs(self, monkeypatch, count, limit):
        monkeypatch.setattr(api, "_job_slots", asyncio.Semaphore(limit))

        live = 0
        peak = 0

        async def fake_generate(job_id, *args, **kwargs):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)
            live -= 1
            api.jobs[job_id] = JobStatus(job_id=job_id, status="completed")

        monkeypatch.setattr(api, "_generate_into_job", fake_generate)

        async def main():
            ids = [f"job-{i}" for i in range(count)]
            for job_id in ids:
                _queue(job_id)
            await asyncio.gather(*[
                api._run_generation_job(job_id, "p", None, 1.0, "detailed",
                                        "landscape", "guide")
                for job_id in ids
            ])
            return ids

        ids = asyncio.run(main())
        return peak, ids

    def test_never_more_than_the_configured_limit_run_at_once(self, monkeypatch):
        peak, _ = self._run_jobs(monkeypatch, count=6, limit=2)

        assert peak == 2

    def test_every_queued_job_still_completes(self, monkeypatch):
        _, ids = self._run_jobs(monkeypatch, count=6, limit=2)

        assert all(api.jobs[job_id].status == "completed" for job_id in ids)

    def test_a_job_evicted_while_queued_is_not_started(self, monkeypatch):
        monkeypatch.setattr(api, "_job_slots", asyncio.Semaphore(1))
        started = []

        async def fake_generate(job_id, *args, **kwargs):
            started.append(job_id)

        monkeypatch.setattr(api, "_generate_into_job", fake_generate)

        async def main():
            await api._run_generation_job("gone", "p", None, 1.0, "detailed",
                                          "landscape", "guide")

        asyncio.run(main())  # never queued, so not in the store

        assert started == []

    def test_a_job_is_pending_until_it_holds_a_slot(self, monkeypatch):
        monkeypatch.setattr(api, "_job_slots", asyncio.Semaphore(1))
        seen = {}

        async def fake_generate(job_id, *args, **kwargs):
            seen[job_id] = api.jobs["waiter"].status
            await asyncio.sleep(0.02)

        monkeypatch.setattr(api, "_generate_into_job", fake_generate)

        async def main():
            _queue("holder")
            _queue("waiter")
            await asyncio.gather(
                api._run_generation_job("holder", "p", None, 1.0, "detailed",
                                        "landscape", "guide"),
                api._run_generation_job("waiter", "p", None, 1.0, "detailed",
                                        "landscape", "guide"),
            )

        asyncio.run(main())

        # While the first job held the only slot, the second was still pending.
        assert seen["holder"] == "pending"


class TestGenerateEndpoint:
    def test_queues_a_job_and_returns_its_id(self, client, monkeypatch):
        async def fake_run(job_id, *args, **kwargs):
            api.jobs[job_id] = JobStatus(job_id=job_id, status="completed")

        monkeypatch.setattr(api, "_run_generation_job", fake_run)

        response = client.post("/generate", json={"topic": "binary search"})

        assert response.status_code == 200
        assert response.json()["job_id"] in api.jobs

    def test_requires_a_prompt_or_topic(self, client):
        assert client.post("/generate", json={}).status_code == 422

    def test_unknown_job_is_a_404(self, client):
        assert client.get("/jobs/nope").status_code == 404

    def test_download_before_completion_is_rejected(self, client):
        _queue("waiting")

        response = client.get("/jobs/waiting/download")

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]


class TestGenerateRequestDefaults:
    def test_topic_is_used_as_the_prompt(self):
        assert GenerateRequest(topic="circles").prompt == "circles"

    def test_quality_names_are_normalised(self):
        assert GenerateRequest(prompt="x", quality="high").quality == "h"


class TestJobTimeout:
    """Only the per-render subprocess was bounded; a hung LLM call kept a job
    "running" forever."""

    def _run(self, monkeypatch, generate, timeout=0.05):
        from src.config import Settings

        settings = Settings(job_timeout=timeout)
        monkeypatch.setattr(api, "get_settings", lambda: settings)

        import src.agent.graph as graph_module
        monkeypatch.setattr(graph_module, "generate_video", generate)

        _queue("slow")
        asyncio.run(api._generate_into_job(
            "slow", "p", None, 1.0, "detailed", "landscape", "guide",
            False, None, None, None,
        ))
        return api.jobs["slow"]

    def test_a_hung_generation_fails_the_job(self, monkeypatch):
        async def never_finishes(**kwargs):
            await asyncio.sleep(10)

        job = self._run(monkeypatch, never_finishes)

        assert job.status == "failed"
        assert "job timeout" in job.error

    def test_a_generation_that_finishes_in_time_is_unaffected(self, monkeypatch):
        async def finishes(**kwargs):
            return {"final_output_path": "/tmp/out.mp4", "code": "x"}

        job = self._run(monkeypatch, finishes, timeout=5)

        assert job.status == "completed"
        assert job.video_path == "/tmp/out.mp4"


class TestSearchEndpoint:
    """hybrid_search is blocking Neo4j/Chroma I/O; running it inline stalled
    every other request for the length of the round trip."""

    def test_search_runs_off_the_event_loop(self, client, monkeypatch):
        import src.graph_rag.retriever as retriever_module

        threads = []

        class _Fake:
            def hybrid_search(self, query, limit=5):
                import threading
                threads.append(threading.current_thread().name)
                return []

        monkeypatch.setattr(retriever_module, "get_retriever", lambda: _Fake())

        response = client.get("/search?query=circle")

        assert response.status_code == 200
        assert threads and threads[0] != "MainThread"

    def test_search_serialises_results(self, client, monkeypatch):
        import src.graph_rag.retriever as retriever_module
        from src.graph_rag.retriever import RetrievalResult

        class _Fake:
            def hybrid_search(self, query, limit=5):
                return [RetrievalResult(
                    example_id="e1", prompt="p" * 300, code="c", score=0.5,
                    used_classes=["Circle"], used_animations=["Create"],
                )]

        monkeypatch.setattr(retriever_module, "get_retriever", lambda: _Fake())

        body = client.get("/search?query=circle").json()

        assert body["query"] == "circle"
        assert body["results"][0]["id"] == "e1"
        assert len(body["results"][0]["prompt"]) == 200
        assert body["results"][0]["classes"] == ["Circle"]
