import json

from fastapi.testclient import TestClient

from debate import server
from debate.episodic import EpisodicMemory
from debate.server import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "has_key" in r.json()


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "multi-agent-debate" in r.text


def test_start_and_poll_debate():
    r = client.post("/api/debate", json={"topic": "Should we adopt Go?"})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] in {"queued", "running", "done"}
    # poll until done
    import time
    for _ in range(50):
        g = client.get(f"/api/debate/{job['id']}")
        data = g.json()
        if data["status"] in {"done", "error"}:
            break
        time.sleep(0.05)
    assert data["status"] == "done"
    assert data["verdict"] in {"ACCEPT", "REJECT", "CONTINUE"}
    assert len(data["rounds"]) >= 1


# -- background job pipeline, driven by deterministic stubs ------------------

def _seed_job(job_id: str, topic: str) -> None:
    with server._LOCK:
        server._JOBS[job_id] = {"id": job_id, "status": "queued", "topic": topic,
                                "rounds": [], "verdict": None, "error": None,
                                "metrics": None}


def _rubric_json(winner="proposer", conf=0.9):
    return json.dumps({"evidence": 3, "reasoning": 2, "rebuttal_strength": 1,
                       "clarity": 2, "winner": winner, "confidence": conf})


class RubricJudgeStub:
    """Proposer/critic free text; judge always answers with a valid rubric."""

    def __init__(self):
        self.prompts = []

    def __call__(self, system, prompt):
        last = prompt[-1]["content"]
        self.prompts.append(last)
        if "Decide" in last:
            return _rubric_json()
        if "Counter" in last:
            return "The rollback plan is thin: no rehearsal, no metrics owner."
        return "Adopt the change; LRU eviction restores latency with no new hardware."


def test_job_flow_with_stub_llm_no_rubric(stub_llm):
    _seed_job("jstub", "Should we adopt Go?")
    server._run_job("jstub", "Should we adopt Go?", 1, rubric=False, llm=stub_llm)
    job = server._JOBS["jstub"]
    assert job["status"] == "done"
    assert job["verdict"] == "ACCEPT"          # stub judge says ACCEPT outright
    assert len(job["rounds"]) == 1
    assert job["metrics"] is None              # calibration only with rubric on


def test_job_flow_with_rubric_metrics():
    _seed_job("jrub", "Roll back the eviction change?")
    server._run_job("jrub", "Roll back the eviction change?", 2,
                    rubric=True, llm=RubricJudgeStub())
    job = server._JOBS["jrub"]
    assert job["status"] == "done"
    assert job["verdict"] == "ACCEPT"          # derived from rubric winner
    rub = job["rounds"][0]["rubric"]
    assert rub["winner"] == "proposer" and rub["evidence"] == 3
    m = job["metrics"]
    assert m["n_judged"] == 1
    assert m["agreement"] == 1.0               # BT on the round agrees
    assert 0.0 <= m["ece"] <= 1.0
    assert m["winner_stability"] == 1.0


def test_job_flow_recall_history_uses_episodic_store(monkeypatch):
    store = EpisodicMemory(":memory:")
    monkeypatch.setattr(server, "_EPISODIC", store)

    _seed_job("jh1", "Should we roll back the cache eviction change to fix latency?")
    server._run_job("jh1", "Should we roll back the cache eviction change to fix latency?",
                    1, rubric=True, recall_history=True, llm=RubricJudgeStub())
    assert len(store) == 2                     # both arguments persisted

    _seed_job("jh2", "Does latency justify rolling back the eviction policy?")
    stub2 = RubricJudgeStub()
    server._run_job("jh2", "Does latency justify rolling back the eviction policy?",
                    1, rubric=True, recall_history=True, llm=stub2)
    opener = stub2.prompts[0]
    assert "Prior evidence from earlier debates" in opener
    assert "Should we roll back the cache eviction change to fix latency?" in opener
    assert server._JOBS["jh2"]["status"] == "done"
