from fastapi.testclient import TestClient

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
