from fastapi.testclient import TestClient

from api.main import app


def test_health_returns_200():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
