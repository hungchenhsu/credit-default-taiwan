# tests/test_app.py
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "live" in r.json().get("msg", "")

def test_predict_empty():
    r = client.post("/predict", json={"records":[]})
    assert r.status_code == 200
    assert r.json() == []
