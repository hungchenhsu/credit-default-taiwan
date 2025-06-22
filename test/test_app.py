# tests/test_app.py
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict_ok():
    response = client.post("/predict", json={
        "records": [{
            "LIMIT_BAL": 200000,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 29,
            "PAY_0": 1, "PAY_2": 0, "PAY_3": 0,
            "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
            "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
            "PAY_AMT1": 0, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
            "PAY_AMT4": 1000, "PAY_AMT5": 0, "PAY_AMT6": 0
        }]
    })
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "probability" in data[0]
    assert "label" in data[0]
