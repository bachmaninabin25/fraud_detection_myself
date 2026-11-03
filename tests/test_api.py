import pandas as pd
from fastapi.testclient import TestClient

from api import app, scorer
from fraudcore.scoring import clean_record

client = TestClient(app)
raw = pd.read_parquet("data/processed/test_raw.parquet")
fraud = clean_record(raw[raw.isFraud == 1].drop(columns=["isFraud"]).iloc[5].to_dict())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_matches_scorer():
    r = client.post("/score", json=fraud)
    assert r.status_code == 200
    body = r.json()
    assert abs(body["fraud_probability"] - float(scorer.predict([fraud])[0])) < 1e-9
    assert body["is_fraud"] is True


def test_missing_amount_rejected():
    no_amt = {k: v for k, v in fraud.items() if k != "TransactionAmt"}
    assert client.post("/score", json=no_amt).status_code == 422


def test_negative_amount_rejected():
    assert client.post("/score", json={**fraud, "TransactionAmt": -5}).status_code == 422
