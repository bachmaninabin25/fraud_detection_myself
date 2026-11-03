import time

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from fraudcore.scoring import Scorer

scorer = Scorer("models/model.txt", "models/spec.json")
app = FastAPI(title="Fraud scorer")


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float = Field(gt=0)


class Score(BaseModel):
    TransactionID: int
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    scoring_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": scorer.version}


@app.post("/score", response_model=Score)
async def score(txn: Transaction):
    t0 = time.perf_counter()
    p = float(scorer.predict([txn.model_dump()])[0])
    ms = (time.perf_counter() - t0) * 1000
    return Score(
        TransactionID=txn.TransactionID,
        fraud_probability=p,
        is_fraud=p >= scorer.threshold,
        threshold=scorer.threshold,
        model_version=scorer.version,
        scoring_ms=ms,
    )
