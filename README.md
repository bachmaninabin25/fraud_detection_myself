# Fraud detection, built myself

LightGBM fraud model on IEEE-CIS, picked by dollar cost, served by a FastAPI endpoint
and a Kafka streaming worker, with PSI drift detection.

| file | what it does |
|---|---|
| `train.py` | time split 60/20/20, categories fit on train only, LightGBM with early stopping on PR-AUC, saves `models/` |
| `threshold.py` | cost-based threshold: missed fraud costs the amount, false alarm costs $25 |
| `api.py` | FastAPI: `POST /score`, `GET /health`, Pydantic validation (422 on bad input) |
| `consumer.py` | Kafka worker: batch consume, one model call per batch, commit after publish (at-least-once) |
| `psi.py` | Population Stability Index for data drift |
| `fraudcore/` | shared feature + fast-scoring helpers |
| `tools/` | producer (replays test transactions into Kafka) and a live score watcher |

## Results (test set, never tuned on)

- PR-AUC 0.501 (random 0.034), ROC-AUC 0.888
- threshold 0.05: precision 32.3%, recall 59.0%, cost $402k vs $503k at 0.5 and $610k with no model
- streaming: ~1,000 tx/s per worker at p50 ~8ms end to end

## Setup

```bash
conda activate fraud            # or: pip install -r requirements.txt
```

Data: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection).
`data/` is a link to `../fraud-pipeline/data`; on a fresh machine put the Kaggle CSVs in `data/raw/`.

## Run

```bash
python train.py                 # ~2 min, writes models/model.txt + models/spec.json
python psi.py                   # drift on real amounts: ~0.008 normal, ~1.58 at x3
pytest -q                       # 10 tests
uvicorn api:app --port 8100     # then open http://localhost:8100/docs
```

Streaming (separate terminals):

```bash
docker compose up -d kafka-init
python consumer.py
python -m tools.watch_scores
python -m tools.producer --rate 1000 --limit 20000
docker compose down             # when done
```
