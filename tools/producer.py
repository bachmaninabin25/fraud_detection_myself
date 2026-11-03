"""Replay the held-out test transactions into Kafka, in time order.

Stands in for a payment processor emitting transactions as they happen.
The label (isFraud) is stripped: a live stream doesn't know the answer.

    python -m tools.producer --rate 2000      # 2,000 tx/s
    python -m tools.producer --rate 0         # as fast as possible
    python -m tools.producer --amount-scale 3 # simulate drift: amounts 3x bigger
"""
import argparse
import json
import os
import time

import pandas as pd
from confluent_kafka import Producer

from fraudcore.scoring import clean_record

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092")
TOPIC = "transactions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=1000, help="messages/sec, 0 = unthrottled")
    ap.add_argument("--limit", type=int, default=0, help="stop after N messages, 0 = all")
    ap.add_argument("--data", default="data/processed/test_raw.parquet")
    ap.add_argument("--amount-scale", type=float, default=1.0,
                    help="multiply TransactionAmt to simulate data drift")
    a = ap.parse_args()

    raw = pd.read_parquet(a.data).drop(columns=["isFraud"])
    if a.limit:
        raw = raw.head(a.limit)
    if a.amount_scale != 1.0:
        raw["TransactionAmt"] *= a.amount_scale
        print(f"simulating drift: TransactionAmt x{a.amount_scale}", flush=True)
    records = [clean_record(r) for r in raw.to_dict("records")]
    print(f"loaded {len(records):,} transactions, sending to '{TOPIC}' at "
          f"{'max' if not a.rate else f'{a.rate:,.0f}/s'}", flush=True)

    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        "linger.ms": 2,            # wait up to 2ms to batch messages together
        "compression.type": "lz4",
        "acks": "all",             # broker confirms each write
    })

    t_start = time.perf_counter()
    for i, rec in enumerate(records):
        if a.rate:  # pace to the target rate
            ahead = t_start + i / a.rate - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
        rec["produced_at"] = time.time() * 1000
        # Key by card: all of one card's transactions land on the same
        # partition, so they're scored in order.
        key = str(rec.get("card1", rec["TransactionID"]))
        while True:
            try:
                producer.produce(TOPIC, key=key, value=json.dumps(rec))
                break
            except BufferError:      # local queue full: let it drain
                producer.poll(0.05)
        producer.poll(0)
        if (i + 1) % 20_000 == 0:
            el = time.perf_counter() - t_start
            print(f"  sent {i + 1:,} ({(i + 1) / el:,.0f}/s)", flush=True)

    producer.flush()
    el = time.perf_counter() - t_start
    print(f"done: {len(records):,} messages in {el:.1f}s ({len(records) / el:,.0f}/s)", flush=True)


if __name__ == "__main__":
    main()
