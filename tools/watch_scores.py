"""Watch the `scores` topic live: throughput, end-to-end latency, fraud alerts.

    python -m tools.watch_scores
"""
import json
import os
import time
from collections import Counter

import numpy as np
from confluent_kafka import Consumer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092")


def main():
    c = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": f"watcher-{os.getpid()}",   # own group: sees every score, doesn't steal work
        "auto.offset.reset": "latest",
    })
    c.subscribe(["scores"])
    print("watching 'scores'... (Ctrl+C to stop)")
    n, flagged, lat, workers, t0 = 0, 0, [], Counter(), time.perf_counter()
    try:
        while True:
            for m in c.consume(num_messages=1000, timeout=0.2):
                if m.error():
                    continue
                s = json.loads(m.value())
                n += 1
                lat.append(s["e2e_ms"])
                workers[s["worker"]] += 1
                if s["is_fraud"]:
                    flagged += 1
                    if flagged % 200 == 1:
                        print(f"  ALERT tx {s['TransactionID']}  ${s['TransactionAmt']:.2f}  "
                              f"p={s['fraud_probability']:.2f}")
            el = time.perf_counter() - t0
            if el >= 2 and n:
                print(f"{n / el:7,.0f} scores/s | e2e p50 {np.percentile(lat, 50):6.1f}ms "
                      f"p99 {np.percentile(lat, 99):6.1f}ms | flagged {flagged / n:5.1%} | "
                      f"workers {len(workers)}")
                n, flagged, lat, workers, t0 = 0, 0, [], Counter(), time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        c.close()


if __name__ == "__main__":
    main()
