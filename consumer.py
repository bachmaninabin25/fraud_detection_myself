import json
import signal
import time

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from fraudcore.scoring import Scorer

BOOTSTRAP = "localhost:29092"
IN_TOPIC, OUT_TOPIC = "transactions", "scores"
GROUP_ID = "my-scorers"
WORKER = "mine"
BATCH = 500
POLL_TIMEOUT = 0.005

running = True


def stop(*_):
    global running
    running = False


def make_consumer():
    c = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "session.timeout.ms": 10000,
        "heartbeat.interval.ms": 1000,
        "partition.assignment.strategy": "cooperative-sticky",
    })
    c.subscribe([IN_TOPIC])
    return c


def make_producer():
    return Producer({"bootstrap.servers": BOOTSTRAP, "linger.ms": 2})


def score_batch(scorer, txns, producer):
    probs = scorer.predict(txns)
    now = time.time() * 1000
    for t, p in zip(txns, probs):
        p = float(p)
        producer.produce(OUT_TOPIC, key=str(t["TransactionID"]), value=json.dumps({
            "TransactionID": t["TransactionID"],
            "TransactionAmt": t["TransactionAmt"],
            "fraud_probability": p,
            "is_fraud": p >= scorer.threshold,
            "model_version": scorer.version,
            "worker": WORKER,
            "produced_at": t["produced_at"],
            "scored_at": now,
            "e2e_ms": now - t["produced_at"],
        }))
    producer.poll(0)


def main():
    scorer = Scorer("models/model.txt", "models/spec.json")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    consumer, producer = make_consumer(), make_producer()
    print(f"[{WORKER}] ready, model {scorer.version}")

    done, t_last = 0, time.time()
    while running:
        msgs = consumer.consume(num_messages=BATCH, timeout=POLL_TIMEOUT)
        if not msgs:
            continue
        txns = []
        for m in msgs:
            if m.error():
                if m.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[{WORKER}] error: {m.error()}")
                continue
            txns.append(json.loads(m.value()))
        if txns:
            score_batch(scorer, txns, producer)
        consumer.commit(asynchronous=True)

        done += len(txns)
        if time.time() - t_last >= 5:
            print(f"[{WORKER}] {done / (time.time() - t_last):,.0f} msg/s")
            done, t_last = 0, time.time()

    print(f"[{WORKER}] shutting down ...")
    producer.flush(10)
    try:
        consumer.commit(asynchronous=False)
    except KafkaException:
        pass
    finally:
        consumer.close()
    print(f"[{WORKER}] stopped cleanly")


if __name__ == "__main__":
    main()
