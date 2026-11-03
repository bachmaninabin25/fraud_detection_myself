import numpy as np

from threshold import best_threshold, precision_recall_at, total_cost


def test_total_cost():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    amt = np.array([100.0, 50.0, 10.0, 10.0])
    assert total_cost(y, p, amt, 0.5, fp_cost=25) == 50 + 25


def test_precision_recall():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    assert precision_recall_at(y, p, 0.5) == (0.5, 0.5)


def test_best_threshold_flags_expensive_fraud():
    y = np.array([1, 0, 0])
    p = np.array([0.3, 0.2, 0.1])
    amt = np.array([1000.0, 5.0, 5.0])
    thr, cost = best_threshold(y, p, amt)
    assert 0.2 < thr <= 0.3 and cost == 0
