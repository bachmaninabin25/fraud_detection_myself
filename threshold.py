import numpy as np

FP_COST = 25.0


def total_cost(y, p, amt, thr, fp_cost=FP_COST):
    flag = p >= thr
    missed = amt[(y == 1) & ~flag].sum()
    false_alarms = ((y == 0) & flag).sum() * fp_cost
    return missed + false_alarms


def best_threshold(y, p, amt, fp_cost=FP_COST):
    grid = np.round(np.arange(0.01, 1, 0.01), 2)
    costs = np.array([total_cost(y, p, amt, t, fp_cost) for t in grid])
    return float(grid[costs.argmin()]), float(costs.min())


def precision_recall_at(y, p, thr):
    flag = p >= thr
    precision = y[flag].mean()
    recall = flag[y == 1].mean()
    return precision, recall
