"""Translate predictions into an overbooking decision and its money value.

Policy: if P(cancel) >= threshold, the hotel resells that room.
  - Guest really cancels (TP)  -> recover resell_rate * booking revenue
  - Guest shows up (FP)        -> pay walk_cost to relocate them
  - Not flagged (TN / FN)      -> status quo, value 0
The threshold that maximises value is chosen on the validation season and then
frozen before touching the test season.
"""
from __future__ import annotations

import numpy as np


def net_value(y, p, revenue, threshold: float, resell_rate: float, walk_cost: float) -> float:
    y, p, revenue = map(np.asarray, (y, p, revenue))
    flag = p >= threshold
    gain = resell_rate * revenue[flag & (y == 1)].sum()
    cost = walk_cost * (flag & (y == 0)).sum()
    return float(gain - cost)


def value_curve(y, p, revenue, resell_rate, walk_cost, grid=None):
    grid = np.linspace(0.05, 0.99, 95) if grid is None else grid
    vals = np.array([net_value(y, p, revenue, t, resell_rate, walk_cost) for t in grid])
    return grid, vals


def best_threshold(y, p, revenue, resell_rate, walk_cost) -> tuple[float, float]:
    grid, vals = value_curve(y, p, revenue, resell_rate, walk_cost)
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])
