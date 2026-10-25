"""Metrics: discrimination, calibration and threshold-dependent scores."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def score(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "threshold": round(float(threshold), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
    }
