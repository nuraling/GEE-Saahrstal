"""Accuracy figures, defined once.

WAPE (Σ|error| / Σ|actual|) is used instead of MAPE because small piles would
otherwise dominate the percentage. It is reported as computed. (The original
workflow divided both MAPEs by 10 before display; that is not a metric.)
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def object_metrics(actual: Sequence[float], predicted: Sequence[float]) -> Dict:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if a.size == 0:
        return {"n": 0}
    e = p - a
    ss = float(((a - a.mean()) ** 2).sum())
    return {
        "n": int(a.size),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "mae": float(np.abs(e).mean()),
        "bias": float(e.mean()),
        "bias_pct": float(100 * e.sum() / a.sum()) if a.sum() else None,
        "wape_pct": float(100 * np.abs(e).sum() / np.abs(a).sum()) if np.abs(a).sum() else None,
        "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 and a.size > 2 else None,
        "total_actual": float(a.sum()), "total_predicted": float(p.sum()),
    }


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float(inter / union) if union else float("nan")
