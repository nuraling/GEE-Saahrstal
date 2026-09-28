"""Optical imagery → pile height and footprint, scored honestly.

The original workflow trained a Random Forest on 1,000 pixels and then scored
it on the same site it was trained on, so neighbouring pixels of the same pile
sat on both sides of the "validation". Here every number is out-of-fold under
*spatial* cross-validation: the site is cut into blocks, whole blocks are held
out, and a pile is only ever predicted by a model that never saw it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import StockpileConfig
from .depth import affine_calibrate
from .grid import Grid


def spatial_folds(grid: Grid, block_m: float, k: int, seed: int,
                  ref: Optional[Grid] = None) -> np.ndarray:
    """Fold id (0..k-1) per pixel, constant over block_m × block_m ground blocks.

    Blocks are defined on the ground (from ``ref``'s origin and extent), so a
    1 m, 3 m and 10 m sensor all hold out exactly the same piles — the only
    way their scores are comparable.
    """
    ref = ref or grid
    nbr = int(np.ceil(ref.height * ref.res / block_m)) + 1
    nbc = int(np.ceil(ref.width * ref.res / block_m)) + 1
    rng = np.random.default_rng(seed)
    ids = (rng.permutation(nbr * nbc) % k).reshape(nbr, nbc)
    ys = (ref.y1 - (grid.y1 - (np.arange(grid.height) + 0.5) * grid.res)) // block_m
    xs = ((grid.x0 + (np.arange(grid.width) + 0.5) * grid.res) - ref.x0) // block_m
    br = np.clip(ys.astype(int), 0, nbr - 1)
    bc = np.clip(xs.astype(int), 0, nbc - 1)
    return ids[br[:, None], bc[None, :]].astype(np.int16)


def _sample(idx: np.ndarray, n: int, rng) -> np.ndarray:
    return idx if idx.size <= n else rng.choice(idx, n, replace=False)


def _balanced_sample(flat_idx: np.ndarray, pos: np.ndarray, n: int, rng) -> np.ndarray:
    """Half pile pixels, half background — piles are a small share of a yard,
    and a model trained on the natural mix learns to predict 'flat'."""
    p = flat_idx[pos[flat_idx]]
    q = flat_idx[~pos[flat_idx]]
    return np.concatenate([_sample(p, n // 2, rng), _sample(q, n - min(p.size, n // 2), rng)])


def fit_predict_oof(X: np.ndarray, height: np.ndarray, pile_mask: np.ndarray,
                    folds: np.ndarray, feature_names: List[str], method: str = "gbm",
                    cfg=StockpileConfig, log=print, in_poly: Optional[np.ndarray] = None,
                    only_folds: Optional[List[int]] = None) -> Dict:
    """Out-of-fold height map and pile probability map for one method.

    in_poly: known-footprint mode. The operator draws the pile polygons, so
    the model only has to explain height *inside* them: training and
    prediction are restricted to polygon pixels, and no footprint classifier
    is needed. Site-wide mode (in_poly=None) must also tell piles from 30 m
    blast-furnace sheds, which is most of what went wrong in the first run.
    only_folds: predict just these folds (e.g. the port hold-out).

    method:
      "rf_spectral"  – the original approach: RF on raw bands only
      "gbm"          – gradient boosting on bands + shading/texture (+ Depth
                       Anything features when present)
      "depth_affine" – Depth Anything local relief, linearly calibrated
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor, RandomForestRegressor)

    H, W, F = X.shape
    Xf = X.reshape(-1, F)
    y = np.clip(np.nan_to_num(height.reshape(-1), nan=0.0), 0, cfg.MAX_PILE_HEIGHT_M)
    valid = np.isfinite(height.reshape(-1)) & np.all(np.isfinite(Xf), axis=1)
    if in_poly is not None:
        valid &= in_poly.reshape(-1).astype(bool)
    pos = pile_mask.reshape(-1).astype(bool)
    fold = folds.reshape(-1)
    rng = np.random.default_rng(cfg.RAND_SEED)
    k = int(fold.max()) + 1

    if method == "rf_spectral":
        cols = [i for i, n in enumerate(feature_names) if n in ("blue", "green", "red", "nir")]
    elif method == "depth_affine":
        if "da_relief_20m" not in feature_names:
            return {"available": False, "reason": "Depth Anything features not present"}
        cols = [feature_names.index("da_relief_20m")]
    else:
        cols = list(range(F))

    h_oof = np.full(H * W, np.nan, np.float32)
    p_oof = np.full(H * W, np.nan, np.float32)
    in_sample = None
    for f in (only_folds if only_folds is not None else range(k)):
        tr = np.flatnonzero(valid & (fold != f))
        te = np.flatnonzero(valid & (fold == f))
        if tr.size < 50 or te.size == 0:
            continue
        n_tr = cfg.N_TRAIN_PX // k * (k - 1)
        s = _balanced_sample(tr, pos, n_tr, rng)
        if method == "depth_affine":
            m = np.zeros(y.size, bool)
            m[s] = True
            fn = poly_calibrate(Xf[:, cols[0]], y, m, cfg.DEPTH_CALIB_DEGREE)
            h_oof[te] = fn(Xf[te, cols[0]])
            p_oof[te] = (h_oof[te] > cfg.MIN_PILE_HEIGHT_M).astype(np.float32)
            continue
        if method == "rf_spectral":
            # as in the original: 30 trees, raw bands
            reg = RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=cfg.RAND_SEED,
                                        min_samples_leaf=2)
            s = _sample(tr, min(n_tr, 50_000), rng)
        else:
            reg = HistGradientBoostingRegressor(max_iter=cfg.GBM_MAX_ITER, learning_rate=0.08,
                                                max_leaf_nodes=63, l2_regularization=1.0,
                                                random_state=cfg.RAND_SEED)
        reg.fit(Xf[s][:, cols], y[s])
        h_oof[te] = np.clip(_predict_chunked(reg, Xf, te, cols), 0, None)
        if method == "rf_spectral" and f == 0:
            # the number the original workflow reported: scored on its own training pixels
            in_sample = _pixel_metrics(reg.predict(Xf[s][:, cols]), y[s], pos[s])
        if in_poly is not None:
            p_oof[te] = 1.0
            continue
        clf = HistGradientBoostingClassifier(max_iter=cfg.GBM_MAX_ITER // 2, learning_rate=0.1,
                                             max_leaf_nodes=31, random_state=cfg.RAND_SEED)
        cs = _balanced_sample(tr, pos, n_tr, rng)
        if len(np.unique(pos[cs])) == 2:
            clf.fit(Xf[cs][:, cols], pos[cs])
            p_oof[te] = _predict_chunked(clf, Xf, te, cols, proba=True)
    return {"available": True, "method": method,
            "height_oof": h_oof.reshape(H, W), "prob_oof": p_oof.reshape(H, W),
            "features_used": [feature_names[i] for i in cols],
            "in_sample_metrics": in_sample}


def poly_calibrate(feature: np.ndarray, target: np.ndarray, mask: np.ndarray, degree: int = 2):
    """height ≈ polynomial(feature) on the training pixels, robust to
    outliers, clipped at 0 and then scaled so the mean prediction equals the
    mean UAV height on those pixels (clipping otherwise shifts the mean)."""
    x, y = feature[mask], target[mask]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 10:
        c = float(np.mean(y)) if y.size else 0.0
        return lambda v: np.full(np.shape(v), c, np.float32)
    coef = np.polyfit(x, y, degree)
    r = y - np.polyval(coef, x)
    keep = np.abs(r) < 2.5 * max(r.std(), 1e-6)
    if keep.sum() > 10:
        coef = np.polyfit(x[keep], y[keep], degree)
    fit = np.clip(np.polyval(coef, x), 0, None)
    k = float(y.mean() / fit.mean()) if fit.mean() > 0 else 1.0
    return lambda v: (k * np.clip(np.polyval(coef, v), 0, None)).astype(np.float32)


def _predict_chunked(model, Xf, idx, cols, proba=False, chunk=1_000_000):
    out = np.empty(idx.size, np.float32)
    for i in range(0, idx.size, chunk):
        sl = idx[i:i + chunk]
        Xi = Xf[sl][:, cols]
        out[i:i + chunk] = model.predict_proba(Xi)[:, 1] if proba else model.predict(Xi)
    return out


def _pixel_metrics(pred, true, mask) -> Dict:
    ok = np.isfinite(pred) & np.isfinite(true) & mask
    if ok.sum() < 2:
        return {}
    e = pred[ok] - true[ok]
    ss_tot = float(((true[ok] - true[ok].mean()) ** 2).sum())
    return {"n": int(ok.sum()), "rmse_m": float(np.sqrt((e ** 2).mean())),
            "mae_m": float(np.abs(e).mean()), "bias_m": float(e.mean()),
            "r2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else None}


def pixel_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> Dict:
    return _pixel_metrics(pred.reshape(-1), true.reshape(-1), mask.reshape(-1).astype(bool))


def fit_full(X: np.ndarray, height: np.ndarray, pile_mask: np.ndarray, cfg=StockpileConfig):
    """Final production model on every pixel (for the map, not for scoring)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    H, W, F = X.shape
    Xf = X.reshape(-1, F)
    y = np.clip(np.nan_to_num(height.reshape(-1)), 0, cfg.MAX_PILE_HEIGHT_M)
    valid = np.flatnonzero(np.isfinite(height.reshape(-1)) & np.all(np.isfinite(Xf), axis=1))
    rng = np.random.default_rng(cfg.RAND_SEED)
    s = _balanced_sample(valid, pile_mask.reshape(-1).astype(bool), cfg.N_TRAIN_PX, rng)
    reg = HistGradientBoostingRegressor(max_iter=cfg.GBM_MAX_ITER, learning_rate=0.08,
                                        max_leaf_nodes=63, random_state=cfg.RAND_SEED)
    reg.fit(Xf[s], y[s])
    return reg


def fit_full_mode(X: np.ndarray, height: np.ndarray, in_poly: np.ndarray, cfg=StockpileConfig):
    """Known-footprint production model: every UAV pixel inside the polygons."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    F = X.shape[2]
    Xf = X.reshape(-1, F)
    y = np.clip(np.nan_to_num(height.reshape(-1)), 0, cfg.MAX_PILE_HEIGHT_M)
    ok = np.flatnonzero(np.isfinite(height.reshape(-1)) & in_poly.reshape(-1) & np.all(np.isfinite(Xf), axis=1))
    rng = np.random.default_rng(cfg.RAND_SEED)
    s = _sample(ok, cfg.N_TRAIN_PX, rng)
    reg = HistGradientBoostingRegressor(max_iter=cfg.GBM_MAX_ITER, learning_rate=0.08,
                                        max_leaf_nodes=63, l2_regularization=1.0,
                                        random_state=cfg.RAND_SEED)
    reg.fit(Xf[s], y[s])
    return reg
