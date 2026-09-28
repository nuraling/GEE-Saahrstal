"""Sentinel-2 10 m → 1 m / 3 m.

Three methods, in the order the pipeline tries them:

* ``geoai``     – the pretrained network Solain already uses (opengeos/geoai).
* ``reference`` – trained on *this* site: Pléiades is degraded to 10 m, and a
                  model learns the detail bicubic interpolation misses; that
                  model is then applied to Sentinel-2. It can only restore
                  detail whose 10 m signature it has seen, and it is scored on
                  held-out blocks against real Pléiades.
* ``bicubic``   – interpolation. The honest floor every other method must beat.

No method recovers what 10 m pixels never recorded: a 2 m gap between two
piles is gone. Super-resolution sharpens footprints; it does not create height.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import StockpileConfig
from .grid import Grid, block_mean, resample

BANDS = ("blue", "green", "red", "nir")


def bicubic(bands10: Dict[str, np.ndarray], g10: Grid, ghr: Grid) -> Dict[str, np.ndarray]:
    return {k: resample(v, g10, ghr, order=3) for k, v in bands10.items()}


def _match_radiometry(s2: np.ndarray, ref10: np.ndarray) -> Tuple[float, float]:
    """ref ≈ a·s2 + b over pixels valid in both (different sensor, date, units)."""
    ok = np.isfinite(s2) & np.isfinite(ref10)
    if ok.sum() < 20:
        return 1.0, 0.0
    A = np.c_[s2[ok], np.ones(ok.sum())]
    (a, b), *_ = np.linalg.lstsq(A, ref10[ok], rcond=None)
    return float(a), float(b)


def _detail_features(up: Dict[str, np.ndarray], res: float) -> np.ndarray:
    fs = []
    for k in BANDS:
        if k not in up:
            continue
        a = np.nan_to_num(up[k])
        gy, gx = np.gradient(a)
        fs += [a, gx, gy, a - ndimage.gaussian_filter(a, 10.0 / res / 2),
               ndimage.laplace(a)]
    return np.dstack(fs).astype(np.float32)


def reference_sr(s2_10: Dict[str, np.ndarray], g10: Grid, ref_hr: Dict[str, np.ndarray],
                 ghr: Grid, folds_hr: Optional[np.ndarray] = None,
                 cfg=StockpileConfig) -> Dict:
    """Site-trained SR. Returns {"bands": {...}, "metrics": {...}}.

    ref_hr: Pléiades bands on ghr. folds_hr: spatial folds for scoring; when
    given, the reported metrics are out-of-fold.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    ref10 = {k: block_mean(ref_hr[k], ghr, g10) for k in BANDS if k in ref_hr}
    # what a bicubic upsample of a 10 m image of this site looks like
    ref_up = {k: resample(v, g10, ghr, order=3) for k, v in ref10.items()}
    Xtr = _detail_features(ref_up, ghr.res)
    # the S2 input, first mapped onto Pléiades radiometry
    s2m = {}
    for k in ref10:
        a, b = _match_radiometry(s2_10[k], ref10[k])
        s2m[k] = a * s2_10[k] + b
    s2_up = {k: resample(v, g10, ghr, order=3) for k, v in s2m.items()}
    Xap = _detail_features(s2_up, ghr.res)

    H, W, F = Xtr.shape
    rng = np.random.default_rng(cfg.RAND_SEED)
    out, metrics = {}, {}
    for bi, k in enumerate(ref10):
        target = (ref_hr[k] - ref_up[k]).reshape(-1)
        valid = np.flatnonzero(np.isfinite(target) & np.all(np.isfinite(Xtr.reshape(-1, F)), axis=1))
        oof = np.full(H * W, np.nan, np.float32)
        if folds_hr is not None:
            fl = folds_hr.reshape(-1)
            for f in range(int(fl.max()) + 1):
                tr = valid[fl[valid] != f]
                te = valid[fl[valid] == f]
                if tr.size < 100 or te.size == 0:
                    continue
                s = tr if tr.size <= 100_000 else rng.choice(tr, 100_000, replace=False)
                m = HistGradientBoostingRegressor(max_iter=150, random_state=cfg.RAND_SEED)
                m.fit(Xtr.reshape(-1, F)[s], target[s])
                oof[te] = m.predict(Xtr.reshape(-1, F)[te])
            truth = ref_hr[k].reshape(-1)
            base = ref_up[k].reshape(-1)
            ok = np.isfinite(oof) & np.isfinite(truth) & np.isfinite(base)
            rm_b = float(np.sqrt(np.mean((base[ok] - truth[ok]) ** 2)))
            rm_s = float(np.sqrt(np.mean((base[ok] + oof[ok] - truth[ok]) ** 2)))
            metrics[k] = {"rmse_bicubic": rm_b, "rmse_sr": rm_s,
                          "improvement_pct": 100 * (rm_b - rm_s) / rm_b if rm_b else 0.0}
        s = valid if valid.size <= 150_000 else rng.choice(valid, 150_000, replace=False)
        m = HistGradientBoostingRegressor(max_iter=150, random_state=cfg.RAND_SEED)
        m.fit(Xtr.reshape(-1, F)[s], target[s])
        detail = m.predict(np.nan_to_num(Xap.reshape(-1, F))).reshape(H, W).astype(np.float32)
        out[k] = s2_up[k] + detail
    return {"bands": out, "metrics": metrics, "method": "reference"}


def geoai_sr(s2_10: Dict[str, np.ndarray], g10: Grid, ghr: Grid, log=print) -> Optional[Dict]:
    """Pretrained SR via geoai (as in Solain). None when unavailable."""
    try:
        from geoai import super_resolution  # noqa: F401
    except Exception as exc:
        log(f"geoai not available ({type(exc).__name__}) — using the next SR method")
        return None
    import os
    import tempfile
    import rasterio
    try:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "s2.tif")
            dst = os.path.join(td, "s2_sr.tif")
            keys = [k for k in BANDS if k in s2_10]
            with rasterio.open(src, "w", driver="GTiff", width=g10.width, height=g10.height,
                               count=len(keys), dtype="float32", crs=g10.crs,
                               transform=g10.transform) as ds:
                for i, k in enumerate(keys, 1):
                    ds.write(s2_10[k].astype("float32"), i)
            super_resolution(src, dst)
            with rasterio.open(dst) as ds:
                sr_res = ds.transform.a
                sg = Grid(ds.transform.c, ds.transform.f, sr_res, ds.width, ds.height, g10.crs)
                bands = {k: resample(ds.read(i).astype(np.float32), sg, ghr, order=1)
                         for i, k in enumerate(keys, 1)}
        return {"bands": bands, "metrics": {}, "method": "geoai"}
    except Exception as exc:
        log(f"geoai super-resolution failed ({exc}) — using the next SR method")
        return None


def super_resolve(s2_10: Dict[str, np.ndarray], g10: Grid, ghr: Grid,
                  ref_hr: Optional[Dict[str, np.ndarray]] = None,
                  folds_hr: Optional[np.ndarray] = None,
                  method: str = StockpileConfig.S2_SR_METHOD, log=print) -> Dict:
    order = {"geoai": ["geoai", "reference", "bicubic"],
             "reference": ["reference", "bicubic"],
             "bicubic": ["bicubic"]}.get(method, ["bicubic"])
    for m in order:
        if m == "geoai":
            r = geoai_sr(s2_10, g10, ghr, log)
            if r:
                return r
        elif m == "reference" and ref_hr:
            return reference_sr(s2_10, g10, ref_hr, ghr, folds_hr)
        elif m == "bicubic":
            return {"bands": bicubic(s2_10, g10, ghr), "metrics": {}, "method": "bicubic"}
    return {"bands": bicubic(s2_10, g10, ghr), "metrics": {}, "method": "bicubic"}


class ReferenceSR:
    """The site-trained SR model, fitted once and applied to any Sentinel-2
    scene (reference_sr retrains on every call — fine for one scene, not for
    a time series)."""

    def __init__(self, ref_hr: Dict[str, np.ndarray], g10: Grid, ghr: Grid, cfg=StockpileConfig):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.g10, self.ghr = g10, ghr
        self.ref10 = {k: block_mean(ref_hr[k], ghr, g10) for k in BANDS if k in ref_hr}
        ref_up = {k: resample(v, g10, ghr, order=3) for k, v in self.ref10.items()}
        X = _detail_features(ref_up, ghr.res)
        F = X.shape[2]
        Xf = X.reshape(-1, F)
        rng = np.random.default_rng(cfg.RAND_SEED)
        self.models = {}
        for k in self.ref10:
            t = (ref_hr[k] - ref_up[k]).reshape(-1)
            ok = np.flatnonzero(np.isfinite(t) & np.all(np.isfinite(Xf), axis=1))
            s = ok if ok.size <= 150_000 else rng.choice(ok, 150_000, replace=False)
            m = HistGradientBoostingRegressor(max_iter=150, random_state=cfg.RAND_SEED)
            m.fit(Xf[s], t[s])
            self.models[k] = m

    def apply(self, s2_10: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        s2m = {}
        for k in self.ref10:
            a, b = _match_radiometry(s2_10[k], self.ref10[k])
            s2m[k] = a * s2_10[k] + b
        up = {k: resample(v, self.g10, self.ghr, order=3) for k, v in s2m.items()}
        X = _detail_features(up, self.ghr.res)
        F = X.shape[2]
        return {k: up[k] + self.models[k].predict(np.nan_to_num(X.reshape(-1, F))).reshape(X.shape[:2]).astype(np.float32)
                for k in self.models}
