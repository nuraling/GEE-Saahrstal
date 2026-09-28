"""Port (area 3 / DSM_3) from SPOT 6/7 at 1.5 m: calibrate on 2025, apply to 2026.

  SPOT 2025-03-27 + UAV 2025-05-01 (5 weeks apart) → per-pixel models,
  scored leave-one-pile-out over the port piles, each required to pass an
  erase test (a pile removed from the image must lose volume) → the best
  one applied to SPOT 2026-04-07 after radiometric matching.

Transfer to other sensors: newer Pléiades scenes (cfg.PLEIADES_RECENT) are
averaged to the 1.5 m SPOT grid, radiometrically matched to SPOT 2025 and run
through the same SPOT-calibrated model — there is no Pléiades 2025 reference
over the port. Piles less than 80 % imaged get no estimate.

Two candidate chains:
  footprint  material classifier → footprint area → V = c·A^b
  height     GBM on SPOT bands + shading/texture + Depth Anything (on SPOT)
             + polygon geometry → height → volume → power correction
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, List

import numpy as np
from scipy import ndimage

from . import terrain
from .config import StockpileConfig
from .depth import DepthAnything, local_relief, to_uint8_rgb
from .features import build_features
from .grid import Grid, resample, to_grid
from .metrics import object_metrics

ASSETS = {"2025-03-27": "spot_20250327_area3", "2026-04-07": "spot_20260407_area3"}
ORTHO = "ortho_area3"
SEARCH_M = 15.0


def port_subgrid(g: Grid, labels: np.ndarray, port_labels: List[int], margin_m: float = 150.0) -> Grid:
    rr, cc = np.nonzero(np.isin(labels, port_labels))
    r0 = max(int(rr.min() - margin_m / g.res) // 30 * 30, 0)
    c0 = max(int(cc.min() - margin_m / g.res) // 30 * 30, 0)
    r1 = min(-(-int(rr.max() + margin_m / g.res) // 30) * 30, g.height)
    c1 = min(-(-int(cc.max() + margin_m / g.res) // 30) * 30, g.width)
    return Grid(g.x0 + c0 * g.res, g.y1 - r0 * g.res, g.res, c1 - c0, r1 - r0, g.crs), (r0, r1, c0, c1)


def fetch_port_imagery(g1: Grid, cfg=StockpileConfig, log=print) -> Dict:
    import ee
    from .sources_ee import fetch, zero_is_nodata
    g15 = g1.with_res(1.5)
    out = {"grid15": g15, "spot": {}}
    for date, a in ASSETS.items():
        img = ee.Image(f"{cfg.ASSET_ROOT}/{a}").select([0, 1, 2, 3], ["blue", "green", "red", "nir"])
        out["spot"][date] = zero_is_nodata(fetch(img, g15, ["blue", "green", "red", "nir"], log=log))
    # other sensors onto the SPOT grid, block-averaged (not resampled) so a
    # 0.3 m scene looks like what a 1.5 m sensor would record
    out["transfer"] = {}
    for date, asset in getattr(cfg, "PLEIADES_RECENT", {}).items():
        m = cfg.PLEIADES_RECENT_BAND_MAP
        img = ee.Image(asset).select([m[k] for k in ("blue", "green", "red", "nir")], ["blue", "green", "red", "nir"])
        img = img.updateMask(img.reduce(ee.Reducer.min()).gt(0))          # 0 = no data in Pléiades assets
        img = img.reduceResolution(ee.Reducer.mean(), maxPixels=256)
        try:
            out["transfer"][date] = {"sensor": "pleiades", "asset": asset,
                                     "bands": fetch(img, g15, ["blue", "green", "red", "nir"], log=log)}
        except Exception as exc:
            log(f"  Pléiades {date} over the port not fetched: {exc}")
    o = ee.Image(f"{cfg.ASSET_ROOT}/{ORTHO}").select([0, 1, 2], ["red", "green", "blue"])
    out["ortho"] = fetch(o, g1, ["red", "green", "blue"], log=log)
    return out


def _normalise(new: Dict, ref: Dict, exclude: np.ndarray) -> Dict:
    """Per-band robust linear match of a scene onto the reference scene over
    pixels outside the piles (quay, roofs, roads)."""
    out, fit = {}, {}
    for k in ref:
        ok = ~exclude & np.isfinite(new[k]) & np.isfinite(ref[k])
        x, y = new[k][ok], ref[k][ok]
        A = np.c_[x, np.ones_like(x)]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ c
        keep = np.abs(r) < 2 * r.std()
        c, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        pred = A[keep] @ c
        fit[k] = {"gain": float(c[0]), "offset": float(c[1]),
                  "r2": float(1 - ((y[keep] - pred) ** 2).sum() / ((y[keep] - y[keep].mean()) ** 2).sum())}
        out[k] = (c[0] * new[k] + c[1]).astype(np.float32)
    return {"bands": out, "fit": fit}


class SpotPortStudy:
    def __init__(self, site: Dict, ref: Dict, cfg=StockpileConfig, log=print):
        from .pipeline import _power_fit, polygon_features
        self.cfg, self.log, self._power_fit = cfg, log, _power_fit
        port_pid = [i for i, p in enumerate(ref["piles"]) if p.get("survey_block") == f"DSM_{cfg.PORT_SURVEY_BLOCK}"
                    and p.get("volume_m3") is not None]
        self.piles = [ref["piles"][i] for i in port_pid]
        port_labels = [p["label"] for p in self.piles]
        self.g1, (r0, r1, c0, c1) = port_subgrid(site["grid"], ref["labels"], port_labels)
        lab = ref["labels"][r0:r1, c0:c1]
        self.labels1 = np.where(np.isin(lab, port_labels), lab, 0).astype(np.int32)
        self.h1 = ref["ndsm"][r0:r1, c0:c1]
        self.dsm1 = site["uav"]["dsm"][r0:r1, c0:c1]
        self.occ1 = ref["occupied"][r0:r1, c0:c1] & (self.labels1 > 0)
        log(f"Port subgrid {self.g1.width}×{self.g1.height} m, {len(self.piles)} UAV-measured piles")
        self.img = fetch_port_imagery(self.g1, cfg, log)
        self.g15: Grid = self.img["grid15"]
        self.labels15 = np.rint(resample(self.labels1.astype(np.float32), self.g1, self.g15, order=0)).astype(np.int32)
        dist, (ri, ci) = ndimage.distance_transform_edt(self.labels15 == 0, return_indices=True)
        self.zone15 = np.where(dist * self.g15.res <= SEARCH_M, self.labels15[ri, ci], 0).astype(np.int32)
        self.h15 = to_grid(self.h1, self.g1, self.g15)
        self.occ15 = to_grid(self.occ1.astype(np.float32), self.g1, self.g15)
        self.uav = np.array([p["volume_m3"] for p in self.piles])
        self.uav_area = np.array([p["occupied_area_m2"] for p in self.piles])
        self.ref_date = "2025-03-27"
        self.ref_bands = self.img["spot"][self.ref_date]
        self.depth = DepthAnything(log=log)
        self.PX, self.pnames = polygon_features(self.labels15, self.g15.res)

    # ── features ─────────────────────────────────────────────────────────────
    def features(self, bands: Dict) -> tuple:
        rgb = to_uint8_rgb(bands["red"], bands["green"], bands["blue"])
        disp = self.depth.infer(rgb)
        X, names = build_features(bands, self.g15.res, disp)
        return np.dstack([X, self.PX]), names + self.pnames, disp

    # ── chain 1: footprint ───────────────────────────────────────────────────
    def _img(self, X):
        """Image-only features for the footprint classifier: with the polygon
        geometry it learns 'inside the outline', not 'material' (4 % area
        error and a 0 % erase response in the first run)."""
        return X[..., :X.shape[2] - len(self.pnames)]

    def _footprint_area(self, clf, X) -> np.ndarray:
        X = self._img(X)
        ok = np.all(np.isfinite(X), axis=2) & (self.zone15 > 0)
        p = np.zeros(self.zone15.shape, np.float32)
        p[ok] = clf.predict_proba(X[ok])[:, 1]
        return np.array([p[self.zone15 == pl["label"]].sum() * self.g15.pixel_area for pl in self.piles]), p

    def _fit_clf(self, X, mask):
        from sklearn.ensemble import HistGradientBoostingClassifier
        y = (self.occ15 > 0.5).astype(int)
        c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=self.cfg.RAND_SEED)
        Xi = self._img(X)
        c.fit(Xi[mask], y[mask])
        return c

    # ── chain 2: height ──────────────────────────────────────────────────────
    def _fit_reg(self, X, mask):
        from sklearn.ensemble import HistGradientBoostingRegressor
        y = np.clip(np.nan_to_num(self.h15), 0, None)
        r = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=self.cfg.RAND_SEED)
        r.fit(X[mask], y[mask])
        return r

    def _volumes(self, reg, X) -> np.ndarray:
        h = np.zeros(self.labels15.shape, np.float32)
        m = (self.labels15 > 0) & np.all(np.isfinite(X), axis=2)
        h[m] = np.clip(reg.predict(X[m]), 0, None)
        h1 = to_grid(h, self.g15, self.g1, order=1)
        vols = terrain.pile_volumes(np.nan_to_num(h1), self.labels1, self.g1)
        by = {v["label"]: v["volume_m3"] for v in vols}
        return np.array([by.get(p["label"], np.nan) for p in self.piles]), h

    # ── leave-one-pile-out study on 2025 ─────────────────────────────────────
    def study(self) -> Dict:
        X, names, disp = self.features(self.ref_bands)
        self.X_ref, self.names = X, names
        valid = np.all(np.isfinite(X), axis=2) & np.isfinite(self.occ15)
        da_rel = local_relief(np.nan_to_num(disp), self.g15.res, 20.0) if disp is not None else None
        in_piles = self.labels15 > 0
        da_std = float(np.nanstd(da_rel[in_piles])) if da_rel is not None else None
        n = len(self.piles)
        a_loo, v_loo = np.full(n, np.nan), np.full(n, np.nan)
        for i, p in enumerate(self.piles):
            others = self.zone15 != p["label"]
            clf = self._fit_clf(X, valid & (self.zone15 > 0) & others)
            a_loo[i] = self._footprint_area(clf, X)[0][i]
            reg = self._fit_reg(X, valid & in_piles & (self.labels15 != p["label"]) & np.isfinite(self.h15))
            v_loo[i] = self._volumes(reg, X)[0][i]
        res = {"depth_anything_relief_std_in_piles": da_std,
               "area_loo": object_metrics(self.uav_area, a_loo)}
        # chain 1: area → volume, power fitted on the other piles' satellite areas
        v1 = np.full(n, np.nan)
        for i in range(n):
            o = np.arange(n) != i
            fn, _ = self._power_fit(self.uav[o], a_loo[o])
            v1[i] = fn(a_loo[i]) if fn else np.nan
        # chain 2: height volumes, raw and power-corrected on the other piles
        v2c = np.full(n, np.nan)
        for i in range(n):
            o = np.arange(n) != i
            fn, _ = self._power_fit(self.uav[o], v_loo[o])
            v2c[i] = fn(v_loo[i]) if fn else np.nan
        self._v_loo_raw = v_loo
        res["footprint_chain"] = object_metrics(self.uav, v1)
        res["height_chain_raw"] = object_metrics(self.uav, v_loo)
        res["height_chain_power"] = object_metrics(self.uav, v2c)
        self.loo = {"area": a_loo, "v_footprint": v1, "v_height": v_loo, "v_height_power": v2c}
        # production models on all port piles
        self.clf = self._fit_clf(X, valid & (self.zone15 > 0))
        self.reg = self._fit_reg(X, valid & in_piles & np.isfinite(self.h15))
        a_ref, self.prob_ref = self._footprint_area(self.clf, X)
        v_ref, self.hmap_ref = self._volumes(self.reg, X)
        self.fp_fn, self.fp_b = self._power_fit(self.uav, a_ref)
        # power correction only if it beats the raw height volumes out of sample
        self.h_power = res_power_better = (object_metrics(self.uav, v2c)["wape_pct"]
                                           < object_metrics(self.uav, v_loo)["wape_pct"])
        pf, self.h_b = self._power_fit(self.uav, v_ref)
        self.h_fn = pf if res_power_better else (lambda v: v)
        self.a_ref, self.v_ref = a_ref, v_ref
        res["sensitivity"] = self.erase_test(X)
        res["height_uses_power_correction"] = bool(self.h_power)
        if not self.h_power:
            self.loo["v_height_power"] = v_loo           # the chain as used
        return res

    def erase_test(self, X_ref) -> Dict:
        """Largest pile replaced by its surrounding ground in the 2025 image."""
        i = int(np.argmax(self.uav))
        k = self.piles[i]["label"]
        m = self.labels15 == k
        ring = ndimage.binary_dilation(m, iterations=6) & ~ndimage.binary_dilation(m, iterations=2) & (self.labels15 == 0)
        b = {kk: v.copy() for kk, v in self.ref_bands.items()}
        for kk in b:
            b[kk][m] = np.nanmedian(b[kk][ring])
        Xe, _, _ = self.features(b)
        a_e = self._footprint_area(self.clf, Xe)[0][i]
        v_e = self._volumes(self.reg, Xe)[0][i]
        return {"pile_id": self.piles[i]["pile_id"],
                "footprint_response": float(1 - self.fp_fn(a_e) / self.fp_fn(self.a_ref[i])),
                "height_response": float(1 - self.h_fn(v_e) / self.h_fn(self.v_ref[i]))}

    def coverage(self, bands: Dict) -> np.ndarray:
        """Fraction of each pile's 1.5 m pixels with data in every band."""
        ok = np.all(np.stack([np.isfinite(bands[k]) for k in ("blue", "green", "red", "nir")]), axis=0)
        return np.array([ok[self.labels15 == p["label"]].mean() if (self.labels15 == p["label"]).any() else 0.0
                         for p in self.piles])

    def apply(self, date: str, bands: Dict = None, min_coverage: float = None) -> Dict:
        bands = self.img["spot"][date] if bands is None else bands
        excl = ndimage.binary_dilation(self.labels15 > 0, iterations=10)
        nb = _normalise(bands, self.ref_bands, excl)
        X, _, _ = self.features(nb["bands"])
        a, prob = self._footprint_area(self.clf, X)
        v, hmap = self._volumes(self.reg, X)
        vf, vh = np.asarray(self.fp_fn(a), float), np.asarray(self.h_fn(v), float)
        cov = self.coverage(bands)
        if min_coverage:
            # an unimaged pixel predicts 0 m, which would read as a cleared pile
            vf, vh = np.where(cov >= min_coverage, vf, np.nan), np.where(cov >= min_coverage, vh, np.nan)
        return {"date": date, "fit": nb["fit"], "area": a, "prob": prob, "hmap": hmap,
                "v_footprint": vf, "v_height": vh, "coverage": cov}


def run_spot_port(site: Dict, ref: Dict, outdir: str, cfg=StockpileConfig, log=print) -> Dict:
    from .outputs import _jsonable, write_csv
    os.makedirs(outdir, exist_ok=True)
    S = SpotPortStudy(site, ref, cfg, log)
    st = S.study()
    for k in ("area_loo", "footprint_chain", "height_chain_raw", "height_chain_power"):
        m = st[k]
        log(f"  {k:20s} WAPE {m['wape_pct']:.0f}%  bias {m['bias_pct']:+.0f}%  R² {m.get('r2') or 0:.2f}")
    log(f"  Depth Anything relief spread in piles (SPOT 1.5 m): {st['depth_anything_relief_std_in_piles']}")
    log(f"  erase test on {st['sensitivity']['pile_id']}: footprint chain {-100*st['sensitivity']['footprint_response']:+.0f}%, "
        f"height chain {-100*st['sensitivity']['height_response']:+.0f}%")
    # choose: must pass the erase test (≥ 20 %), then lowest LOO WAPE
    cands = []
    if st["sensitivity"]["footprint_response"] >= 0.2:
        cands.append(("footprint", st["footprint_chain"]["wape_pct"]))
    if st["sensitivity"]["height_response"] >= 0.2:
        cands.append(("height", min(st["height_chain_power"]["wape_pct"], st["height_chain_raw"]["wape_pct"])))
    chosen = min(cands, key=lambda c: c[1])[0] if cands else None
    log(f"  chosen chain: {chosen or 'NONE passes the erase test'}")

    # scrap vs bulk (UAV roughness + ortho patchiness at 1 m)
    for p in S.piles:
        o = {k: v.astype(np.float32) / 255.0 for k, v in S.img["ortho"].items()}
        p.update(terrain.material_class(S.dsm1, S.labels1, p["label"], o, S.labels1, S.g1.res))
        # patchiness from a 0.15 m ortho sampled at 1 m is not the S2 10 m measure: re-threshold on roughness
        p["material_class"] = "scrap_metal" if p["surface_roughness_m"] >= cfg.SCRAP_MIN_ROUGHNESS_M else "bulk"
        if p.get("commodity_source") == "client" and p.get("commodity") not in (None, "unknown"):
            # reviewed outlines carry their material; roughness is only the fallback
            p["material_class"] = "scrap_metal" if p["commodity"] == "scrap_metal" else "bulk"

    new = S.apply("2026-04-07")
    rows = []
    for i, p in enumerate(S.piles):
        rows.append({"pile_id": p["pile_id"], "material_class": p["material_class"],
                     "uav_2025_05_01_m3": S.uav[i], "uav_area_m2": S.uav_area[i],
                     "spot_2025_area_m2": S.a_ref[i], "spot_2026_area_m2": new["area"][i],
                     "loo_footprint_2025_m3": S.loo["v_footprint"][i], "loo_height_2025_m3": S.loo["v_height_power"][i],
                     "spot_2026_footprint_m3": new["v_footprint"][i], "spot_2026_height_m3": new["v_height"][i],
                     "spot_2025_height_m3": float(S.h_fn(S.v_ref[i])),
                     # change measured by the same model on both dates; its bias cancels
                     "change_pct": float(100 * (new["v_height"][i] / S.h_fn(S.v_ref[i]) - 1)) if S.v_ref[i] > 0 else None,
                     "uav_scaled_2026_m3": float(S.uav[i] * new["v_height"][i] / S.h_fn(S.v_ref[i])) if S.v_ref[i] > 0 else None})
    transfer = []
    for date, t in S.img.get("transfer", {}).items():
        tr = S.apply(date, t["bands"], min_coverage=0.8)
        seen = 0
        for i, (p, r) in enumerate(zip(S.piles, rows)):
            ok = np.isfinite(tr["v_height"][i]) and S.v_ref[i] > 0
            r[f"{t['sensor']}_{date}_m3"] = float(S.uav[i] * tr["v_height"][i] / S.h_fn(S.v_ref[i])) if ok else None
            r[f"{t['sensor']}_{date}_coverage"] = float(tr["coverage"][i])
            seen += ok
        vals = [(r[f"{t['sensor']}_{date}_m3"], r) for r in rows if r[f"{t['sensor']}_{date}_m3"] is not None]
        uav_seen = sum(r["uav_2025_05_01_m3"] for _, r in vals)
        tot_t = sum(v for v, _ in vals)
        transfer.append({"date": date, "sensor": t["sensor"], "asset": t["asset"], "n_piles_seen": int(seen),
                         "n_piles": len(rows), "total_m3": tot_t, "uav_2025_same_piles_m3": uav_seen,
                         "change_pct": 100 * (tot_t - uav_seen) / uav_seen if uav_seen else None,
                         "scrap_m3": sum(v for v, r in vals if r["material_class"] == "scrap_metal"),
                         "scrap_uav_2025_same_piles_m3": sum(r["uav_2025_05_01_m3"] for _, r in vals
                                                             if r["material_class"] == "scrap_metal"),
                         "radiometric_fit": tr["fit"]})
        log(f"  {t['sensor']} {date} (SPOT-calibrated): {tot_t:,.0f} m³ on {seen}/{len(rows)} piles")
    write_csv(os.path.join(outdir, "port_spot_piles.csv"), _jsonable(rows))
    key = {"footprint": "spot_2026_footprint_m3", "height": "uav_scaled_2026_m3"}.get(chosen)
    tot = {"uav_2025": float(S.uav.sum()),
           "scrap_2025": float(sum(r["uav_2025_05_01_m3"] for r in rows if r["material_class"] == "scrap_metal"))}
    if key:
        tot["spot_2026"] = float(sum(r[key] for r in rows))
        tot["scrap_2026"] = float(sum(r[key] for r in rows if r["material_class"] == "scrap_metal"))
    result = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "study_2025": st, "chosen_chain": chosen, "radiometric_fit_2026": new["fit"],
              "piles": rows, "totals": tot, "transfer": transfer}
    with open(os.path.join(outdir, "port_spot.json"), "w") as fh:
        json.dump(_jsonable(result), fh, indent=2)
    figs = spot_figures(S, new, rows, chosen, st, outdir)
    write_spot_html(os.path.join(outdir, "port_spot_report.html"), result, figs)
    return result


def spot_figures(S, new, rows, chosen, st, outdir) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .outputs import SERIES, SURFACE, INK, INK2, GRID, _style, _cmap
    figs = {}
    g = S.g15
    ext = (g.x0, g.x1, g.y0, g.y1)

    def rgb(b):
        ch = []
        for k in ("red", "green", "blue"):
            a = b[k]
            lo, hi = np.nanpercentile(a, [2, 98])
            ch.append(np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1))
        return np.dstack(ch)
    X, Y = S.g15.xy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE)
    panels = [(rgb(S.ref_bands), "SPOT 2025-03-27"), (rgb(new and S.img["spot"]["2026-04-07"]), "SPOT 2026-04-07")]
    for ax, (im, t) in zip(axes[:2], panels):
        ax.imshow(im, extent=ext)
        ax.contour(X, Y, (S.labels15 > 0).astype(np.float32), levels=[0.5], colors=["#ffffff"], linewidths=1.2)
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    # change as the validated height model sees it (the footprint classifier failed validation)
    d = np.where(S.labels15 > 0, new["hmap"] - S.hmap_ref, np.nan)
    lim = max(float(np.nanpercentile(np.abs(d), 98)), 0.5) if np.isfinite(d).any() else 1.0
    axes[2].imshow(rgb(S.img["spot"]["2026-04-07"]).mean(axis=2), cmap="gray", extent=ext, alpha=0.6)
    im = axes[2].imshow(np.ma.masked_invalid(d), cmap=_cmap(["#2a78d6", "#f0efec", "#e34948"]),
                        vmin=-lim, vmax=lim, extent=ext)
    axes[2].contour(X, Y, (S.labels15 > 0).astype(np.float32), levels=[0.5], colors=[INK], linewidths=0.8)
    axes[2].set_title("Height 2026 − 2025, height model (red higher, blue lower)", loc="left", fontsize=10, color=INK)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], shrink=0.7).set_label("Δ height (m)", color=INK2)
    fig.tight_layout()
    figs["change"] = os.path.join(outdir, "port_spot_change.png")
    fig.savefig(figs["change"], dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # validation scatter (2025 LOO) for both chains
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), facecolor=SURFACE)
    act = np.array([r["uav_2025_05_01_m3"] for r in rows])
    for ax, key, title, m in ((axes[0], "loo_footprint_2025_m3", "Footprint area → volume", st["footprint_chain"]),
                              (axes[1], "loo_height_2025_m3", "Height model" + (" + power correction" if st.get("height_uses_power_correction") else " (raw)"),
                               st["height_chain_power"] if st.get("height_uses_power_correction") else st["height_chain_raw"])):
        _style(ax)
        pr = np.array([r[key] for r in rows])
        top = max(act.max(), np.nanmax(pr)) * 1.15 / 1000
        ax.plot([0, top], [0, top], ls="--", color=INK2, lw=1)
        ax.scatter(act / 1000, pr / 1000, s=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        for a, b_, r in zip(act, pr, rows):
            ax.annotate(r["pile_id"].replace("Pile_", ""), (a / 1000, b_ / 1000), fontsize=7, color=INK2,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        ax.set_xlabel("UAV 2025 (1000 m³)", fontsize=8, color=INK2)
        ax.set_title(f"{title}\nWAPE {m['wape_pct']:.0f}%  bias {m['bias_pct']:+.0f}%  R² {(m.get('r2') or 0):.2f}",
                     loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("SPOT, leave-one-pile-out (1000 m³)", fontsize=8, color=INK2)
    fig.tight_layout()
    figs["validation"] = os.path.join(outdir, "port_spot_validation.png")
    fig.savefig(figs["validation"], dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # per pile 2025 UAV vs 2026 SPOT
    key = {"footprint": "spot_2026_footprint_m3", "height": "uav_scaled_2026_m3"}.get(chosen)
    if key:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
        _style(ax)
        xs = np.arange(len(rows))
        ax.bar(xs - 0.2, [r["uav_2025_05_01_m3"] / 1000 for r in rows], 0.38, color=SERIES[0], label="UAV 2025-05-01",
               edgecolor=SURFACE, linewidth=2)
        ax.bar(xs + 0.2, [r[key] / 1000 for r in rows], 0.38, color=SERIES[1], label="SPOT 2026-04-07",
               edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['pile_id'].replace('Pile_', '')}\n{r['material_class'].replace('_', ' ')}" for r in rows],
                           fontsize=8)
        ax.set_ylabel("Volume (1000 m³)", fontsize=9, color=INK2)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("Port piles — UAV 2025 vs SPOT 2026", loc="left", fontsize=11, color=INK)
        fig.tight_layout()
        figs["piles"] = os.path.join(outdir, "port_spot_piles.png")
        fig.savefig(figs["piles"], dpi=130, facecolor=SURFACE)
        plt.close(fig)
    return figs


def _transfer_html(transfer: List[Dict]) -> str:
    import html
    from .outputs import _fmt
    if not transfer:
        return ""

    def fit(t):
        return ", ".join(f"{k} {v['r2']:.2f}" for k, v in t["radiometric_fit"].items())

    rows = "".join(
        f"<tr><td>{html.escape(t['sensor'])} {html.escape(t['date'])}</td><td class=n>{t['n_piles_seen']}/{t['n_piles']}</td>"
        f"<td class=n>{_fmt(t['uav_2025_same_piles_m3'], 0)}</td><td class=n><b>{_fmt(t['total_m3'], 0)}</b></td>"
        f"<td class=n>{_fmt(t['change_pct'], 0)} %</td><td class=n>{_fmt(t['scrap_uav_2025_same_piles_m3'], 0)} → {_fmt(t['scrap_m3'], 0)}</td>"
        f"<td>{html.escape(fit(t))}</td></tr>"
        for t in transfer)
    return ("<h2>Other sensors through the SPOT model</h2><p class=sub>No Pléiades 2025 image covers the port, so newer "
            "Pléiades scenes are block-averaged to 1.5 m, matched to SPOT 2025 on unchanged pixels and run through the "
            "SPOT-calibrated height model (same ratio to UAV). Cross-sensor: band responses differ, so treat the "
            "matching R² as the reliability signal. Piles less than 80 % imaged are left out; totals are over the piles seen.</p>"
            "<div class=tw><table><thead><tr><th>Scene</th><th>Piles seen</th><th>UAV 2025 (same piles) m³</th>"
            "<th>Now m³</th><th>Change</th><th>Scrap m³ (UAV → now)</th><th>Match R²</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


def write_spot_html(path, result, figs):
    import html
    from .outputs import _img_tag, _fmt
    st, tot, ch = result["study_2025"], result["totals"], result["chosen_chain"]
    key = {"footprint": "spot_2026_footprint_m3", "height": "uav_scaled_2026_m3"}.get(ch)
    rows = "".join(
        f"<tr><td>{html.escape(r['pile_id'])}</td><td>{html.escape(r['material_class'].replace('_', ' '))}</td>"
        f"<td class=n>{_fmt(r['uav_2025_05_01_m3'], 0)}</td><td class=n>{_fmt(r['loo_footprint_2025_m3'], 0)}</td>"
        f"<td class=n>{_fmt(r['loo_height_2025_m3'], 0)}</td><td class=n>{_fmt(r['uav_area_m2'], 0)}</td>"
        f"<td class=n>{_fmt(r['spot_2026_area_m2'], 0)}</td>"
        f"<td class=n>{_fmt(r.get('change_pct'), 0)}</td>"
        f"<td class=n><b>{_fmt(r[key], 0) if key else '–'}</b></td></tr>" for r in result["piles"])
    mrow = lambda n, m: (f"<tr><td>{n}</td><td class=n>{_fmt(m.get('wape_pct'), 0)}</td><td class=n>{_fmt(m.get('bias_pct'), 0)}</td>"
                         f"<td class=n>{_fmt(m.get('r2'), 2)}</td></tr>")
    sens = st["sensitivity"]
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Port SPOT Monitor</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#383835;--card:#242423}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}} main{{max-width:1100px;margin:auto;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin-top:28px}} .sub{{color:var(--ink2)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}} .kpi b{{display:block;font-size:20px}} .kpi span{{color:var(--ink2);font-size:12px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} img{{max-width:100%;border:1px solid var(--line);border-radius:6px}} figure{{margin:14px 0}}
</style></head><body><main>
<h1>Port stock — SPOT 6/7 (1.5 m), area 3</h1>
<div class=sub>Calibrated on SPOT 2025-03-27 against the UAV survey of 2025-05-01; applied to SPOT 2026-04-07 · {html.escape(result['generated_utc'])}</div>
<div class=kpis>
<div class=kpi><b>{_fmt(tot['uav_2025'], 0)} m³</b><span>port, UAV 2025-05-01 (scrap {_fmt(tot['scrap_2025'], 0)})</span></div>
<div class=kpi><b>{_fmt(tot.get('spot_2026'), 0)} m³</b><span>port, SPOT 2026-04-07 (scrap {_fmt(tot.get('scrap_2026'), 0)})</span></div>
<div class=kpi><b>{_fmt(100 * (tot['spot_2026'] - tot['uav_2025']) / tot['uav_2025'], 0) if tot.get('spot_2026') else '–'} %</b><span>change</span></div>
<div class=kpi><b>{html.escape(ch or 'none')}</b><span>chain used (passes erase test, lowest error)</span></div>
</div>
<figure>{_img_tag(figs.get('piles'))}</figure>
<div class=tw><table><thead><tr><th>Pile</th><th>Class</th><th>UAV 2025 m³</th><th>Footprint chain, LOO 2025</th><th>Height chain, LOO 2025</th>
<th>UAV area m²</th><th>SPOT 2026 area m²</th><th>Change % (SPOT 2026 vs SPOT 2025)</th><th>2026 m³ = UAV × change</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class=sub>2026 volume per pile = UAV 2025 volume × (SPOT 2026 ÷ SPOT 2025, same height model): the model's own bias cancels in the ratio.</p>
<h2>Validation on 2025 (leave-one-pile-out, 7 piles)</h2>
<div class=tw><table><thead><tr><th>Chain</th><th>WAPE %</th><th>Bias %</th><th>R²</th></tr></thead><tbody>
{mrow('Footprint area (SPOT) vs UAV area', st['area_loo'])}{mrow('Footprint area → volume', st['footprint_chain'])}
{mrow('Height model, raw', st['height_chain_raw'])}{mrow('Height model + power correction', st['height_chain_power'])}</tbody></table></div>
<p class=sub>Erase test ({html.escape(sens['pile_id'])} replaced by its surrounding ground in the 2025 image): footprint chain
{_fmt(-100 * sens['footprint_response'], 0)} %, height chain {_fmt(-100 * sens['height_response'], 0)} % (a usable chain must lose ≥ 20 %).
Depth Anything on SPOT 1.5 m: relief spread inside piles {_fmt(st['depth_anything_relief_std_in_piles'], 3)}.</p>
<figure>{_img_tag(figs.get('validation'))}</figure>
{_transfer_html(result.get("transfer") or [])}
<h2>What changed between the two SPOT dates</h2>
<figure>{_img_tag(figs.get('change'))}</figure>
<ul>
<li>2025 SPOT is five weeks before the UAV flight: piles that moved in between add to the calibration error.</li>
<li>2026 is matched to 2025 on unchanged pixels (R² per band: {html.escape(', '.join(f"{k} {v['r2']:.2f}" for k, v in result['radiometric_fit_2026'].items()))}).</li>
<li>Volumes are for the 2025 pile outlines. The footprint classifier failed validation at 1.5 m (area error 77 %), so the map shows the height model's change, not footprints.</li>
<li>The two SPOT dates match only moderately (visible bands R² 0.37–0.62; season and processing differ). Treat changes under ~15 % as within noise.</li>
<li>Scrap vs bulk from the UAV surface roughness (≥ 0.6 m = scrap).</li>
<li>Seven piles make the validation an indication, not a guarantee. Pléiades Neo (Aug 2026, pending) at 0.3 m will sharpen footprints; stereo would add height.</li>
</ul></main></body></html>"""
    with open(path, "w") as fh:
        fh.write(doc)
    return path
