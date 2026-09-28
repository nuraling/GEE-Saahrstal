"""End-to-end run: UAV reference → sensor study → thermal → outputs.

``run_pipeline(site, ...)`` works on an in-memory site (from
``sources_ee.load_site`` or ``synthetic.make_site``) so the same code path is
exercised by the tests and by production.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from . import terrain, thermal
from .config import StockpileConfig
from .depth import DepthAnything, to_uint8_rgb
from .features import build_features
from .grid import (Grid, features_from_geojson, lonlat_geom_to_crs, rasterize_labels,
                   resample, to_grid)
from .height_model import fit_predict_oof, pixel_metrics, spatial_folds
from .metrics import iou, object_metrics
from .super_resolution import super_resolve


class Telemetry:
    def __init__(self, project_uuid: str, log=print):
        self.project_uuid = project_uuid
        self._log = log
        self.timings: Dict[str, float] = {}
        self._t: Dict[str, float] = {}

    def __call__(self, msg: str):
        self._log(f"[{dt.datetime.now():%H:%M:%S}] [{self.project_uuid}] {msg}")

    def start(self, name):
        self._t[name] = time.time()
        self(f"▶ {name}")

    def stop(self, name):
        self.timings[name] = round(time.time() - self._t.pop(name, time.time()), 2)
        self(f"✔ {name} ({self.timings[name]} s)")


# ── inputs ───────────────────────────────────────────────────────────────────

def _features_in_crs(fc: Optional[dict], crs: str) -> List[dict]:
    feats = features_from_geojson(fc)
    src = (fc or {}).get("crs_code", "EPSG:4326")
    if src != crs:
        feats = [{**f, "geometry": lonlat_geom_to_crs(f["geometry"], crs)} for f in feats]
    return feats


def _optical_on(grid: Grid, site: Dict) -> Optional[Dict[str, np.ndarray]]:
    """Best available 4-band optical on the UAV grid (Pléiades, else ortho RGB)."""
    ple = site.get("optical", {}).get("pleiades")
    if ple:
        return {k: to_grid(v, ple["grid"], grid) for k, v in ple["bands"].items()}
    u = site["uav"]
    if "red" in u:
        return {k: u[k] for k in ("red", "green", "blue")}
    return None


# ── stage A: UAV reference inventory ─────────────────────────────────────────

def reference_inventory(site: Dict, stockpile_fc, cfg, tel) -> Dict:
    tel.start("UAV reference: nDSM, pile extraction, volumes")
    g: Grid = site["grid"]
    dsm, dtm = site["uav"]["dsm"], site["uav"]["dtm"]
    # DSM_3 (the port) has no DTM: ground there comes from a morphological
    # filter of the DSM. Flagged per pile; its error vs the UAV DTM is reported.
    ground_derived = np.isfinite(dsm) & ~np.isfinite(dtm)
    dtm_check = None
    flat_levels = {}
    if ground_derived.any() and cfg.NO_DTM_GROUND == "flat_min" and "dsm_id" in site["uav"]:
        water = None
        if site.get("s2_raw"):
            b = site["s2_raw"]["bands"]
            ndwi_s2 = (b["green"] - b["nir"]) / (b["green"] + b["nir"] + 1e-6)
            water = resample(ndwi_s2.astype(np.float32), site["s2_raw"]["grid"], g, order=0) > 0.0
        dtm = dtm.copy()
        for blk in np.unique(site["uav"]["dsm_id"][ground_derived]):
            if blk <= 0:
                continue
            m = (site["uav"]["dsm_id"] == blk) & ground_derived
            lvl = terrain.flat_ground(dsm, m, water, cfg.FLAT_GROUND_PERCENTILE)
            dtm[m] = lvl
            flat_levels[f"DSM_{int(blk)}"] = lvl
        tel(f"  flat ground for blocks without a UAV DTM: "
            + ", ".join(f"{k} = {v:.2f} m" for k, v in flat_levels.items()))
        dtm_check = {"method": "flat_min", "levels_m": flat_levels,
                     "percentile": cfg.FLAT_GROUND_PERCENTILE}
    elif ground_derived.any():
        dtm_d = terrain.derive_dtm(dsm, g.res)
        both = np.isfinite(dtm) & np.isfinite(dtm_d)
        if both.any():
            e = dtm_d[both] - dtm[both]
            dtm_check = {"method": "morphological", "bias_m": float(e.mean()),
                         "rmse_m": float(np.sqrt((e ** 2).mean())),
                         "mae_m": float(np.abs(e).mean()), "n_px": int(both.sum())}
        dtm = np.where(np.isfinite(dtm), dtm, dtm_d)
        tel(f"  ground derived from DSM for {ground_derived.sum() * g.pixel_area / 1e4:,.1f} ha without "
            f"a UAV DTM (check vs UAV DTM: {dtm_check})")
    h = terrain.ndsm(dsm, dtm)
    opt = _optical_on(g, site)
    ndvi = ndwi = None
    if opt and "nir" in opt:
        ndvi = (opt["nir"] - opt["red"]) / (opt["nir"] + opt["red"] + 1e-6)
        ndwi = (opt["green"] - opt["nir"]) / (opt["green"] + opt["nir"] + 1e-6)
    det = terrain.detect_piles(h, dsm, g, ndvi, cfg)
    auto_labels = terrain.pile_labels_only(det)

    client_feats = _features_in_crs(stockpile_fc, g.crs)
    if client_feats:
        labels = rasterize_labels([f["geometry"] for f in client_feats], g)
        source = "client_polygons"
        names = [f["properties"].get("name") or f"Pile_{i + 1}" for i, f in enumerate(client_feats)]
        props = [f["properties"] for f in client_feats]
    else:
        labels = auto_labels
        source = "auto_detected"
        names = [f"Pile_{i + 1}" for i in range(int(labels.max()))]
        props = [{} for _ in names]

    h_clean = terrain.remove_thin_structures(h, g.res, cfg.BOOM_REMOVAL_RADIUS_M)
    vols_raw = terrain.pile_volumes(h, labels, g, dsm, "dtm", cfg.SIGMA_Z_UAV_M)
    vols_dtm = terrain.pile_volumes(h_clean, labels, g, dsm, "dtm", cfg.SIGMA_Z_UAV_M)
    vols_toe = terrain.pile_volumes(h_clean, labels, g, dsm - (h - np.nan_to_num(h_clean)), "toe",
                                    cfg.SIGMA_Z_UAV_M)
    n_lab = int(labels.max())
    cov = (ndimage.mean(np.isfinite(h).astype(np.float32), labels, np.arange(1, n_lab + 1))
           if n_lab else np.array([]))
    p99 = {}
    for k in range(1, n_lab + 1):
        v = h_clean[(labels == k) & np.isfinite(h_clean)]
        p99[k] = float(np.percentile(v, 99)) if v.size else None
    # material occupies the polygon only where it stands above ground
    occupied = (labels > 0) & np.isfinite(h_clean) & (h_clean > cfg.MIN_PILE_HEIGHT_M)

    piles = []
    refl = terrain.pseudo_reflectance(opt) if opt else None
    colours = {}
    for v in vols_dtm:
        k = v["label"]
        m = (labels == k) & occupied
        if refl and m.any():
            colours[k] = {b: float(np.nanmedian(refl[b][m])) for b in refl}
    client_com = {i + 1: p.get("commodity") for i, p in enumerate(props) if p.get("commodity")}
    learned = terrain.commodity_from_examples(colours, client_com) if client_com else {}
    for v, vt, vr in zip(vols_dtm, vols_toe, vols_raw):
        k = v["label"]
        m = labels == k
        occ_area = float((occupied & m).sum() * g.pixel_area)
        measured = bool(cov[k - 1] >= cfg.MIN_UAV_COVERAGE)
        rec = {"pile_id": names[k - 1], "label": k, "source": source,
               "ground_source": (("flat_min_level" if cfg.NO_DTM_GROUND == "flat_min" else "derived_from_dsm")
                                 if ground_derived[m].mean() > 0.5 else "uav_dtm")
               if m.any() else None,
               "survey_block": (f"DSM_{int(np.nanmedian(site['uav']['dsm_id'][m]))}"
                                if "dsm_id" in site["uav"] and m.any()
                                and np.nanmedian(site["uav"]["dsm_id"][m]) > 0 else None),
               "uav_coverage": float(cov[k - 1]), "uav_measured": measured,
               "footprint_area_m2": v["area_m2"],
               "occupied_area_m2": occ_area if measured else None,
               "volume_m3": v["volume_m3"] if measured else None,
               "volume_toe_base_m3": vt["volume_m3"] if measured else None,
               "volume_raw_incl_structures_m3": vr["volume_m3"] if measured else None,
               "volume_sigma_m3": v["volume_sigma_m3"] if measured else None,
               "mean_height_m": (v["volume_m3"] / occ_area if occ_area else 0.0) if measured else None,
               "p99_height_m": p99.get(k) if measured else None,
               "max_height_m": v["max_height_m"] if measured else None}
        col = colours.get(k)
        auto = (terrain.classify_commodity(col)
                if col and all(np.isfinite(list(col.values()))) else "unknown")
        if client_com.get(k):
            rec["commodity"], rec["commodity_source"] = client_com[k], "client"
        elif k in learned:
            rec["commodity"], rec["commodity_source"] = learned[k], "learned_from_client_labels"
        else:
            rec["commodity"], rec["commodity_source"] = auto, "spectral_rule"
        rec["commodity_auto"] = auto
        rec["tonnage_t"] = terrain.tonnage(rec["volume_m3"], rec["commodity"]) if measured else None
        rec["bulk_density_t_m3"] = cfg.BULK_DENSITY_T_M3.get(rec["commodity"], 1.0)
        for pk, pv in props[k - 1].items():
            rec.setdefault(f"client_{pk}", pv)
        piles.append(rec)
    tel(f"  {len(piles)} piles ({source}), {sum(p['uav_measured'] for p in piles)} inside UAV coverage; "
        f"{sum(p['volume_m3'] or 0 for p in piles):,.0f} m³ measured; "
        f"auto-detection found {int(auto_labels.max())} pile objects, "
        f"{sum(1 for c in det['classes'].values() if c == terrain.BUILDING)} structures")
    tel.stop("UAV reference: nDSM, pile extraction, volumes")
    return {"dtm_check": dtm_check, "ground_derived": ground_derived,
            "ndsm": h_clean, "ndsm_raw": h, "labels": labels, "auto_labels": auto_labels, "detection": det,
            "occupied": occupied, "piles": piles, "ndvi": ndvi, "ndwi": ndwi, "source": source}


# ── stage B: sensor study ────────────────────────────────────────────────────

def _sensor_inputs(site: Dict, cfg, tel, sensors: List[str], folds_ref: Grid) -> Dict:
    out = {}
    opt = site.get("optical", {})
    if "pleiades" in sensors and "pleiades" in opt:
        out["pleiades"] = opt["pleiades"]
    if "spot" in sensors and "spot" in opt:
        out["spot"] = opt["spot"]
    s2 = site.get("s2_raw")
    if s2:
        bands10 = {k: v for k, v in s2["bands"].items() if k in cfg.OPTICAL_BANDS}
        if "s2_10m" in sensors:
            out["s2_10m"] = {"grid": s2["grid"], "bands": bands10}
        for name in ("s2_sr_1m", "s2_sr_3m"):
            if name not in sensors:
                continue
            gh = site["grid"].with_res(cfg.SENSORS[name]["grid_m"])
            ref = None
            if "pleiades" in opt:
                ref = {k: to_grid(v, opt["pleiades"]["grid"], gh) for k, v in opt["pleiades"]["bands"].items()}
                # reference SR is trained in reflectance-like units: put Pléiades on S2's scale
                ref = {k: v * (np.nanmedian(bands10[k]) / max(np.nanmedian(v), 1e-9)) for k, v in ref.items()}
            folds = spatial_folds(gh, cfg.CV_BLOCK_M, cfg.CV_FOLDS, cfg.RAND_SEED, folds_ref)
            tel.start(f"Sentinel-2 super-resolution → {name}")
            sr = super_resolve(bands10, s2["grid"], gh, ref, folds, cfg.S2_SR_METHOD, log=tel)
            tel.stop(f"Sentinel-2 super-resolution → {name}")
            out[name] = {"grid": gh, "bands": sr["bands"], "sr_method": sr["method"],
                         "sr_metrics": sr["metrics"]}
    return out


def polygon_features(labels_s: np.ndarray, res: float):
    """Where a pixel sits inside its pile polygon: distance to the edge (m),
    the same relative to the polygon's deepest point, and polygon size.
    Legitimate in known-footprint mode — the polygon is an input."""
    inside = labels_s > 0
    dist = ndimage.distance_transform_edt(inside) * res
    n = int(labels_s.max())
    if n == 0:
        z = np.zeros(labels_s.shape, np.float32)
        return np.dstack([z, z, z]), ["poly_edge_dist_m", "poly_rel_depth", "poly_log_area"]
    idx = np.arange(1, n + 1)
    dmax = np.concatenate([[1.0], np.maximum(ndimage.maximum(dist, labels_s, idx), res)])
    area = np.concatenate([[1.0], ndimage.sum(inside, labels_s, idx) * res * res])
    rel = dist / dmax[labels_s]
    la = np.where(inside, np.log(np.maximum(area[labels_s], 1.0)), 0.0)
    return (np.dstack([dist, rel, la]).astype(np.float32),
            ["poly_edge_dist_m", "poly_rel_depth", "poly_log_area"])


def bias_correct(v_true: np.ndarray, v_pred: np.ndarray, pile_fold: np.ndarray) -> np.ndarray:
    """Volume bias correction without leakage: piles in fold f are scaled by
    Σ actual / Σ predicted over the piles of the OTHER folds."""
    out = np.full_like(v_pred, np.nan, dtype=float)
    ok = np.isfinite(v_true) & np.isfinite(v_pred)
    for f in np.unique(pile_fold[np.isfinite(v_pred)]):
        tr = ok & (pile_fold != f)
        if tr.sum() < 3 or v_pred[tr].sum() <= 0:
            continue
        out[pile_fold == f] = v_pred[pile_fold == f] * v_true[tr].sum() / v_pred[tr].sum()
    return out


def _quad_fit(v_true: np.ndarray, v_pred: np.ndarray):
    """v_true ≈ a·v + b·v² (through the origin), then scaled so the corrected
    total equals the actual total on the fitting piles (mean matching).
    Falls back to the ratio when the curve would not be increasing."""
    ok = np.isfinite(v_true) & np.isfinite(v_pred) & (v_pred > 0)
    x, y = v_pred[ok], v_true[ok]
    if x.size < 4 or x.sum() <= 0:
        return None
    ratio = y.sum() / x.sum()
    A = np.c_[x, x * x]
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    xmax = x.max() * 1.5
    if a <= 0 or (a + 2 * b * xmax) <= 0:          # not monotonic over the range
        return lambda v: ratio * v
    f = lambda v: a * v + b * v * v
    k = y.sum() / max(f(x).sum(), 1e-9)
    return lambda v: k * f(v)


def _power_fit(v_true: np.ndarray, v_pred: np.ndarray):
    """v_true ≈ c·v^b fitted in log space, then mean-matched (Σ corrected =
    Σ actual on the fitting piles). The exponent measures the curvature: a
    model that shrinks big piles towards the average gives b > 1."""
    ok = np.isfinite(v_true) & np.isfinite(v_pred) & (v_pred > 0) & (v_true > 0)
    x, y = v_pred[ok], v_true[ok]
    if x.size < 4:
        return None, None
    b, la = np.polyfit(np.log(x), np.log(y), 1)
    b = float(np.clip(b, 0.5, 2.5))
    f = lambda v: np.exp(la) * np.power(np.maximum(v, 0), b)
    k = y.sum() / max(f(x).sum(), 1e-9)
    return (lambda v: k * f(v)), b


def power_correct(v_true: np.ndarray, v_pred: np.ndarray, pile_fold: np.ndarray):
    out = np.full_like(v_pred, np.nan, dtype=float)
    exps = []
    for f in np.unique(pile_fold[np.isfinite(v_pred)]):
        tr = pile_fold != f
        fn, b = _power_fit(np.where(tr, v_true, np.nan), np.where(tr, v_pred, np.nan))
        if fn is not None:
            sel = pile_fold == f
            out[sel] = fn(v_pred[sel])
            exps.append(b)
    return out, (float(np.median(exps)) if exps else None)


def port_local_correction(ref: Dict, study: Dict) -> None:
    """Power-law bias correction of the port hold-out, calibrated with the
    port's own piles, leave-one-out: each port pile is corrected by c·v^b
    fitted (mean-matched) on the OTHER port piles only. The operational
    analogue: a few UAV-measured piles calibrate the satellite model in a
    new area. The steel-works-only correction cannot remove the port bias,
    because the port behaves differently."""
    port = np.array(study.get("in_port", []), bool)
    if port.sum() < 4:
        return
    act = np.array([np.nan if p["volume_m3"] is None else p["volume_m3"] for p in ref["piles"]])
    from .metrics import object_metrics
    for name, sr in study["results"].items():
        for m, r in (sr.get("port_holdout") or {}).items():
            if not isinstance(r, dict) or not r.get("n_port_piles_seen"):
                continue
            vp = np.array([p.get(f"port_{name}_{m}_m3") or np.nan for p in ref["piles"]], float)
            idx = np.flatnonzero(port & np.isfinite(vp) & np.isfinite(act))
            if idx.size < 4:
                continue
            out, exps = np.full(len(act), np.nan), []
            for i in idx:
                others = np.setdiff1d(idx, [i])
                fn, b = _power_fit(act[others], vp[others])
                if fn is not None:
                    out[i] = fn(vp[i])
                    exps.append(b)
            r["power_corrected_port_loo"] = object_metrics(act[idx], out[idx])
            r["power_exponent_port_loo"] = float(np.median(exps)) if exps else None
            for i in idx:
                ref["piles"][i][f"port_{name}_{m}_pl_m3"] = float(out[i]) if np.isfinite(out[i]) else None


def quadratic_correct(v_true: np.ndarray, v_pred: np.ndarray, pile_fold: np.ndarray) -> np.ndarray:
    """Non-linear bias correction, fold-wise: a pile is corrected by the curve
    fitted on the other folds' piles only."""
    out = np.full_like(v_pred, np.nan, dtype=float)
    for f in np.unique(pile_fold[np.isfinite(v_pred)]):
        tr = pile_fold != f
        fn = _quad_fit(np.where(tr, v_true, np.nan), np.where(tr, v_pred, np.nan))
        if fn is not None:
            sel = pile_fold == f
            out[sel] = fn(v_pred[sel])
    return out


def _estimate_sensor(inputs: Dict) -> Optional[str]:
    for s_ in ("pleiades", "spot", "s2_sr_3m", "s2_sr_1m", "s2_10m"):
        if s_ in inputs:
            return s_
    return None


METHODS = (
    # name, height_model method, known footprint?
    ("original_rf", "rf_spectral", False),   # v1: RF on bands, whole site
    ("gbm_site", "gbm", False),              # all features, whole site
    ("gbm_pile", "gbm", True),               # all features + polygon geometry, inside polygons
    ("depth_quadratic", "depth_affine", True),  # Depth Anything alone, quadratic calibration
)


def _pile_values(labels, g, n):
    rows = {}
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is not None:
            rr, cc = np.nonzero(labels[sl] == k)
            rows[k] = (sl[0].start + int(np.median(rr)), sl[1].start + int(np.median(cc)))
    return rows


def sensor_study(site: Dict, ref: Dict, cfg, tel, sensors: List[str],
                 depth: Optional[DepthAnything]) -> Dict:
    g: Grid = site["grid"]
    h, labels = ref["ndsm"], ref["labels"]
    occupied = ref["occupied"]
    region = ndimage.binary_dilation(labels > 0, iterations=int(round(5 / g.res)))
    inputs = _sensor_inputs(site, cfg, tel, sensors, g)
    results, maps = {}, {}
    skipped = {s: "no imagery for this source in the site bundle" for s in sensors if s not in inputs}
    for s_, why in skipped.items():
        tel(f"  sensor '{s_}' skipped: {why}")
    ref_vol = np.array([np.nan if p["volume_m3"] is None else p["volume_m3"] for p in ref["piles"]])
    ref_area = np.array([np.nan if p["occupied_area_m2"] is None else p["occupied_area_m2"]
                         for p in ref["piles"]])
    n = int(labels.max())
    folds_g = spatial_folds(g, cfg.CV_BLOCK_M, cfg.CV_FOLDS, cfg.RAND_SEED)
    centres = _pile_values(labels, g, n)
    pile_fold = np.array([folds_g[centres[k]] if k in centres else -1 for k in range(1, n + 1)])
    dsm_id = site["uav"].get("dsm_id")
    port_block = cfg.PORT_SURVEY_BLOCK
    in_port = np.array([p.get("survey_block") == f"DSM_{port_block}" for p in ref["piles"]])

    for name, inp in inputs.items():
        tel.start(f"Sensor study: {name}")
        gs: Grid = inp["grid"]
        hs = to_grid(h, g, gs)
        ms = to_grid(occupied.astype(np.float32), g, gs) > 0.5
        labels_s = np.rint(resample(labels.astype(np.float32), g, gs, order=0)).astype(np.int32)
        disp = None
        da_status = "not attempted (grid coarser than 3 m)"
        if depth is not None and gs.res <= 3.0 and all(k in inp["bands"] for k in ("red", "green", "blue")):
            rgb = to_uint8_rgb(inp["bands"]["red"], inp["bands"]["green"], inp["bands"]["blue"])
            disp = depth.infer(rgb)
            da_status = depth.status if disp is None else f"used ({depth.backend})"
        X, names = build_features(inp["bands"], gs.res, disp)
        PX, pnames = polygon_features(labels_s, gs.res)
        folds = spatial_folds(gs, cfg.CV_BLOCK_M, cfg.CV_FOLDS, cfg.RAND_SEED, g)
        coverage = float(np.mean([np.isfinite(X[..., 0][labels_s == k]).mean() > 0.8
                                  for k in range(1, n + 1) if (labels_s == k).any()])) if n else 0.0
        sres = {"grid_m": gs.res, "native_m": cfg.SENSORS.get(name, {}).get("native_m", gs.res),
                "depth_anything": da_status, "pile_coverage": coverage, "methods": {}}
        if "sr_method" in inp:
            sres["super_resolution"] = {"method": inp["sr_method"], "vs_pleiades": inp["sr_metrics"]}
        for label, method, pile_mode in METHODS:
            Xm, nm = (np.dstack([X, PX]), names + pnames) if pile_mode else (X, names)
            in_poly = (labels_s > 0) if pile_mode else None
            r = fit_predict_oof(Xm, hs, ms, folds, nm, method, cfg, tel, in_poly=in_poly)
            if not r.get("available"):
                sres["methods"][label] = {"available": False, "reason": r.get("reason")}
                continue
            h_up = to_grid(r["height_oof"], gs, g, order=1)
            vols = terrain.pile_volumes(h_up, labels, g) if n else []
            vpred = np.array([v["volume_m3"] for v in vols], float)
            # a pile the sensor does not see gets no prediction
            seen = np.array([np.isfinite(h_up[labels == k]).mean() > 0.8 if (labels == k).any() else False
                             for k in range(1, n + 1)])
            vpred[~seen] = np.nan
            vbc = bias_correct(ref_vol, vpred, pile_fold)
            vqc = quadratic_correct(ref_vol, vpred, pile_fold)
            vpc, pexp = power_correct(ref_vol, vpred, pile_fold)
            mres = {"available": True, "known_footprint": pile_mode, "features": r["features_used"],
                    "pixel_height_oof": pixel_metrics(r["height_oof"], hs, ms),
                    "pile_volume_known_footprint": object_metrics(ref_vol, vpred),
                    "pile_volume_bias_corrected": object_metrics(ref_vol, vbc),
                    "pile_volume_quadratic_corrected": object_metrics(ref_vol, vqc),
                    "pile_volume_power_corrected": object_metrics(ref_vol, vpc),
                    "power_exponent": pexp}
            if not pile_mode:
                p_up = to_grid(r["prob_oof"], gs, g, order=1)
                fp_pred = (p_up > 0.5) & region
                area_pred = ndimage.sum(fp_pred, labels, np.arange(1, n + 1)) * g.pixel_area if n else np.array([])
                vsat = (ndimage.sum(np.where(fp_pred, np.clip(np.nan_to_num(h_up), 0, None), 0),
                                    labels, np.arange(1, n + 1)) * g.pixel_area if n else np.array([]))
                mres.update({"pile_volume_satellite_only": object_metrics(ref_vol, vsat),
                             "pile_area": object_metrics(ref_area, area_pred),
                             "footprint_iou": iou(fp_pred, occupied)})
            if r.get("in_sample_metrics"):
                mres["pixel_height_in_sample_(original_method)"] = r["in_sample_metrics"]
            sres["methods"][label] = mres
            for i, p in enumerate(ref["piles"]):
                p[f"vol_{name}_{label}_m3"] = float(vpred[i]) if np.isfinite(vpred[i]) else None
                p[f"vol_{name}_{label}_bc_m3"] = float(vbc[i]) if np.isfinite(vbc[i]) else None
                p[f"vol_{name}_{label}_qc_m3"] = float(vqc[i]) if np.isfinite(vqc[i]) else None
                p[f"vol_{name}_{label}_pc_m3"] = float(vpc[i]) if np.isfinite(vpc[i]) else None
            if label == "gbm_pile":
                maps[name] = {"grid": gs, "height_oof": r["height_oof"], "prob_oof": r["prob_oof"]}
                # overall correction factor (for applying the model to new imagery)
                ok = np.isfinite(ref_vol) & np.isfinite(vpred)
                sres["volume_correction_factor"] = (float(ref_vol[ok].sum() / vpred[ok].sum())
                                                    if ok.any() and vpred[ok].sum() > 0 else 1.0)

        # ── the port hold-out: train on the other survey blocks, test on DSM_3
        if dsm_id is not None and in_port.any():
            port_s = np.rint(resample(dsm_id.astype(np.float32), g, gs, order=0)) == port_block
            pfolds = port_s.astype(np.int16)                  # 1 = port (test), 0 = train
            port_res = {}
            for label, method, pile_mode in METHODS:
                if label not in ("original_rf", "gbm_pile", "depth_quadratic"):
                    continue
                Xm, nm = (np.dstack([X, PX]), names + pnames) if pile_mode else (X, names)
                r = fit_predict_oof(Xm, hs, ms, pfolds, nm, method, cfg, tel,
                                    in_poly=(labels_s > 0) if pile_mode else None, only_folds=[1])
                if not r.get("available"):
                    continue
                h_up = to_grid(r["height_oof"], gs, g, order=1)
                vp = np.array([v["volume_m3"] for v in terrain.pile_volumes(h_up, labels, g)], float)
                vp[~in_port] = np.nan
                seen = np.array([np.isfinite(h_up[labels == k]).mean() > 0.8 for k in range(1, n + 1)])
                vp[~seen] = np.nan
                # correction learned only from the non-port piles (their CV predictions)
                cv_pred = np.array([p.get(f"vol_{name}_{label}_m3") or np.nan for p in ref["piles"]], float)
                okc = ~in_port & np.isfinite(cv_pred) & np.isfinite(ref_vol)
                a = float(ref_vol[okc].sum() / cv_pred[okc].sum()) if okc.any() and cv_pred[okc].sum() > 0 else 1.0
                qfn = _quad_fit(np.where(okc, ref_vol, np.nan), np.where(okc, cv_pred, np.nan))
                vq = qfn(vp) if qfn is not None else a * vp
                pfn, pb = _power_fit(np.where(okc, ref_vol, np.nan), np.where(okc, cv_pred, np.nan))
                vpw = pfn(vp) if pfn is not None else a * vp
                port_res[label] = {"n_port_piles_seen": int(np.isfinite(vp).sum()),
                                   "raw": object_metrics(ref_vol[in_port], vp[in_port]),
                                   "bias_corrected": object_metrics(ref_vol[in_port], a * vp[in_port]),
                                   "quadratic_corrected": object_metrics(ref_vol[in_port], vq[in_port]),
                                   "power_corrected": object_metrics(ref_vol[in_port], vpw[in_port]),
                                   "power_exponent": pb,
                                   "correction_factor": a,
                                   "pixel_height": pixel_metrics(r["height_oof"], hs, ms & port_s)}
                for i, p in enumerate(ref["piles"]):
                    if in_port[i]:
                        p[f"port_{name}_{label}_m3"] = float(vp[i]) if np.isfinite(vp[i]) else None
                        p[f"port_{name}_{label}_bc_m3"] = float(a * vp[i]) if np.isfinite(vp[i]) else None
                        p[f"port_{name}_{label}_qc_m3"] = float(vq[i]) if np.isfinite(vq[i]) else None
                        p[f"port_{name}_{label}_pc_m3"] = float(vpw[i]) if np.isfinite(vpw[i]) else None
            sres["port_holdout"] = port_res or {"note": "sensor does not cover the port"}
        results[name] = sres
        tel.stop(f"Sensor study: {name}")
    out = {"results": results, "maps": maps, "skipped": skipped, "in_port": in_port.tolist()}
    port_local_correction(ref, out)
    return out


# ── stage B2: apply the validated models to recent scenes (S2, Pléiades) ───

def _normalise_to_reference(bands: Dict, ref_bands: Dict, exclude: np.ndarray) -> Dict:
    """Map a new scene onto the reference scene's radiometry with a per-band
    linear fit over pseudo-invariant pixels (not piles, not vegetation, not
    water, NDVI unchanged)."""
    nd_new = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"] + 1e-6)
    nd_ref = (ref_bands["nir"] - ref_bands["red"]) / (ref_bands["nir"] + ref_bands["red"] + 1e-6)
    inv = (~exclude & np.isfinite(nd_new) & np.isfinite(nd_ref) & (np.abs(nd_new - nd_ref) < 0.05)
           & (nd_ref < 0.3) & (nd_ref > 0.0))
    out, fit = {}, {}
    for k in ("blue", "green", "red", "nir"):
        x, y = bands[k][inv], ref_bands[k][inv]
        A = np.c_[x, np.ones_like(x)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        keep = np.abs(r) < 2 * r.std()
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        pred = A[keep] @ coef
        ss = ((y[keep] - y[keep].mean()) ** 2).sum()
        fit[k] = {"gain": float(coef[0]), "offset": float(coef[1]), "n_px": int(keep.sum()),
                  "r2": float(1 - ((y[keep] - pred) ** 2).sum() / ss) if ss > 0 else None}
        out[k] = coef[0] * bands[k] + coef[1]
    return {"bands": out, "fit": fit}


def _pile_coverage(bands: Dict, labels_s: np.ndarray, n: int) -> np.ndarray:
    """Fraction of each pile's pixels with data in every band."""
    ok = np.all(np.stack([np.isfinite(bands[k]) for k in ("blue", "green", "red", "nir")]), axis=0)
    if not n:
        return np.array([])
    tot = ndimage.sum(np.ones_like(ok, np.float32), labels_s, np.arange(1, n + 1))
    return np.where(tot > 0, ndimage.sum(ok.astype(np.float32), labels_s, np.arange(1, n + 1)) / np.maximum(tot, 1), 0.0)


def _apply_to_recent(site: Dict, ref: Dict, cfg, tel, *, ref_bands: Dict, gs: Grid, recent: List[Dict],
                     val: Dict, model_name: str, key: str, ref_date, min_coverage: Optional[float] = None) -> Dict:
    """A validated known-footprint model applied to recent scenes of the same
    sensor: pile volumes now vs the UAV survey. With `min_coverage`, a pile
    covered by less than that fraction on either date gets no estimate
    (Pléiades stores outside-the-scene as no-data over parts of the site)."""
    from .height_model import fit_full_mode
    g: Grid = site["grid"]
    labels = ref["labels"]
    n = int(labels.max())
    labels_s = np.rint(resample(labels.astype(np.float32), g, gs, order=0)).astype(np.int32)
    X0, names = build_features(ref_bands, gs.res)
    PX, pnames = polygon_features(labels_s, gs.res)
    hs = to_grid(ref["ndsm"], g, gs)
    reg = fit_full_mode(np.dstack([X0, PX]), hs, labels_s > 0, cfg)
    a = val.get("volume_correction_factor", 1.0)
    excl = ndimage.binary_dilation(labels_s > 0, iterations=2)
    out = {"model": model_name, "correction_factor": a,
           "validation": {"cv": val["methods"]["gbm_pile"].get("pile_volume_bias_corrected"),
                          "port": val.get("port_holdout", {}).get("gbm_pile", {}).get("bias_corrected")},
           "reference_date": ref_date, "scenes": []}

    def volumes(bands):
        X, _ = build_features(bands, gs.res)
        Xm = np.dstack([X, PX])
        hp = np.clip(reg.predict(np.nan_to_num(Xm.reshape(-1, Xm.shape[2]))).reshape(gs.shape), 0, None)
        hp = np.where(labels_s > 0, hp, 0).astype(np.float32)
        h_up = to_grid(hp, gs, g, order=1)
        return np.array([v["volume_m3"] for v in terrain.pile_volumes(h_up, labels, g)])

    # Change, not absolute level: the production model has seen these piles,
    # so its absolute volumes on the reference date are not independent.
    # Each pile's volume now = UAV volume × (model now / model on the
    # reference scene); the model's own bias cancels in the ratio.
    v_ref = volumes(ref_bands)
    cov_ref = _pile_coverage(ref_bands, labels_s, n) if min_coverage else None
    uav = np.array([np.nan if p["volume_m3"] is None else p["volume_m3"] for p in ref["piles"]])
    for sc in recent:
        nb = _normalise_to_reference({k: sc["bands"][k] for k in cfg.OPTICAL_BANDS}, ref_bands, excl)
        v_now = volumes(nb["bands"])
        ratio = np.where(v_ref > 1, v_now / np.maximum(v_ref, 1), np.nan)
        est = np.where(np.isfinite(uav) & np.isfinite(ratio), uav * ratio, a * v_now)
        extra = {}
        if min_coverage:
            seen = (cov_ref >= min_coverage) & (_pile_coverage(sc["bands"], labels_s, n) >= min_coverage)
            ratio, est = np.where(seen, ratio, np.nan), np.where(seen, est, np.nan)
            extra = {"n_piles_seen": int(seen.sum()), "n_piles": n}
        both = np.isfinite(uav) & np.isfinite(est)
        # totals over piles seen on both dates, so a partial scene is not read as a drop
        tot_uav = float(np.nansum(np.where(both, uav, np.nan)))
        tot_now = float(np.nansum(np.where(both, est, np.nan)))
        none_seen = not both.any()                 # e.g. the scenes do not overlap: no estimate, not zero
        scene = {"date": sc["date"], "radiometric_fit": nb["fit"],
                 "total_m3": None if none_seen else tot_now, "total_ref_m3": None if none_seen else tot_uav,
                 "change_pct": float(100 * (tot_now - tot_uav) / tot_uav) if tot_uav and not none_seen else None,
                 "model_total_ref_m3": float(np.nansum(v_ref)),
                 "model_total_now_m3": float(np.nansum(v_now)), **extra}
        for w in ("band_check", "asset"):
            if sc.get(w):
                scene[w] = sc[w]
        out["scenes"].append(scene)
        for i, p in enumerate(ref["piles"]):
            p[f"{key}_{sc['date']}_m3"] = float(est[i]) if np.isfinite(est[i]) else None
            p[f"{key}_{ref_date}_m3"] = float(uav[i]) if np.isfinite(uav[i]) else float(a * v_ref[i])
            ck = f"change_{sc['date']}_pct" if key == "s2" else f"change_{key}_{sc['date']}_pct"
            p[ck] = float(100 * (ratio[i] - 1)) if np.isfinite(ratio[i]) else None
    tel(f"  {key} volumes (UAV-measured piles): " + ", ".join(
        f"{s_['date']}: no piles imaged on both dates" if s_["total_m3"] is None else
        f"{s_['date']}: {s_['total_m3']:,.0f} m³ ({(s_['change_pct'] or 0):+.0f}% vs UAV)"
        for s_ in out["scenes"]))
    return out


def recent_sentinel(site: Dict, ref: Dict, study: Dict, cfg, tel) -> Dict:
    """The Sentinel-2 10 m known-footprint model, validated above, applied to
    recent scenes: pile volumes now vs the UAV survey."""
    s2, recent = site.get("s2_raw"), site.get("s2_recent") or []
    val = study["results"].get("s2_10m", {})
    if not s2 or not recent or "gbm_pile" not in val.get("methods", {}):
        return {}
    tel.start("Sentinel-2 recent scenes: pile volumes now")
    out = _apply_to_recent(site, ref, cfg, tel,
                           ref_bands={k: v for k, v in s2["bands"].items() if k in cfg.OPTICAL_BANDS},
                           gs=s2["grid"], recent=recent, val=val, model_name="s2_10m gbm_pile",
                           key="s2", ref_date=s2.get("date"))
    tel.stop("Sentinel-2 recent scenes: pile volumes now")
    return out


def recent_pleiades(site: Dict, ref: Dict, study: Dict, cfg, tel) -> Dict:
    """The Pléiades known-footprint model (reference: pleiades_saarfactory
    against the UAV survey) applied to newer Pléiades scenes, e.g. 2026-08-03.
    Only piles imaged on both dates get an estimate."""
    ple, recent = site.get("optical", {}).get("pleiades"), site.get("pleiades_recent") or []
    val = study["results"].get("pleiades", {})
    if not ple or not recent or "gbm_pile" not in val.get("methods", {}):
        return {}
    tel.start("Pléiades recent scenes: pile volumes now")
    out = _apply_to_recent(site, ref, cfg, tel,
                           ref_bands={k: v for k, v in ple["bands"].items() if k in cfg.OPTICAL_BANDS},
                           gs=ple["grid"], recent=recent, val=val, model_name="pleiades gbm_pile",
                           key="ple", ref_date="ref", min_coverage=0.8)
    tel.stop("Pléiades recent scenes: pile volumes now")
    return out


# ── stage C: thermal ─────────────────────────────────────────────────────────

def thermal_stage(site: Dict, ref: Dict, building_fc, cfg, tel) -> Dict:
    g: Grid = site["grid"]
    scenes = site.get("thermal") or []
    out = {"n_scenes": len(scenes), "piles": [], "buildings": [], "per_scene": []}
    if not scenes:
        tel("No thermal scenes — thermal stage skipped.")
        return out
    tel.start("Thermal: per-pile contrast, persistence, hot-spot bound")
    g30: Grid = site["thermal_grid"]
    det = ref["detection"]
    structures = np.isin(det["labels"], [k for k, c in det["classes"].items()
                                         if c in (terrain.BUILDING, terrain.VEGETATION)])
    exclude = structures | (ref["labels"] > 0)
    if ref["ndwi"] is not None:
        exclude |= ref["ndwi"] > 0.1       # the harbour basin: cool by day, warm at night
    res = thermal.analyse_scenes(scenes, g30, ref["labels"], g, exclude, "Pile", cfg)
    names = {p["label"]: p["pile_id"] for p in ref["piles"]}
    for s in res["summary"]:
        s["object_id"] = names.get(s["label"], s["object_id"])
    out["piles"] = res["summary"]
    out["per_scene"] = res["per_scene"]

    day = [s["lst"] for s in scenes if not s.get("night")]
    if day:
        comp = np.nanmedian(np.stack(day), axis=0)
        out["composite_lst"] = comp
        opt = _optical_on(g, site)
        if opt and "nir" in opt:
            g3 = g.with_res(3.0)
            b3 = {k: to_grid(v, g, g3) for k, v in opt.items()}
            ds = thermal.downscale_lst(comp, g30, b3, g3, cfg)
            out["downscaled_lst"] = {"grid": g3, **ds}

    # ── the thermal product: building assets (roofs, chimneys/stacks, pipes)
    bfeats = _features_in_crs(building_fc, g.crs)
    st = terrain.detect_structures(ref["ndsm_raw"], g, ref["ndvi"])
    if bfeats:
        blabels = rasterize_labels([f["geometry"] for f in bfeats], g)
        bclass = {i + 1: f["properties"].get("class", "building") for i, f in enumerate(bfeats)}
        out["asset_source"] = "client_polygons"
    else:
        fp = _features_in_crs(site.get("building_footprints"), g.crs)
        blabels = rasterize_labels([f["geometry"] for f in fp], g) if fp else np.zeros(g.shape, np.int32)
        if fp:
            # the client's site = where the UAV flew; the footprint dataset
            # also covers the town and the neighbouring works
            u = site["uav"]
            surveyed = np.isfinite(u["dsm"]) | (np.isfinite(u["red"]) & (u["red"] > 0) if "red" in u else False)
            share = ndimage.mean(surveyed.astype(np.float32), blabels, np.arange(1, len(fp) + 1))
            keep = np.concatenate([[0], np.cumsum(share >= 0.5) * (share >= 0.5)]).astype(np.int32)
            blabels = keep[blabels]
            fp = [f for f, sh in zip(fp, share) if sh >= 0.5]
        bclass = {i + 1: "roof" for i in range(len(fp))}
        out["asset_source"] = ("building_footprints+uav_stacks_pipes" if fp else "uav_detected")
        # the UAV surface adds what footprint datasets miss: stacks and pipe bridges
        nxt = int(blabels.max())
        geo = {}
        for o in st["objects"]:
            m = st["labels"] == o["label"]
            if o["class"] == terrain.ROOF and fp:
                continue                               # roofs come from the footprints
            if fp and (blabels[m] > 0).mean() > 0.5:
                continue                               # already a footprint
            nxt += 1
            blabels = np.where(m & (blabels == 0), nxt, blabels)
            bclass[nxt] = o["class"]
            geo[nxt] = o
        out["asset_objects"] = [{"label": k, **v} for k, v in geo.items()]
    # heights for every asset from the UAV surface where it exists
    n_as = int(blabels.max())
    hmax_as = (ndimage.maximum(np.nan_to_num(ref["ndsm_raw"]), blabels, np.arange(1, n_as + 1))
               if n_as else [])
    area_as = (ndimage.sum(np.ones(g.shape, np.float32), blabels, np.arange(1, n_as + 1)) * g.pixel_area
               if n_as else [])
    if blabels.max() > 0:
        bres = thermal.analyse_scenes(scenes, g30, blabels, g, exclude | (blabels > 0), "Asset", cfg)
        comp = out.get("composite_lst")
        ds = out.get("downscaled_lst")
        for srow in bres["summary"]:
            k = srow["label"]
            srow["class"] = bclass.get(k, "building")
            srow["area_m2"] = float(area_as[k - 1])
            srow["max_height_m"] = float(hmax_as[k - 1]) if hmax_as[k - 1] > 0 else None
            if comp is not None:
                m30 = to_grid((blabels == k).astype(np.float32), g, g30) > 0
                srow["max_lst_composite_c"] = float(np.nanmax(comp[m30])) if m30.any() else None
            if ds is not None:
                m3 = to_grid((blabels == k).astype(np.float32), g, ds["grid"]) > 0.3
                srow["max_lst_downscaled_c"] = float(np.nanmax(ds["lst"][m3])) if m3.any() else None
            srow["alert"] = _alert_level(srow)
        vals = [b.get("max_lst_composite_c") for b in bres["summary"] if b.get("max_lst_composite_c") is not None]
        p90 = float(np.percentile(vals, 90)) if vals else None
        for b in bres["summary"]:
            b["top10pct_hottest"] = bool(p90 is not None and (b.get("max_lst_composite_c") or -1e9) >= p90)
        out["buildings"] = bres["summary"]
        out["building_p90_c"] = p90
        out["building_labels"] = blabels
        out["asset_per_scene"] = bres["per_scene"]
        from collections import Counter
        tel(f"  thermal assets ({out['asset_source']}): {dict(Counter(bclass.values()))}; "
            f"top-10% threshold {p90:.1f} °C" if p90 is not None else "  no asset temperatures")

    swir = site.get("swir") or []
    if swir:
        sw = thermal.swir_hotspots(swir, ref["labels"], g, site["swir_grid"], cfg)
        out["swir_n_scenes"] = sw["n_scenes"]
        out["swir_hot_frequency"] = sw["hot_frequency"]
        for s in out["piles"]:
            hits = sw["per_object"].get(s["label"], [])
            s["swir_hot_scenes"] = len(hits)
            s["swir_hot_dates"] = [h["date"] for h in hits]
        pool_swir_neighbours(out["piles"], ref["labels"], g, cfg.SWIR_POOL_M)
    for s in out["piles"]:
        s["alert"] = _alert_level(s)
    tel(f"  {sum(1 for s in out['piles'] if s['alert'] != 'none')} piles with a thermal alert")
    tel.stop("Thermal: per-pile contrast, persistence, hot-spot bound")
    return out


def pool_swir_neighbours(piles: List[Dict], labels: np.ndarray, g: Grid, dist_m: float) -> None:
    """A Sentinel-2 SWIR pixel (10–20 m) cannot tell adjacent piles apart: one
    smouldering spot can land on either neighbour from one date to the next.
    Piles within ``dist_m`` of a pile with hits share its dates
    (``swir_pool_dates``); the alert counts distinct pooled dates."""
    hit = [s for s in piles if s.get("swir_hot_dates")]
    if not hit:
        return
    it = max(int(round(dist_m / g.res)), 1)
    near = {}
    for s in hit:
        m = labels == s["label"]
        rr, cc = np.nonzero(m)
        r0, r1 = max(rr.min() - it, 0), min(rr.max() + it + 1, labels.shape[0])
        c0, c1 = max(cc.min() - it, 0), min(cc.max() + it + 1, labels.shape[1])
        d = ndimage.binary_dilation(m[r0:r1, c0:c1], iterations=it)
        near[s["label"]] = set(np.unique(labels[r0:r1, c0:c1][d]).tolist()) - {0}
    by_label = {s["label"]: s for s in piles}
    for s in piles:
        dates, via = set(s.get("swir_hot_dates") or []), []
        for h in hit:
            if h["label"] != s["label"] and s["label"] in near[h["label"]]:
                new = set(h["swir_hot_dates"]) - dates
                if new:
                    dates |= new
                    via.append(by_label[h["label"]]["pile_id"] if "pile_id" in by_label[h["label"]] else str(h["label"]))
        s["swir_pool_dates"] = sorted(dates)
        s["swir_pool_via"] = via


def _alert_level(s: Dict) -> str:
    """none / watch / warning / critical — conservative by design.

    Daytime warmth alone never escalates past "watch": a black coal pile in
    the sun is warmer than a grey yard without any self-heating. Night-time
    excess, persistent across passes, or SWIR combustion does.
    """
    s["alert_reason"] = []
    swir = max(s.get("swir_hot_scenes", 0), len(s.get("swir_pool_dates") or []))
    if swir >= 2:
        via = s.get("swir_pool_via") or []
        s["alert_reason"].append(f"SWIR combustion signal on {swir} dates"
                                 + (f" (with adjacent {', '.join(via)})" if via else ""))
        return "critical"
    night_hits, n_night = s.get("n_night_anomalous", 0), s.get("n_night_scenes", 0)
    from math import comb
    p0 = StockpileConfig.NOISE_ANOMALY_RATE
    p_night = sum(comb(n_night, j) * p0 ** j * (1 - p0) ** (n_night - j)
                  for j in range(night_hits, n_night + 1)) if n_night else 1.0
    s["p_value_night"] = p_night
    # warm at night more often than a 2σ test fires by chance
    if (night_hits >= 2 and p_night < StockpileConfig.NOISE_P_VALUE) or (night_hits >= 1 and swir >= 1):
        s["alert_reason"].append(f"warm at night in {night_hits}/{n_night} passes")
        return "warning"
    if s.get("persistent_anomaly") and s.get("significant_vs_noise"):
        s["alert_reason"].append("warm in most passes (day); check at night")
        return "warning"
    if swir == 1 or night_hits >= 1:
        s["alert_reason"].append("single SWIR date" if swir == 1 else
                                 f"warm in {night_hits}/{n_night} night passes (within chance)")
        return "watch"
    if s.get("significant_vs_noise"):
        s["alert_reason"].append(f"warm by day in {s.get('n_anomalous')}/{s.get('n_scenes')} passes — "
                                 "consistent with solar heating of dark material; needs night data")
        return "watch"
    return "none"


# ── orchestration ────────────────────────────────────────────────────────────

def run_pipeline(site: Dict, project_uuid: str, outdir: str,
                 stockpile_fc: Optional[dict] = None, building_fc: Optional[dict] = None,
                 sensors: Optional[List[str]] = None, depth: Optional[DepthAnything] = None,
                 cfg=StockpileConfig, log=print, upload=None, defer_outputs: bool = False) -> Dict:
    from . import outputs

    tel = Telemetry(project_uuid, log)
    sensors = sensors or list(cfg.SENSORS)
    stockpile_fc = stockpile_fc or site.get("stockpiles")
    building_fc = building_fc or site.get("buildings")
    tel(f"STOCKPILE INTELLIGENCE — {project_uuid} — sensors: {', '.join(sensors)}")

    ref = reference_inventory(site, stockpile_fc, cfg, tel)
    if depth is None and cfg.ENABLE_DEPTH:
        depth = DepthAnything(log=tel)
    study = sensor_study(site, ref, cfg, tel, sensors, depth)
    study["recent"] = recent_sentinel(site, ref, study, cfg, tel)
    study["recent_pleiades"] = recent_pleiades(site, ref, study, cfg, tel)
    therm = thermal_stage(site, ref, building_fc, cfg, tel)
    # checkpoint: the expensive part is done — outputs can be rebuilt from this
    import gc
    import os
    import pickle
    os.makedirs(outdir, exist_ok=True)
    try:
        with open(os.path.join(outdir, "results_checkpoint.pkl"), "wb") as fh:
            pickle.dump({"ref": ref, "study": study, "therm": therm}, fh, protocol=4)
    except Exception as exc:
        tel(f"  checkpoint not written: {exc}")
    gc.collect()
    if defer_outputs:
        # large sites: outputs are written by a fresh process from the
        # checkpoint (drawing them here, next to every model's arrays, ran a
        # 16 GB container out of memory twice)
        return {"deferred": True, "checkpoint": os.path.join(outdir, "results_checkpoint.pkl")}

    tel.start("Outputs")
    report = outputs.write_all(outdir, project_uuid, site, ref, study, therm, cfg, tel.timings)
    if upload:
        report["uploaded"] = upload(outdir, project_uuid)
    tel.stop("Outputs")
    report["timings_s"] = tel.timings
    return report
