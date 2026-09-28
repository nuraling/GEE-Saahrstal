"""Thermal monitoring of stockpiles and buildings.

What a thermal satellite can and cannot say about a pile:

* Landsat 8/9 band 10 is *measured* at 100 m and delivered resampled to 30 m;
  ECOSTRESS at ~70 m. A 40 × 20 m coal pile fills a fraction of one measured
  pixel, so its temperature is always mixed with the yard around it.
* A smouldering spot of a few square metres therefore shows up as a small
  excess over the background — the question is whether that excess is larger
  than the background's own scatter. Every anomaly here is reported against
  that scatter (a z-score), never as a bare temperature.
* The original "peak 5×5 m temperature" divided the excess by 25/900, treating
  band 10 as 30 m and radiance as T⁴. At 10.9 µm radiance follows Planck, not
  Stefan–Boltzmann, and the footprint is 100 m: the same formula with the
  right physics is kept, reported as the *hot-spot temperature that would
  explain the excess*, and only where the excess is significant.
* Surface combustion (> ~300 °C) is visible in shortwave infrared at 20 m
  (Sentinel-2 B11/B12), far sharper than any thermal band — used as a second,
  independent detector (NHI, Marchese et al. 2019).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from .config import StockpileConfig
from .grid import Grid, block_mean, resample

C1 = 1.191042e8      # W µm⁴ m⁻² sr⁻¹
C2 = 1.4387752e4     # µm K


def planck_radiance(t_k, wl_um: float = StockpileConfig.LANDSAT_BAND10_UM):
    t_k = np.asarray(t_k, dtype=np.float64)
    return C1 / (wl_um ** 5 * (np.exp(C2 / (wl_um * t_k)) - 1.0))


def inverse_planck(L, wl_um: float = StockpileConfig.LANDSAT_BAND10_UM):
    L = np.asarray(L, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return C2 / (wl_um * np.log(C1 / (wl_um ** 5 * L) + 1.0))


def hotspot_temperature(t_obs_c, t_bg_c, hotspot_area_m2: float, footprint_m: float,
                        wl_um: float = StockpileConfig.LANDSAT_BAND10_UM):
    """Temperature a hot-spot of hotspot_area_m2 must have, inside one sensor
    footprint of footprint_m × footprint_m at background t_bg_c, to raise the
    pixel to t_obs_c. Linear mixing of radiance, not of temperature."""
    # a hot-spot larger than the footprint fills it: the pixel IS the hot-spot
    phi = min(hotspot_area_m2 / (footprint_m ** 2), 1.0)
    L_obs = planck_radiance(np.asarray(t_obs_c) + 273.15, wl_um)
    L_bg = planck_radiance(np.asarray(t_bg_c) + 273.15, wl_um)
    L_hot = (L_obs - (1 - phi) * L_bg) / phi
    t = inverse_planck(np.where(L_hot > 0, L_hot, np.nan), wl_um) - 273.15
    return t


def min_detectable_hotspot(t_bg_c: float, sigma_c: float, k: float, hotspot_area_m2: float,
                           footprint_m: float, wl_um: float = StockpileConfig.LANDSAT_BAND10_UM) -> float:
    """Coolest hot-spot of that size a sensor with this footprint can flag."""
    return float(hotspot_temperature(t_bg_c + k * sigma_c, t_bg_c, hotspot_area_m2, footprint_m, wl_um))


def psf_coverage(mask_hr: np.ndarray, g_hr: Grid, g_th: Grid, native_m: float) -> np.ndarray:
    """Share of each thermal pixel's *measured* footprint covered by mask."""
    sigma = native_m / g_hr.res / 2.355          # FWHM ≈ native resolution
    sm = ndimage.gaussian_filter(mask_hr.astype(np.float32), sigma)
    return block_mean(sm, g_hr, g_th)


def _robust(vals: np.ndarray):
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan, np.nan
    med = float(np.median(vals))
    return med, float(1.4826 * np.median(np.abs(vals - med)))


def object_geometry(g_th: Grid, labels_hr: np.ndarray, g_hr: Grid, exclude_hr: np.ndarray,
                    native_m: float, cfg=StockpileConfig) -> List[Dict]:
    """Per object, once: its footprint coverage of each thermal pixel and its
    background ring. Independent of the scene, so computed a single time and
    reused for every pass (it was recomputed per scene: 48× the work)."""
    n = int(labels_hr.max())
    geo = []
    if n == 0:
        return geo
    excl_th = psf_coverage(exclude_hr, g_hr, g_th, native_m)
    ring_in = cfg.BG_RING_INNER_M / g_hr.res
    ring_out = cfg.BG_RING_OUTER_M / g_hr.res
    objs = ndimage.find_objects(labels_hr)
    kf = int(round(g_th.res / g_hr.res))
    for k in range(1, n + 1):
        sl = objs[k - 1]
        if sl is None:
            geo.append(None)
            continue
        pad = int(ring_out) + 2
        r0 = max(sl[0].start - pad, 0) // kf * kf
        c0 = max(sl[1].start - pad, 0) // kf * kf
        r1 = min(-(-(sl[0].stop + pad) // kf) * kf, labels_hr.shape[0])
        c1 = min(-(-(sl[1].stop + pad) // kf) * kf, labels_hr.shape[1])
        g_sub = _sub(g_hr, r0, c0, r1, c1)
        g_sub_th = Grid(g_hr.x0 + c0 * g_hr.res, g_hr.y1 - r0 * g_hr.res, g_th.res,
                        -(-(c1 - c0) // kf), -(-(r1 - r0) // kf), g_th.crs)
        tr0, tc0 = r0 // kf, c0 // kf
        m = labels_hr[r0:r1, c0:c1] == k
        cov = psf_coverage(m, g_sub, g_sub_th, native_m)
        dist = ndimage.distance_transform_edt(~m)
        ring = (dist <= ring_out) & (dist > ring_in)
        ring_th = block_mean(ring.astype(np.float32), g_sub, g_sub_th)
        hh = min(cov.shape[0], excl_th.shape[0] - tr0)
        ww = min(cov.shape[1], excl_th.shape[1] - tc0)
        ex = excl_th[tr0:tr0 + hh, tc0:tc0 + ww]
        cov, ring_th = cov[:hh, :ww], ring_th[:hh, :ww]
        geo.append({"win": (tr0, tc0, hh, ww), "cov": cov,
                    "bg": (ring_th > 0.5) & (ex < 0.2) & (cov < 0.05)})
    return geo


def object_contrast(lst: np.ndarray, g_th: Grid, labels_hr: np.ndarray, g_hr: Grid,
                    exclude_hr: np.ndarray, native_m: float, cfg=StockpileConfig,
                    geometry: Optional[List[Dict]] = None) -> List[Dict]:
    """Per object: its hottest well-covered thermal pixel vs a background ring."""
    geometry = geometry if geometry is not None else object_geometry(
        g_th, labels_hr, g_hr, exclude_hr, native_m, cfg)
    rows = []
    for k, gm in enumerate(geometry, start=1):
        if gm is None:
            continue
        tr0, tc0, hh, ww = gm["win"]
        lst_sub = lst[tr0:tr0 + hh, tc0:tc0 + ww]
        cov = gm["cov"][:lst_sub.shape[0], :lst_sub.shape[1]]
        bgm = gm["bg"][:lst_sub.shape[0], :lst_sub.shape[1]]
        bg = lst_sub[bgm]
        t_bg, s_bg = _robust(bg)
        inside = np.isfinite(lst_sub) & (cov > 0)
        if not inside.any() or not np.isfinite(t_bg):
            rows.append({"label": k, "valid": False})
            continue
        # pixel most dominated by the object; ties → hotter
        score = np.where(inside, cov + 1e-3 * np.nan_to_num(lst_sub), -np.inf)
        i = np.unravel_index(np.argmax(score), score.shape)
        t_obj, f = float(lst_sub[i]), float(cov[i])
        dt = t_obj - t_bg
        z = dt / s_bg if s_bg and s_bg > 0 else np.nan
        rows.append({"label": k, "valid": True, "t_object_c": t_obj, "t_background_c": t_bg,
                     "sigma_background_c": s_bg, "delta_t_c": dt, "z": float(z),
                     # excess the pile surface would need if it were uniformly warm
                     "delta_t_surface_equiv_c": dt / f if f > 0.05 else None,
                     "coverage": f, "n_background_px": int(np.isfinite(bg).sum())})
    return rows


def _sub(g: Grid, r0, c0, r1, c1) -> Grid:
    return Grid(g.x0 + c0 * g.res, g.y1 - r0 * g.res, g.res, c1 - c0, r1 - r0, g.crs)


def analyse_scenes(scenes: List[Dict], g_th: Grid, labels_hr: np.ndarray, g_hr: Grid,
                   exclude_hr: np.ndarray, id_prefix: str = "Pile",
                   cfg=StockpileConfig) -> Dict:
    """Per-scene contrasts → per-object persistence, trend and hot-spot bound.

    scenes: [{"lst": array on g_th, "date": "YYYY-MM-DD", "sensor": str,
              "native_m": float, "night": bool}]
    """
    per_scene = []
    geo_cache = {}
    for sc in scenes:
        if sc["native_m"] not in geo_cache:
            geo_cache[sc["native_m"]] = object_geometry(g_th, labels_hr, g_hr, exclude_hr, sc["native_m"], cfg)
        for row in object_contrast(sc["lst"], g_th, labels_hr, g_hr, exclude_hr, sc["native_m"], cfg,
                                   geometry=geo_cache[sc["native_m"]]):
            if not row.get("valid"):
                continue
            anomalous = (row["delta_t_c"] > cfg.ANOMALY_MIN_DT_C and
                         np.isfinite(row["z"]) and row["z"] > cfg.ANOMALY_K_SIGMA)
            row.update({"date": sc["date"], "sensor": sc["sensor"], "night": bool(sc.get("night")),
                        "anomalous": bool(anomalous), "native_m": sc["native_m"]})
            if anomalous:
                row["hotspot_equiv_c"] = float(hotspot_temperature(
                    row["t_object_c"], row["t_background_c"],
                    cfg.HOTSPOT_SIDE_M ** 2, sc["native_m"]))
            per_scene.append(row)

    summary = []
    n = int(labels_hr.max())
    for k in range(1, n + 1):
        rows = [r for r in per_scene if r["label"] == k]
        if not rows:
            summary.append({"label": k, "object_id": f"{id_prefix}_{k}", "n_scenes": 0})
            continue
        dts = np.array([r["delta_t_c"] for r in rows])
        zs = np.array([r["z"] for r in rows])
        anom = np.array([r["anomalous"] for r in rows])
        hs = [r["hotspot_equiv_c"] for r in rows if "hotspot_equiv_c" in r and np.isfinite(r["hotspot_equiv_c"])]
        trend = None
        if len(rows) >= 4:
            import datetime as _dt
            t = np.array([(_dt.date.fromisoformat(r["date"]) - _dt.date(2000, 1, 1)).days for r in rows], float)
            if np.ptp(t) > 0:
                trend = float(np.polyfit(t / 30.44, dts, 1)[0])
        persist = float(anom.mean())
        night = np.array([r["night"] for r in rows])
        from math import comb
        n_, a_ = len(rows), int(anom.sum())
        p0 = cfg.NOISE_ANOMALY_RATE
        p_noise = float(sum(comb(n_, j) * p0 ** j * (1 - p0) ** (n_ - j) for j in range(a_, n_ + 1)))
        sig = np.array([r["sigma_background_c"] for r in rows])
        bg = np.array([r["t_background_c"] for r in rows])
        native = rows[0]["native_m"]
        summary.append({
            "label": k, "object_id": f"{id_prefix}_{k}", "n_scenes": len(rows),
            "mean_delta_t_c": float(np.nanmean(dts)), "max_delta_t_c": float(np.nanmax(dts)),
            "surface_delta_t_equiv_c_median": (float(np.median([r["delta_t_surface_equiv_c"] for r in rows
                                                                  if r.get("delta_t_surface_equiv_c") is not None]))
                                               if any(r.get("delta_t_surface_equiv_c") is not None for r in rows)
                                               else None),
            "max_z": float(np.nanmax(zs)) if np.isfinite(zs).any() else None,
            "persistence": persist, "n_anomalous": int(anom.sum()),
            "p_value_vs_noise": p_noise,
            "significant_vs_noise": bool(p_noise < cfg.NOISE_P_VALUE),
            "n_night_scenes": int(night.sum()),
            "n_night_anomalous": int((anom & night).sum()),
            "day_persistence": float(anom[~night].mean()) if (~night).any() else None,
            "persistent_anomaly": bool(persist >= cfg.PERSISTENCE_FLAG and anom.sum() >= 2),
            "delta_t_trend_c_per_month": trend,
            "hotspot_equiv_5m_c_median": float(np.median(hs)) if hs else None,
            "min_detectable_5m_hotspot_c": min_detectable_hotspot(
                float(np.nanmedian(bg)), float(np.nanmedian(sig)), cfg.ANOMALY_K_SIGMA,
                cfg.HOTSPOT_SIDE_M ** 2, native),
            "mean_coverage": float(np.mean([r["coverage"] for r in rows])),
        })
    return {"per_scene": per_scene, "summary": summary}


def swir_hotspots(scenes: List[Dict], labels_hr: np.ndarray, g_hr: Grid, g_sw: Grid,
                  cfg=StockpileConfig) -> Dict:
    """Sentinel-2 SWIR high-temperature detection per object.

    scenes: [{"b11": arr, "b12": arr, "date": str}] on g_sw (reflectance).
    NHI_SWIR = (B12 − B11) / (B12 + B11); ordinary surfaces are ≤ 0 because
    their reflectance falls from 1.6 to 2.2 µm. Emission from burning material
    adds more at 2.2 µm and pushes it positive.
    """
    lab_sw = np.rint(resample(labels_hr.astype(np.float32), g_hr, g_sw, order=0)).astype(np.int32)
    n = int(labels_hr.max())
    hits = {k: [] for k in range(1, n + 1)}
    maps = []
    if not scenes:
        return {"per_object": hits, "hot_frequency": None, "n_scenes": 0}
    nhis, b12s = [], []
    for sc in scenes:
        b11, b12 = sc["b11"].astype(np.float32), sc["b12"].astype(np.float32)
        nhis.append((b12 - b11) / (b12 + b11 + 1e-6))
        b12s.append(b12)
    # each pixel's own normal: dark coal and scrap sit at NHI ≥ 0 in every
    # scene; burning is a departure from that, not the sign of the index
    nhi_med = np.nanmedian(np.stack(nhis), axis=0)
    b12_med = np.nanmedian(np.stack(b12s), axis=0)
    for sc, nhi, b12 in zip(scenes, nhis, b12s):
        hot = ((nhi > cfg.NHI_SWIR_THRESHOLD) & (nhi - nhi_med > cfg.NHI_ANOMALY)
               & (b12 > cfg.B12_ANOMALY_RATIO * b12_med) & (b12 > 0.02) & np.isfinite(nhi))
        maps.append(hot)
        for k in range(1, n + 1):
            m = lab_sw == k
            if m.any():
                c = int((hot & m).sum())
                # overlapping S2 tiles deliver the same date twice: count it once
                if c and sc["date"] not in {h["date"] for h in hits[k]}:
                    hits[k].append({"date": sc["date"], "hot_px": c,
                                    "max_nhi": float(np.nanmax(np.where(m, nhi, np.nan)))})
    freq = np.mean(maps, axis=0).astype(np.float32) if maps else None
    return {"per_object": hits, "hot_frequency": freq, "n_scenes": len(scenes)}


def downscale_lst(lst_th: np.ndarray, g_th: Grid, bands_hr: Dict[str, np.ndarray],
                  g_hr: Grid, cfg=StockpileConfig) -> Dict:
    """Modelled fine-scale LST: learn LST ~ optical at the thermal scale,
    predict at the fine scale, then add back the coarse residual so each
    thermal pixel's mean is preserved (the same correction Solain uses).

    A map for context, not a measurement: fine-scale variation comes from the
    optical bands, not from the thermal sensor.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from .features import build_features
    X_hr, names = build_features(bands_hr, g_hr.res)
    F = X_hr.shape[2]
    X_th = np.dstack([block_mean(X_hr[..., i], g_hr, g_th) for i in range(F)])
    h, w = min(X_th.shape[0], lst_th.shape[0]), min(X_th.shape[1], lst_th.shape[1])
    X_th, y = X_th[:h, :w].reshape(-1, F), lst_th[:h, :w].reshape(-1)
    ok = np.isfinite(y) & np.all(np.isfinite(X_th), axis=1)
    if ok.sum() < 50:
        up = resample(lst_th, g_th, g_hr, order=1)
        return {"lst": up, "r2_coarse": None, "note": "too few thermal pixels; interpolated"}
    m = HistGradientBoostingRegressor(max_iter=200, random_state=cfg.RAND_SEED)
    m.fit(X_th[ok], y[ok])
    r2 = float(m.score(X_th[ok], y[ok]))
    pred = m.predict(X_hr.reshape(-1, F)).reshape(X_hr.shape[:2]).astype(np.float32)
    resid = lst_th - block_mean(pred, g_hr, g_th)[:lst_th.shape[0], :lst_th.shape[1]]
    out = pred + resample(np.nan_to_num(resid), g_th, g_hr, order=1)
    return {"lst": np.clip(out, cfg.LST_CLAMP_MIN, cfg.LST_CLAMP_MAX), "r2_coarse": r2,
            "note": "modelled from optical predictors; coarse means preserved"}
