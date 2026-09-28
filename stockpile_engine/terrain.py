"""Stockpiles from the UAV surfaces: nDSM, pile extraction, volume, tonnage.

This is the reference ("ground truth") side of the product. Every satellite
estimate is scored against what is computed here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from .config import StockpileConfig
from .grid import Grid

PILE, BUILDING, VEGETATION, SMALL = "stockpile", "structure", "vegetation", "too_small"


def ndsm(dsm: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    """Height above ground. NaN where either surface is missing."""
    h = dsm.astype(np.float32) - dtm.astype(np.float32)
    h[~np.isfinite(h)] = np.nan
    return h


def slope_deg(surface: np.ndarray, res: float) -> np.ndarray:
    z = np.where(np.isfinite(surface), surface, np.nanmedian(surface))
    gy, gx = np.gradient(z, res)
    s = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    s[~np.isfinite(surface)] = np.nan
    return s


def label_stats(values: np.ndarray, labels: np.ndarray, n: int, func) -> np.ndarray:
    """ndimage label reduction for labels 1..n, NaN-safe (NaN pixels ignored)."""
    idx = np.arange(1, n + 1)
    finite = np.isfinite(values)
    lab = np.where(finite, labels, 0)
    v = np.where(finite, values, 0.0)
    return np.asarray(func(v, lab, idx), dtype=np.float64)


def detect_piles(height: np.ndarray, surface: np.ndarray, grid: Grid,
                 ndvi: Optional[np.ndarray] = None, cfg=StockpileConfig) -> Dict:
    """Connected objects above ground, each classed pile / structure / vegetation.

    The original workflow kept objects whose *mean* height was ≤ 3 m. On this
    site that throws away the coal and ore piles (5–15 m) and keeps parked
    wagons. Objects are instead separated by form: bulk material rests at its
    angle of repose, buildings and wagons have vertical walls and flat tops.
    """
    res = grid.res
    cand = np.isfinite(height) & (height > cfg.MIN_PILE_HEIGHT_M) & (height < cfg.MAX_PILE_HEIGHT_M)
    cand = ndimage.binary_opening(cand, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndimage.label(cand, structure=np.ones((3, 3)))
    if n == 0:
        return {"labels": labels, "classes": {}, "objects": []}

    slope = slope_deg(surface, res)
    area = label_stats(np.ones_like(height), labels, n, ndimage.sum) * grid.pixel_area
    hmax = label_stats(height, labels, n, ndimage.maximum)
    hmean = label_stats(height, labels, n, ndimage.mean)
    wall = label_stats((slope > cfg.WALL_SLOPE_DEG).astype(np.float32), labels, n, ndimage.mean)
    # "top" = upper 30% of the object's own height range
    top = np.zeros_like(height, dtype=bool)
    lab_hmax = np.concatenate([[np.inf], hmax])[labels]
    top = (labels > 0) & (height >= 0.7 * lab_hmax)
    flat_lab = np.where(top, labels, 0)
    flat_top = label_stats((slope < cfg.FLAT_SLOPE_DEG).astype(np.float32), flat_lab, n, ndimage.mean)
    veg = (label_stats(ndvi, labels, n, ndimage.mean) if ndvi is not None
           else np.zeros(n))

    classes, objects = {}, []
    for k in range(1, n + 1):
        i = k - 1
        if area[i] < cfg.MIN_PILE_AREA_M2:
            c = SMALL
        elif veg[i] > cfg.MAX_PILE_NDVI:
            c = VEGETATION
        elif (wall[i] > cfg.MAX_WALL_FRACTION
              or (flat_top[i] > cfg.MAX_FLAT_TOP_FRAC and hmax[i] > 2.5)
              or (hmax[i] > 0 and hmean[i] / hmax[i] > cfg.MAX_FILL_RATIO)):
            c = BUILDING
        else:
            c = PILE
        classes[k] = c
        objects.append({"label": k, "class": c, "area_m2": float(area[i]),
                        "max_height_m": float(hmax[i]), "mean_height_m": float(hmean[i]),
                        "wall_fraction": float(wall[i]), "flat_top_fraction": float(flat_top[i]),
                        "fill_ratio": float(hmean[i] / hmax[i]) if hmax[i] > 0 else None,
                        "ndvi_mean": float(veg[i])})
    return {"labels": labels, "classes": classes, "objects": objects}


def pile_labels_only(det: Dict) -> np.ndarray:
    """Relabel so only objects classed as piles remain, numbered 1..m."""
    labels = det["labels"]
    keep = [k for k, c in det["classes"].items() if c == PILE]
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    for new, old in enumerate(keep, start=1):
        lut[old] = new
    return lut[labels]


def remove_thin_structures(height: np.ndarray, res: float,
                           radius_m: float = StockpileConfig.BOOM_REMOVAL_RADIUS_M) -> np.ndarray:
    """Grey opening with a flat disc: removes features narrower than the disc
    (conveyor booms, stacker arms, lamp posts) and keeps the piles under them."""
    r = max(int(round(radius_m / res)), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    finite = np.isfinite(height)
    h = np.where(finite, height, 0.0)
    opened = ndimage.grey_opening(h, footprint=disc)
    out = np.minimum(h, opened).astype(np.float32)
    out[~finite] = np.nan
    return out


def toe_plane(surface: np.ndarray, mask: np.ndarray, grid: Grid) -> Optional[np.ndarray]:
    """Plane through the surface around the pile's toe — the pad it sits on.

    Robust least squares: points more than 2σ off the first fit (a truck, a
    neighbouring pile) are dropped and the plane refitted.
    """
    ring = ndimage.binary_dilation(mask, iterations=2) & ~mask & np.isfinite(surface)
    if ring.sum() < 6:
        return None
    X, Y = grid.xy()
    xs, ys, zs = X[ring], Y[ring], surface[ring]
    xm, ym = xs.mean(), ys.mean()
    A = np.c_[xs - xm, ys - ym, np.ones_like(xs)]
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    r = zs - A @ coef
    keep = np.abs(r) <= 2 * max(r.std(), 1e-6)
    if keep.sum() >= 6:
        coef, *_ = np.linalg.lstsq(A[keep], zs[keep], rcond=None)
    return (coef[0] * (X - xm) + coef[1] * (Y - ym) + coef[2]).astype(np.float32)


def pile_volumes(height: np.ndarray, labels: np.ndarray, grid: Grid,
                 surface: Optional[np.ndarray] = None, base: str = "dtm",
                 sigma_z: float = StockpileConfig.SIGMA_Z_UAV_M) -> List[Dict]:
    """Volume of every labelled pile: Σ max(h, 0) · pixel area.

    ``height`` is height above the chosen base (for base="dtm" the nDSM).
    With base="toe" the base is re-fitted per pile from ``surface``.
    The ± band is the systematic term A·σz — for a UAV survey it is the
    dominant error; for a satellite height model σz is the model RMSE.
    """
    n = int(labels.max())
    out = []
    if n == 0:
        return out
    px = grid.pixel_area
    idx = np.arange(1, n + 1)
    h = np.where(np.isfinite(height), np.clip(height, 0, None), 0.0)
    cnt = ndimage.sum(np.ones_like(h), labels, idx)
    vol = ndimage.sum(h, labels, idx) * px
    hmax = ndimage.maximum(h, labels, idx)
    objs = ndimage.find_objects(labels)
    for k in idx:
        i = k - 1
        v, hm = float(vol[i]), float(hmax[i])
        if base == "toe" and surface is not None and objs[i] is not None:
            sl = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in objs[i])
            sub_mask = labels[sl] == k
            sub_grid = Grid(grid.x0 + sl[1].start * grid.res, grid.y1 - sl[0].start * grid.res,
                            grid.res, sub_mask.shape[1], sub_mask.shape[0], grid.crs)
            plane = toe_plane(surface[sl], sub_mask, sub_grid)
            if plane is not None:
                hh = np.clip(surface[sl] - plane, 0, None)
                hh = np.where(np.isfinite(hh), hh, 0.0)
                v = float(hh[sub_mask].sum() * px)
                hm = float(hh[sub_mask].max()) if sub_mask.any() else 0.0
        area = float(cnt[i] * px)
        out.append({"label": int(k), "area_m2": area, "volume_m3": v,
                    "mean_height_m": v / area if area else 0.0, "max_height_m": hm,
                    "volume_sigma_m3": area * sigma_z, "base": base})
    return out


# ── commodity & tonnage ──────────────────────────────────────────────────────

def pseudo_reflectance(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Scale all bands by ONE factor — the scene's 98th-percentile visible
    brightness — so the rules work on DN, TOA or surface reflectance alike.
    One common factor keeps the ratios between bands (the colour); scaling
    each band separately turned orange wood chips grey.
    Scene-relative, not physical."""
    vis = [bands[k] for k in ("red", "green", "blue") if k in bands]
    ref = np.nanmean(np.stack(vis), axis=0) if vis else next(iter(bands.values()))
    p = np.nanpercentile(ref, 98) if np.isfinite(ref).any() else 1.0
    p = p if p > 0 else 1.0
    return {k: np.clip(a / p, 0, 1.5).astype(np.float32) for k, a in bands.items()}


def classify_commodity(rgbn: Dict[str, float]) -> str:
    """Indicative material from a pile's median colour (pseudo-reflectance).

    Deliberately coarse: colour separates coal from ore from wood chips from
    aggregates, it cannot tell coal from coke or one ore grade from another.
    Clients override with a 'commodity' property on their pile polygons.
    """
    r, g, b = rgbn["red"], rgbn["green"], rgbn["blue"]
    nir = rgbn.get("nir", np.nan)
    bright = (r + g + b) / 3.0
    redness = (r - b) / (r + b + 1e-6)
    if np.isfinite(nir) and (nir - r) / (nir + r + 1e-6) > StockpileConfig.MAX_PILE_NDVI:
        return "unknown"
    # brightness is relative to the scene's bright end (1.0 ≈ its 98th pct)
    if bright < 0.22 and redness < 0.20:
        return "coal"
    if redness > 0.30 and bright >= 0.60:
        return "wood_chips"
    if redness > 0.18 and bright < 0.60:
        return "iron_ore"
    if bright > 1.20 and redness < 0.15:
        return "limestone"
    if bright >= 0.60 and redness <= 0.18:
        return "sand_gravel"
    if redness > 0.10:
        return "scrap_metal"
    return "unknown"


def commodity_from_examples(colours: Dict[int, Dict[str, float]],
                            labelled: Dict[int, str]) -> Dict[int, str]:
    """Nearest-centroid commodity for unlabelled piles, learned from the piles
    the operator did label. Any labels at all beat the colour rules: they are
    calibrated to this site's light, sensor and materials."""
    keys = ["blue", "green", "red", "nir"]
    keys = [k for k in keys if all(k in c for c in colours.values())]
    cents = {}
    for lab, com in labelled.items():
        if lab in colours:
            cents.setdefault(com, []).append([colours[lab][k] for k in keys])
    if not cents:
        return {}
    cents = {c: np.nanmean(np.array(v), axis=0) for c, v in cents.items()}
    out = {}
    for lab, col in colours.items():
        if lab in labelled:
            continue
        x = np.array([col[k] for k in keys])
        if not np.all(np.isfinite(x)):
            continue
        out[lab] = min(cents, key=lambda c: float(np.sum((cents[c] - x) ** 2)))
    return out


def tonnage(volume_m3: float, commodity: str) -> float:
    rho = StockpileConfig.BULK_DENSITY_T_M3.get(commodity, StockpileConfig.BULK_DENSITY_T_M3["unknown"])
    return float(volume_m3 * rho)


# ── ground where there is no DTM (the port, DSM_3) ────────────────────────────

def derive_dtm(dsm: np.ndarray, res: float, window_m: float = 80.0, coarse_m: float = 4.0) -> np.ndarray:
    """Bare-ground surface from a DSM alone (morphological ground filter).

    Block minimum to ``coarse_m``, grey opening with a disc of ``window_m``
    (anything narrower than that — piles, sheds, wagons — is cut away), a
    light smoothing, back to the fine grid, never above the DSM. Validated
    against the UAV DTM where both exist (see reference_inventory).
    """
    k = max(int(round(coarse_m / res)), 1)
    H, W = dsm.shape
    Hc, Wc = -(-H // k), -(-W // k)
    pad = np.full((Hc * k, Wc * k), np.nan, np.float32)
    pad[:H, :W] = dsm
    with np.errstate(all="ignore"):
        coarse = np.nanmin(pad.reshape(Hc, k, Wc, k), axis=(1, 3))
    valid = np.isfinite(coarse)
    if not valid.any():
        return np.full_like(dsm, np.nan)
    # fill gaps with a high value so they never become "ground"
    filled = np.where(valid, coarse, np.nanmax(coarse))
    r = max(int(round(window_m / 2 / coarse_m)), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    ground = ndimage.grey_opening(filled, footprint=disc)
    ground = ndimage.gaussian_filter(ground, 1.0)
    ground[~valid] = np.nan
    from scipy.ndimage import zoom
    fine = zoom(np.where(np.isfinite(ground), ground, np.nanmedian(ground)), k, order=1)[:H, :W]
    fine = np.minimum(fine, np.where(np.isfinite(dsm), dsm, np.inf)).astype(np.float32)
    fine[~np.isfinite(dsm)] = np.nan
    return fine


# ── industrial structures for thermal: roofs, chimneys, pipes/conveyors ──────

ROOF, CHIMNEY, PIPE = "roof", "chimney_stack", "pipe_conveyor"


def detect_structures(height: np.ndarray, grid: Grid, ndvi: Optional[np.ndarray] = None,
                      min_h: float = 2.5, min_area_m2: float = 15.0) -> Dict:
    """Buildings, chimneys/stacks and elevated pipes/conveyors from the nDSM.

    No upper height cap (stacks exceed 40 m) and no morphological opening
    (a 2 m pipe bridge is exactly what an opening removes). Pile-shaped
    objects (low fill ratio, no walls) and trees are left out.
    """
    cand = np.isfinite(height) & (height > min_h)
    labels, n = ndimage.label(cand, structure=np.ones((3, 3)))
    out = {"labels": np.zeros_like(labels), "objects": []}
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    area = ndimage.sum(np.ones_like(height), labels, idx) * grid.pixel_area
    hmax = ndimage.maximum(np.nan_to_num(height), labels, idx)
    hmean = ndimage.mean(np.nan_to_num(height), labels, idx)
    slope = slope_deg(height, grid.res)
    wall = ndimage.mean((slope > StockpileConfig.WALL_SLOPE_DEG).astype(np.float32), labels, idx)
    veg = ndimage.mean(np.nan_to_num(ndvi), labels, idx) if ndvi is not None else np.zeros(n)
    objs = ndimage.find_objects(labels)
    keep = np.zeros(n + 1, np.int32)
    j = 0
    for i in range(n):
        if area[i] < min_area_m2 or veg[i] > StockpileConfig.MAX_PILE_NDVI:
            continue
        fill = hmean[i] / hmax[i] if hmax[i] > 0 else 0
        if fill < StockpileConfig.MAX_FILL_RATIO and wall[i] < StockpileConfig.MAX_WALL_FRACTION:
            continue                     # pile-shaped: not a structure
        rr, cc = np.nonzero(labels[objs[i]] == i + 1)
        if rr.size >= 3:
            cov = np.cov(np.vstack([rr, cc]).astype(float))
            ev = np.sort(np.linalg.eigvalsh(cov))
            elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        if area[i] <= 600 and hmax[i] >= 25 and elong < 3:
            cls = CHIMNEY
        elif elong >= 5 and area[i] < 8000:
            cls = PIPE
        else:
            cls = ROOF
        j += 1
        keep[i + 1] = j
        out["objects"].append({"label": j, "class": cls, "area_m2": float(area[i]),
                               "max_height_m": float(hmax[i]), "mean_height_m": float(hmean[i]),
                               "elongation": elong})
    out["labels"] = keep[labels]
    return out


def flat_ground(dsm: np.ndarray, block: np.ndarray, water: Optional[np.ndarray] = None,
                percentile: float = 5.0) -> float:
    """One ground level for a flat block without a DTM: a low percentile of
    its land surface (the absolute minimum is noise or open water)."""
    m = block & np.isfinite(dsm)
    if water is not None:
        m &= ~water
    return float(np.nanpercentile(dsm[m], percentile)) if m.any() else float("nan")


def material_class(dsm: np.ndarray, labels: np.ndarray, k: int, s2_bands: Dict[str, np.ndarray],
                   labels10: np.ndarray, res: float = 1.0) -> Dict:
    """Scrap metal vs bulk material for pile k.

    Scrap heaps are jagged (surface roughness of the UAV DSM, detrended over
    3 m) and patchy in brightness (shiny steel, rust, paint side by side);
    bulk material rests smooth and uniform. Both must agree. Calibrated on
    the Saarlouis port (piles 41–43 = the scrap yard on the south quay).
    """
    objs = ndimage.find_objects(labels)
    sl = objs[k - 1] if k - 1 < len(objs) else None
    if sl is None:
        return {"material_class": "unknown"}
    sl = tuple(slice(max(x.start - 3, 0), x.stop + 3) for x in sl)
    z = dsm[sl]
    zz = np.where(np.isfinite(z), z, np.nanmedian(z))
    resid = zz - ndimage.gaussian_filter(zz, 3.0 / res)
    m = (labels[sl] == k) & np.isfinite(z)
    rough = float(np.std(resid[m])) if m.any() else float("nan")
    m10 = labels10 == k
    bright = (s2_bands["red"] + s2_bands["green"] + s2_bands["blue"]) / 3.0
    patchy = float(np.nanstd(bright[m10])) if m10.any() else float("nan")
    is_scrap = (rough >= StockpileConfig.SCRAP_MIN_ROUGHNESS_M and patchy >= StockpileConfig.SCRAP_MIN_PATCHINESS)
    return {"material_class": "scrap_metal" if is_scrap else "bulk",
            "surface_roughness_m": rough, "brightness_patchiness": patchy}
