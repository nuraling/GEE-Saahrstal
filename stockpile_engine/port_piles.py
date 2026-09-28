"""Pile outlines for the port (DSM_3) from the UAV surface, not hand-drawn.

The client polygons for the port missed most piles and two of them (41, 42)
are building roofs. The generic detector fails here: with no UAV DTM the
ground is one flat level, so the whole quay deck (1–2 m above it) and
everything touching it becomes one object.

Here:
  height   = DSM − flat ground level (lowest land level, water excluded)
  objects  = connected pixels higher than the deck (DECK_M), opened to cut
             thin bridges between piles and wagons
  class    = vegetation  (Pléiades NDVI)
             linear      (thin: wagons, conveyors, walls; max half-width < MIN_HALF_WIDTH_M)
             building    (vertical walls or a flat top)
             pile        otherwise
  material = from the 2025 orthophoto and the surface: dark → bulk (coal/ore),
             rough + grey/metallic → scrap, light → sand / wood chips
Every threshold is a module constant and is written into the output.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage

from .grid import Grid
from .terrain import slope_deg

DECK_M = 2.0              # quay deck + kerbs stay below this
MIN_AREA_M2 = 40.0
MIN_HALF_WIDTH_M = 3.0    # wagons ~3 m wide → half-width 1.5 m
WALL_SLOPE_DEG = 65.0
MAX_WALL_FRACTION = 0.12
FLAT_SLOPE_DEG = 8.0
MAX_FLAT_TOP = 0.55
MAX_NDVI = 0.30
DARK_MAX = 0.28           # mean ortho brightness (0–1) of dark bulk
LIGHT_MIN = 0.55
SCRAP_ROUGH_M = 0.6       # std of height about a 5 m smoothed surface


def detect(dsm: np.ndarray, ground, grid: Grid, ortho: Optional[Dict[str, np.ndarray]] = None,
           ndvi: Optional[np.ndarray] = None, deck_m: float = DECK_M) -> Dict:
    """``ground`` is a level (port: flat) or a surface array (blocks with a DTM)."""
    res = grid.res
    h = dsm - ground
    cand = np.isfinite(h) & (h > deck_m)
    cand = ndimage.binary_opening(cand, structure=np.ones((3, 3)), iterations=2)
    labels, n = ndimage.label(cand, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    if n == 0:
        return {"labels": labels, "objects": [], "params": _params(ground, deck_m)}

    slope = slope_deg(np.where(np.isfinite(dsm), dsm, ground), res)
    hz = np.nan_to_num(h)
    area = ndimage.sum(np.ones_like(hz), labels, idx) * res * res
    hmax = ndimage.maximum(hz, labels, idx)
    hmean = ndimage.mean(hz, labels, idx)
    half_w = ndimage.maximum(ndimage.distance_transform_edt(cand) * res, labels, idx)
    wall = ndimage.mean((slope > WALL_SLOPE_DEG).astype(np.float32), labels, idx)
    top = (labels > 0) & (hz >= 0.7 * np.concatenate([[np.inf], hmax])[labels])
    flat = ndimage.mean((slope < FLAT_SLOPE_DEG).astype(np.float32), np.where(top, labels, 0), idx)
    smooth = ndimage.uniform_filter(hz, size=max(int(round(5 / res)), 3))
    rough = np.sqrt(ndimage.mean((hz - smooth) ** 2, labels, idx))

    def lab_mean(a):
        if a is None:
            return np.full(n, np.nan)
        ok = np.isfinite(a)
        s = ndimage.sum(np.where(ok, a, 0), labels, idx)
        c = ndimage.sum(ok.astype(np.float32), labels, idx)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    veg = lab_mean(ndvi)
    bright = sat = None
    if ortho:
        rgb = np.dstack([ortho["red"], ortho["green"], ortho["blue"]]).astype(np.float32)
        valid = np.all(np.isfinite(rgb), axis=2) & (rgb.sum(axis=2) > 0)
        scale = np.nanpercentile(rgb[valid], 99.5) if valid.any() else 1.0
        rgb = np.where(valid[..., None], np.clip(rgb / scale, 0, 1), np.nan)
        bright = lab_mean(rgb.mean(axis=2))
        sat = lab_mean(rgb.max(axis=2) - rgb.min(axis=2))

    objects = []
    for i, k in enumerate(idx):
        if area[i] < MIN_AREA_M2:
            continue
        if np.isfinite(veg[i]) and veg[i] > MAX_NDVI:
            c = "vegetation"
        elif half_w[i] < MIN_HALF_WIDTH_M:
            c = "linear"
        elif wall[i] > MAX_WALL_FRACTION or (flat[i] > MAX_FLAT_TOP and hmax[i] > 2.5):
            c = "building"
        else:
            c = "pile"
        mat = None
        if c == "pile":
            b = bright[i] if bright is not None else np.nan
            if rough[i] >= SCRAP_ROUGH_M and not (np.isfinite(b) and b < DARK_MAX):
                mat = "scrap_metal"
            elif np.isfinite(b) and b < DARK_MAX:
                mat = "dark_bulk"
            elif np.isfinite(b) and b >= LIGHT_MIN:
                mat = "light_bulk"
            else:
                mat = "bulk"
        objects.append({"label": int(k), "class": c, "material": mat, "area_m2": float(area[i]),
                        "max_height_m": float(hmax[i]), "mean_height_m": float(hmean[i]),
                        "half_width_m": float(half_w[i]), "wall_fraction": float(wall[i]),
                        "flat_top_fraction": float(flat[i]), "roughness_m": float(rough[i]),
                        "ndvi": None if not np.isfinite(veg[i]) else float(veg[i]),
                        "brightness": None if bright is None or not np.isfinite(bright[i]) else float(bright[i]),
                        "saturation": None if sat is None or not np.isfinite(sat[i]) else float(sat[i])})
    return {"labels": labels, "objects": objects, "params": _params(ground, deck_m)}


def _params(ground, deck_m: float = DECK_M) -> Dict:
    return {"ground_m": float(ground) if np.ndim(ground) == 0 else "surface", "deck_m": deck_m, "min_area_m2": MIN_AREA_M2,
            "min_half_width_m": MIN_HALF_WIDTH_M, "wall_slope_deg": WALL_SLOPE_DEG,
            "max_wall_fraction": MAX_WALL_FRACTION, "flat_slope_deg": FLAT_SLOPE_DEG,
            "max_flat_top": MAX_FLAT_TOP, "max_ndvi": MAX_NDVI, "dark_max": DARK_MAX,
            "light_min": LIGHT_MIN, "scrap_rough_m": SCRAP_ROUGH_M}
