"""A synthetic harbour yard with known answers.

Used by the tests and by ``run_local.py --synthetic`` so the full chain can be
exercised without Earth Engine. The numbers it produces show that the
machinery works; they say nothing about how Pléiades or Sentinel-2 perform on
the real Saarlouis yard — only the real run does.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import ndimage

from .grid import Grid, block_mean
from .thermal import inverse_planck, planck_radiance

COLOURS = {  # pseudo-reflectance (blue, green, red, nir)
    "coal":        (0.06, 0.06, 0.06, 0.08),
    "iron_ore":    (0.12, 0.16, 0.30, 0.34),
    "wood_chips":  (0.25, 0.40, 0.62, 0.55),
    "limestone":   (0.70, 0.72, 0.74, 0.72),
    "sand_gravel": (0.40, 0.44, 0.50, 0.52),
}
GROUND = (0.20, 0.21, 0.22, 0.25)
ROOF = (0.35, 0.30, 0.28, 0.30)
VEG = (0.04, 0.10, 0.05, 0.50)


def _cone(X, Y, cx, cy, rx, ry, h, angle, flat=0.0):
    """Elongated pile at angle of repose, optionally with a flattened top."""
    ca, sa = np.cos(angle), np.sin(angle)
    u = ((X - cx) * ca + (Y - cy) * sa) / rx
    v = (-(X - cx) * sa + (Y - cy) * ca) / ry
    r = np.sqrt(u * u + v * v)
    z = h * np.clip(1 - r, 0, None)
    if flat:
        z = np.minimum(z, h * (1 - flat))
    return z


def make_site(width_m=600, height_m=450, res=1.0, seed=7, n_scenes=8) -> Dict:
    rng = np.random.default_rng(seed)
    g = Grid(360000.0, 5470020.0, res, int(width_m / res), int(height_m / res))
    X, Y = g.xy()
    X, Y = X - g.x0, g.y1 - Y
    dtm = 180.0 + 0.004 * X + 0.002 * Y + rng.normal(0, 0.02, g.shape)

    obj = np.zeros(g.shape, np.float32)
    material = np.full(g.shape, "", dtype=object)
    piles: List[Dict] = []
    specs = [  # cx, cy, rx, ry, h, angle, commodity
        (90, 90, 60, 30, 9.0, 0.5, "coal"), (220, 80, 45, 28, 7.0, 0.5, "coal"),
        (120, 220, 35, 22, 5.0, 0.2, "iron_ore"), (300, 230, 40, 40, 6.0, 0.0, "limestone"),
        (430, 110, 55, 25, 4.0, 0.9, "wood_chips"), (470, 300, 30, 18, 3.0, 0.3, "sand_gravel"),
        (230, 350, 25, 15, 2.2, 0.0, "coal"), (350, 380, 50, 20, 8.0, -0.4, "iron_ore"),
    ]
    for i, (cx, cy, rx, ry, h, a, com) in enumerate(specs):
        z = _cone(X, Y, cx, cy, rx, ry, h, a)
        m = z > 0.05
        obj = np.maximum(obj, z)
        material[m] = com
        ca, sa = np.cos(a), np.sin(a)
        ring = [(cx + (rx + 2) * np.cos(t) * ca - (ry + 2) * np.sin(t) * sa,
                 cy + (rx + 2) * np.cos(t) * sa + (ry + 2) * np.sin(t) * ca)
                for t in np.linspace(0, 2 * np.pi, 33)]
        coords = [[g.x0 + x, g.y1 - y] for x, y in ring]
        piles.append({"type": "Feature", "properties": {"name": f"S{i + 1}", "commodity_truth": com},
                      "geometry": {"type": "Polygon", "coordinates": [coords]}})

    buildings = []
    for (x0, y0, w, h_, ht) in [(520, 20, 60, 35, 9.0), (20, 380, 50, 50, 12.0), (400, 400, 40, 25, 7.0)]:
        m = (X >= x0) & (X < x0 + w) & (Y >= y0) & (Y < y0 + h_)
        obj[m] = ht
        material[m] = "roof"
        coords = [[g.x0 + x0, g.y1 - y0], [g.x0 + x0 + w, g.y1 - y0], [g.x0 + x0 + w, g.y1 - y0 - h_],
                  [g.x0 + x0, g.y1 - y0 - h_], [g.x0 + x0, g.y1 - y0]]
        buildings.append({"type": "Feature", "properties": {},
                          "geometry": {"type": "Polygon", "coordinates": [coords]}})
    for j in range(12):  # a row of wagons
        x0 = 30 + j * 16
        m = (X >= x0) & (X < x0 + 14) & (Y >= 290) & (Y < 293)
        obj[m] = 3.8
        material[m] = "roof"
    trees = ndimage.gaussian_filter(rng.random(g.shape), 3) > 0.53
    trees &= (X > 520) & (Y > 150) & (Y < 380)
    obj[trees] = np.maximum(obj[trees], 6 + rng.normal(0, 1.5, trees.sum()))
    material[trees] = "veg"

    dsm = dtm + obj + rng.normal(0, 0.03, g.shape)

    # optical: material colour × hillshade (sun from SSE, 45° elevation)
    gy, gx = np.gradient(dsm, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, el = np.radians(160), np.radians(45)
    shade = np.clip(np.sin(el) * np.cos(slope) + np.cos(el) * np.sin(slope) * np.cos(az - aspect), 0.1, 1)
    bands = {}
    for bi, name in enumerate(("blue", "green", "red", "nir")):
        base = np.full(g.shape, GROUND[bi], np.float32)
        for com, col in COLOURS.items():
            base[material == com] = col[bi]
        base[material == "roof"] = ROOF[bi]
        base[material == "veg"] = VEG[bi]
        bands[name] = (base * shade * 1.2 + rng.normal(0, 0.012, g.shape)).astype(np.float32) * 10000

    # thermal: coal pile #1 has a 5×5 m spot at 90 °C; mixed in radiance over a 100 m PSF
    g30 = g.with_res(30.0)
    hot = np.zeros(g.shape, bool)
    hot[(np.abs(X - 80) < 2.5) & (np.abs(Y - 85) < 2.5)] = True
    warm = _cone(X, Y, 220, 80, 45, 28, 7.0, 0.5) > 0.05
    scenes = []
    for s in range(n_scenes):
        t_bg = 18 + 8 * np.sin(s / n_scenes * np.pi) + rng.normal(0, 0.5)
        t = t_bg + 2.0 * (material == "roof") - 3.0 * (material == "veg") + rng.normal(0, 0.8, g.shape)
        t = t + 6.0 * warm              # pile S2: whole surface self-heating
        if s % 4 != 3:        # the hot spot is present in 3 of every 4 passes
            t = np.where(hot, 90.0, t)
        L = planck_radiance(t + 273.15)
        L = ndimage.gaussian_filter(L, 100 / res / 2.355)
        lst = block_mean((inverse_planck(L) - 273.15).astype(np.float32), g, g30)
        lst += rng.normal(0, 0.3, lst.shape).astype(np.float32)
        scenes.append({"lst": lst, "date": f"2025-{3 + s // 3:02d}-{1 + (s % 3) * 9:02d}",
                       "sensor": "Landsat", "native_m": 100.0, "night": False})
    g10 = g.with_res(10.0)
    swir = []
    for s in range(4):
        b11 = block_mean(bands["red"] / 10000 * 0.9, g, g10)
        b12 = b11 * 0.8
        if s < 2:
            r_, c_ = int(85 / 10), int(80 / 10)
            b12[r_, c_] = b11[r_, c_] * 1.6
        swir.append({"b11": b11, "b12": b12, "date": f"2025-0{4 + s}-10"})

    return {"grid": g, "uav": {"dsm": dsm.astype(np.float32), "dtm": dtm.astype(np.float32),
                               "red": bands["red"], "green": bands["green"], "blue": bands["blue"]},
            "optical": {"pleiades": {"grid": g, "bands": bands}},
            "thermal": scenes, "thermal_grid": g30, "swir": swir, "swir_grid": g10,
            "stockpiles": {"type": "FeatureCollection", "features": piles, "crs_code": g.crs},
            "buildings": {"type": "FeatureCollection", "features": buildings, "crs_code": g.crs},
            "truth": {"object_height": obj, "hotspot_label": 1}}


def s2_from_pleiades(site: Dict, seed: int = 3) -> Dict:
    """A simulated Sentinel-2 scene: Pléiades area-averaged to 10 m, with a
    different gain/offset and noise, as a different sensor on a different day."""
    rng = np.random.default_rng(seed)
    g = site["grid"]
    g10 = g.with_res(10.0)
    b = {}
    for k, v in site["optical"]["pleiades"]["bands"].items():
        a = block_mean(v / 10000.0, g, g10)
        b[k] = (0.9 * a + 0.01 + rng.normal(0, 0.006, a.shape)).astype(np.float32)
    return {"grid": g10, "bands": b, "date": "2025-05-03"}
