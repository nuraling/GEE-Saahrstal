"""Earth Engine → numpy on a shared metric grid.

Every image is requested through ``ee.data.computePixels`` with an explicit
affine grid in EPSG:25832, so the UAV surfaces, Pléiades, SPOT, Sentinel-2 and
Landsat arrive pixel-aligned without any client-side reprojection. Requests are
tiled under Earth Engine's per-request limit.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from typing import Dict, List, Optional

import numpy as np

from .config import StockpileConfig
from .grid import Grid, grid_for_bounds

EE_REQUEST_BYTES = 40e6       # stay well under the 48 MB computePixels cap
MAX_DIM = 16384


def _ee():
    import ee
    return ee


def max_tile_px(n_bands: int, bytes_per_px: int = 8) -> int:
    return int(min(MAX_DIM, math.floor(math.sqrt(EE_REQUEST_BYTES / (n_bands * bytes_per_px)))))


def fetch(image, grid: Grid, bands: List[str], retries: int = 3, log=print) -> Dict[str, np.ndarray]:
    """Download `bands` of an ee.Image onto `grid`. Masked pixels → NaN."""
    ee = _ee()
    img = image.select(bands).toFloat()
    out = {b: np.full(grid.shape, np.nan, np.float32) for b in bands}
    tiles = grid.subgrids(max_tile_px(len(bands)))
    for t in tiles:
        for attempt in range(1, retries + 1):
            try:
                arr = ee.data.computePixels({"expression": img.unmask(-9999),
                                             "fileFormat": "NUMPY_NDARRAY", "grid": t.ee_grid()})
                break
            except Exception as exc:
                if attempt == retries:
                    raise
                log(f"computePixels retry {attempt}/{retries}: {exc}")
                time.sleep(5 * attempt)
        r0, c0 = t.offset_in(grid)
        for b in bands:
            a = np.asarray(arr[b], dtype=np.float32)
            a[a <= -9998] = np.nan
            out[b][r0:r0 + t.height, c0:c0 + t.width] = a
    return out


def site_grid(cfg=StockpileConfig, res: float = None, aoi_geojson: Optional[dict] = None) -> Grid:
    """Processing grid over the UAV DTM footprint (optionally ∩ AOI)."""
    ee = _ee()
    # every UAV surface, not just the DTM: DSM_3 (the port) lies outside the DTM
    geom = ee.Image(cfg.ASSET_DTM).geometry()
    for a in cfg.ASSET_DSMS:
        geom = geom.union(ee.Image(a).geometry(), 1)
    if aoi_geojson:
        geom = geom.intersection(ee.Geometry(aoi_geojson), 1)
    proj = ee.Projection(cfg.CRS)
    coords = geom.transform(proj, 1).bounds(1, proj).coordinates().getInfo()[0]
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    return grid_for_bounds((min(xs), min(ys), max(xs), max(ys)), res or cfg.PROC_RES_M, cfg.CRS)


# ── UAV ──────────────────────────────────────────────────────────────────────

def fetch_uav(grid: Grid, cfg=StockpileConfig, log=print) -> Dict[str, np.ndarray]:
    ee = _ee()
    dsm = ee.ImageCollection([ee.Image(a).select(0) for a in cfg.ASSET_DSMS]).mosaic().rename("dsm")
    dtm = ee.Image(cfg.ASSET_DTM).select(0).rename("dtm")
    ortho = ee.Image(cfg.ASSET_ORTHO)
    o = ortho.select([0, 1, 2], ["red", "green", "blue"])
    log("Fetching UAV DSM/DTM/ortho ...")
    out = fetch(dsm.addBands(dtm), grid, ["dsm", "dtm"], log=log)
    out["dsm_id"] = dsm_block_ids(grid, cfg)
    out.update(fetch(o, grid, ["red", "green", "blue"], log=log))
    return out


def dsm_block_ids(grid: Grid, cfg=StockpileConfig) -> np.ndarray:
    """Which UAV survey block (1..n) each pixel belongs to, from the DSM
    footprints rasterised locally (an EE mask expression returned no data)."""
    ee = _ee()
    from .grid import lonlat_geom_to_crs, rasterize_labels
    ids = np.zeros(grid.shape, np.float32)
    for i, a in enumerate(cfg.ASSET_DSMS, 1):
        geom = ee.Image(a).geometry().getInfo()
        m = rasterize_labels([lonlat_geom_to_crs(geom, grid.crs)], grid) > 0
        ids[m] = i
    return ids


def fetch_building_footprints(grid: Grid, cfg=StockpileConfig, log=print) -> Optional[dict]:
    """Building footprints over the site (lon/lat FeatureCollection)."""
    ee = _ee()
    try:
        fc = (ee.FeatureCollection(cfg.BUILDINGS_FC).filterBounds(grid_geometry(grid))
              .filter(ee.Filter.gte("area_in_meters", cfg.MIN_BUILDING_AREA_M2))
              .select(["area_in_meters", "bf_source"]))
        out = fc.getInfo()
        for f in out["features"]:
            f["properties"]["class"] = "roof"
            f["properties"]["footprint_source"] = f["properties"].pop("bf_source", "vida")
        log(f"Building footprints: {len(out['features'])} from {cfg.BUILDINGS_FC}")
        return out
    except Exception as exc:
        log(f"Building footprints unavailable: {exc}")
        return None


# ── optical sources ──────────────────────────────────────────────────────────

def _asset_bands(asset: str, band_map: Dict[str, int]):
    ee = _ee()
    img = ee.Image(asset)
    keys = list(band_map)
    return img.select([band_map[k] for k in keys], keys)


def zero_is_nodata(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """pleiades_saarfactory stores outside-the-scene as 0 in every band
    instead of masking it — 45% of the Saarlouis grid. Treated as data it
    became 'black ground' in training and 'imagery' over the port."""
    keys = [k for k in bands if not k.startswith("_")]
    dead = np.all(np.stack([np.nan_to_num(bands[k], nan=0.0) == 0 for k in keys]), axis=0)
    for k in keys:
        bands[k] = np.where(dead, np.nan, bands[k]).astype(np.float32)
    return bands


def fetch_pleiades(grid: Grid, cfg=StockpileConfig, log=print) -> Dict[str, np.ndarray]:
    log(f"Fetching Pléiades at {grid.res} m ...")
    return zero_is_nodata(fetch(_asset_bands(cfg.ASSET_PLEIADES, cfg.PLEIADES_BAND_MAP), grid,
                                cfg.OPTICAL_BANDS, log=log))


def band_order_check(bands: Dict[str, np.ndarray]) -> Dict:
    """Vegetation must show high NDVI and water/shadow low; if the red/NIR
    bands are swapped or mislabelled, NDVI p99 collapses. Same test used to
    work out pleiades_saarfactory's layout (NDVI p99 0.68)."""
    nd = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"] + 1e-6)
    nd = nd[np.isfinite(nd)]
    if nd.size == 0:
        return {"ok": False, "ndvi_p99": None, "note": "no valid pixels"}
    p99 = float(np.percentile(nd, 99))
    return {"ok": p99 > 0.4, "ndvi_p99": p99,
            "note": "ok" if p99 > 0.4 else "NDVI p99 too low — band map probably wrong; check bandNames()"}


def fetch_pleiades_recent(grid: Grid, cfg=StockpileConfig, log=print) -> List[Dict]:
    out = []
    for date, asset in cfg.PLEIADES_RECENT.items():
        log(f"Fetching Pléiades {date} ({asset}) at {grid.res} m ...")
        try:
            b = zero_is_nodata(fetch(_asset_bands(asset, cfg.PLEIADES_RECENT_BAND_MAP), grid,
                                     cfg.OPTICAL_BANDS, log=log))
        except Exception as exc:
            log(f"  Pléiades {date} not fetched: {exc}")
            continue
        chk = band_order_check(b)
        if not chk["ok"]:
            log(f"  WARNING Pléiades {date}: {chk['note']} (NDVI p99 {chk['ndvi_p99']})")
        out.append({"grid": grid, "date": date, "asset": asset, "band_check": chk,
                    "bands": {k: v for k, v in b.items() if not k.startswith("_")}})
    return out


def fetch_spot(grid: Grid, cfg=StockpileConfig, log=print) -> Optional[Dict[str, np.ndarray]]:
    if not cfg.ASSET_SPOT:
        log("No SPOT asset configured (STOCKPILE_ASSET_SPOT) — SPOT column skipped.")
        return None
    log(f"Fetching SPOT at {grid.res} m ...")
    return zero_is_nodata(fetch(_asset_bands(cfg.ASSET_SPOT, cfg.SPOT_BAND_MAP), grid,
                                cfg.OPTICAL_BANDS, log=log))


def _s2_mask(img):
    ee = _ee()
    scl = img.select("SCL")
    ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(ok)


def best_s2_scene(grid_geom, center_date: str, window_days: int, cfg=StockpileConfig):
    ee = _ee()
    d = ee.Date(center_date)
    col = (ee.ImageCollection(cfg.S2_COLLECTION).filterBounds(grid_geom)
           .filterDate(d.advance(-window_days, "day"), d.advance(window_days, "day"))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.S2_MAX_CLOUD_PCT))
           .map(lambda i: i.set("dt", ee.Number(i.date().difference(d, "day")).abs())))
    return col.sort("CLOUDY_PIXEL_PERCENTAGE").sort("dt").first()


def grid_geometry(grid: Grid):
    ee = _ee()
    return ee.Geometry.Rectangle([grid.x0, grid.y0, grid.x1, grid.y1], ee.Projection(grid.crs), False)


def fetch_s2(grid10: Grid, center_date: str, cfg=StockpileConfig, log=print,
             window_days: Optional[int] = None) -> Optional[Dict]:
    ee = _ee()
    img = best_s2_scene(grid_geometry(grid10), center_date, window_days or cfg.S2_WINDOW_DAYS, cfg)
    try:
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    except Exception:
        log("No clear Sentinel-2 scene near the UAV date.")
        return None
    bm = cfg.S2_BAND_MAP
    s2 = _s2_mask(img).select(list(bm.values()), list(bm.keys())).divide(10000)
    log(f"Fetching Sentinel-2 {date} at 10 m ...")
    bands = fetch(s2, grid10, list(bm.keys()), log=log)
    bands["_date"] = date
    return bands


# ── thermal time series ──────────────────────────────────────────────────────

def _landsat_lst(img):
    ee = _ee()
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("lst")
    unc = img.select("ST_QA").multiply(0.01).rename("lst_unc")
    ok = img.select("QA_PIXEL").bitwiseAnd(StockpileConfig.QA_REJECT_BITS).eq(0)
    return lst.addBands(unc).updateMask(ok).copyProperties(img, ["system:time_start", "SPACECRAFT_ID"])


def _window(center_date: str, days: int, start: Optional[str], end: Optional[str]):
    if start and end:
        return start, end
    c = dt.date.fromisoformat(center_date)
    return (str(c - dt.timedelta(days=days)), str(c + dt.timedelta(days=days)))


def _scene_list(col, geom, max_scenes: int, scale: float):
    """[(system:index, date, clear_fraction)] of scenes with enough clear pixels."""
    ee = _ee()

    def frac(img):
        c = img.select(0).mask().reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9)
        return img.set("clear", c.values().get(0))
    info = (col.map(frac).filter(ee.Filter.gt("clear", 0.6))
            .sort("system:time_start").limit(max_scenes)
            .reduceColumns(ee.Reducer.toList(3), ["system:index", "system:time_start", "clear"])
            .get("list").getInfo())
    return info


def fetch_landsat_scenes(grid30: Grid, center_date: str, start=None, end=None, max_scenes=24,
                         cfg=StockpileConfig, log=print) -> List[Dict]:
    ee = _ee()
    s, e = _window(center_date, cfg.THERMAL_WINDOW_DAYS, start, end)
    geom = grid_geometry(grid30)
    col = ee.ImageCollection([])
    for c in cfg.LANDSAT_COLLECTIONS:
        col = col.merge(ee.ImageCollection(c).filterBounds(geom).filterDate(s, e))
    col = col.map(_landsat_lst)
    scenes = []
    for idx, t, clear in _scene_list(col, geom, max_scenes, 30):
        img = col.filter(ee.Filter.eq("system:index", idx)).first()
        a = fetch(img, grid30, ["lst", "lst_unc"], log=log)
        date = dt.datetime.utcfromtimestamp(t / 1000).date().isoformat()
        scenes.append({"lst": a["lst"], "unc": a["lst_unc"], "date": date, "sensor": "Landsat",
                       "native_m": cfg.LANDSAT_TIRS_NATIVE_M, "night": False, "clear": clear})
    log(f"Landsat 8/9: {len(scenes)} usable scenes {s} → {e}")
    return scenes


def fetch_ecostress_scenes(grid30: Grid, center_date: str, start=None, end=None, max_scenes=30,
                           cfg=StockpileConfig, log=print) -> List[Dict]:
    """Day AND night ECOSTRESS passes — night is when a self-heating pile
    stands out from a yard that has cooled; Landsat never sees it."""
    ee = _ee()
    s, e = _window(center_date, cfg.THERMAL_WINDOW_DAYS, start, end)
    geom = grid_geometry(grid30)

    def prep(img):
        lst = img.select("LST").subtract(273.15).rename("lst")
        good = img.select("QC").bitwiseAnd(3).lte(1).And(img.select("cloud").eq(0))
        return lst.updateMask(good).copyProperties(img, ["system:time_start"])
    try:
        col = ee.ImageCollection(cfg.ECOSTRESS_COLLECTION).filterBounds(geom).filterDate(s, e).map(prep)
        # day and night separately: taking the first N by date returned no
        # night pass at all, and night is what building heat loss shows in
        day = col.filter(ee.Filter.calendarRange(7, 17, "hour"))
        night = col.filter(ee.Filter.Or(ee.Filter.calendarRange(19, 23, "hour"),
                                        ee.Filter.calendarRange(0, 4, "hour")))
        lst_ = (_scene_list(day, geom, max_scenes // 2, 70)
                + _scene_list(night, geom, max_scenes // 2, 70))
    except Exception as exc:
        log(f"ECOSTRESS unavailable: {exc}")
        return []
    scenes = []
    for idx, t, clear in lst_:
        img = col.filter(ee.Filter.eq("system:index", idx)).first()
        a = fetch(img, grid30, ["lst"], log=log)
        when = dt.datetime.utcfromtimestamp(t / 1000)
        local_h = (when.hour + 1) % 24   # CET; close enough to split day/night
        scenes.append({"lst": a["lst"], "date": when.date().isoformat(), "sensor": "ECOSTRESS",
                       "native_m": cfg.ECOSTRESS_NATIVE_M, "night": local_h < 6 or local_h >= 20,
                       "clear": clear})
    log(f"ECOSTRESS: {len(scenes)} usable scenes ({sum(s['night'] for s in scenes)} night)")
    return scenes


def fetch_s2_swir_scenes(grid10: Grid, center_date: str, start=None, end=None, max_scenes=20,
                         cfg=StockpileConfig, log=print) -> List[Dict]:
    ee = _ee()
    s, e = _window(center_date, cfg.THERMAL_WINDOW_DAYS, start, end)
    geom = grid_geometry(grid10)
    col = (ee.ImageCollection(cfg.S2_COLLECTION).filterBounds(geom).filterDate(s, e)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(_s2_mask)
           .map(lambda i: i.select(["B11", "B12"]).divide(10000)
                .copyProperties(i, ["system:time_start"])))
    scenes = []
    for idx, t, clear in _scene_list(col, geom, max_scenes, 20):
        img = col.filter(ee.Filter.eq("system:index", idx)).first()
        a = fetch(img, grid10, ["B11", "B12"], log=log)
        scenes.append({"b11": a["B11"], "b12": a["B12"], "clear": clear,
                       "date": dt.datetime.utcfromtimestamp(t / 1000).date().isoformat()})
    log(f"Sentinel-2 SWIR: {len(scenes)} usable scenes")
    return scenes


def load_site(cfg=StockpileConfig, aoi_geojson=None, thermal_start=None, thermal_end=None,
              sensors=None, log=print) -> Dict:
    """Everything the pipeline needs, as aligned numpy arrays."""
    sensors = sensors or list(cfg.SENSORS)
    g1 = site_grid(cfg, cfg.PROC_RES_M, aoi_geojson)
    log(f"Site grid: {g1.width}×{g1.height} px at {g1.res} m ({g1.width * g1.res / 1000:.2f} × "
        f"{g1.height * g1.res / 1000:.2f} km)")
    site = {"grid": g1, "uav": fetch_uav(g1, cfg, log), "optical": {}, "thermal": [], "swir": []}
    g10 = g1.with_res(10.0)
    if "pleiades" in sensors:
        site["optical"]["pleiades"] = {"grid": g1.with_res(cfg.SENSORS["pleiades"]["grid_m"]),
                                       "bands": None}
        site["optical"]["pleiades"]["bands"] = fetch_pleiades(site["optical"]["pleiades"]["grid"], cfg, log)
        site["pleiades_recent"] = fetch_pleiades_recent(site["optical"]["pleiades"]["grid"], cfg, log)
    if "spot" in sensors:
        gs = g1.with_res(cfg.SENSORS["spot"]["grid_m"])
        b = fetch_spot(gs, cfg, log)
        if b:
            site["optical"]["spot"] = {"grid": gs, "bands": b}
    if any(s.startswith("s2") for s in sensors):
        s2 = fetch_s2(g10, cfg.UAV_DATE, cfg, log)
        if s2:
            site["s2_raw"] = {"grid": g10, "bands": {k: v for k, v in s2.items() if not k.startswith("_")},
                              "date": s2["_date"]}
        site["s2_recent"] = []
        for d in cfg.S2_RECENT_DATES:
            r = fetch_s2(g10, d, cfg, log, window_days=cfg.S2_RECENT_WINDOW_DAYS)
            if r:
                site["s2_recent"].append({"grid": g10, "date": r["_date"],
                                          "bands": {k: v for k, v in r.items() if not k.startswith("_")}})
    g30 = g1.with_res(30.0)
    site["thermal_grid"] = g30
    site["thermal"] = (fetch_landsat_scenes(g30, cfg.UAV_DATE, thermal_start, thermal_end, cfg=cfg, log=log)
                       + fetch_ecostress_scenes(g30, cfg.UAV_DATE, thermal_start, thermal_end, cfg=cfg, log=log))
    site["building_footprints"] = fetch_building_footprints(g1, cfg, log)
    site["swir_grid"] = g10
    site["swir"] = fetch_s2_swir_scenes(g10, cfg.UAV_DATE, thermal_start, thermal_end, cfg=cfg, log=log)
    return site
