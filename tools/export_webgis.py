#!/usr/bin/env python3
"""Export a finished run as data files for the Leaflet WebGIS.

  python tools/export_webgis.py <run_dir> <out>/data [company] [place]

Everything is reprojected to EPSG:3857 before it becomes an image overlay: at
6.7°E the UTM-32 grid is rotated ~1.7° against web-mercator, which would shift
the edges of a 3.6 km site by tens of metres if overlaid unwarped.
"""
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image
from matplotlib import colormaps
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

sys.path.insert(0, ".")
from stockpile_engine.config import StockpileConfig as C  # noqa: E402

SRC_CRS = "EPSG:25832"
MERC = 1.0 / np.cos(np.radians(49.35))          # ground metre → web-mercator units here

# typical German loose bulk densities (Schüttdichte, t/m³) — one value per material, for the pitch
DENSITY_DE = {"coal": 0.85, "coke": 0.50, "iron_ore": 2.40, "limestone": 1.55, "sand_gravel": 1.65,
              "wood_chips": 0.30, "scrap_metal": 0.90, "slag": 1.80}
QUAY_LEVEL_M = 179.75       # Area 3 base: the quay deck, the dominant flat level above the water (177.85 m)
LABEL = {"coal": "Coal", "coke": "Coke", "iron_ore": "Iron ore", "limestone": "Limestone",
         "sand_gravel": "Sand / gravel", "wood_chips": "Wood chips", "scrap_metal": "Scrap metal",
         "slag": "Slag", "unknown": "Unclassified bulk"}


def warp(bands, transform, res_m, resampling=Resampling.bilinear):
    """bands: (n, H, W) float32 with NaN no-data, in EPSG:25832."""
    n, H, W = bands.shape
    left, top = transform.c, transform.f
    right, bottom = left + W * transform.a, top + H * transform.e
    dst_t, w, h = calculate_default_transform(SRC_CRS, "EPSG:3857", W, H, left, bottom, right, top,
                                              resolution=res_m * MERC)
    out = np.full((n, h, w), np.nan, np.float32)
    for i in range(n):
        reproject(bands[i], out[i], src_transform=transform, src_crs=SRC_CRS, dst_transform=dst_t,
                  dst_crs="EPSG:3857", resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    l, b, r, t = dst_t.c, dst_t.f + h * dst_t.e, dst_t.c + w * dst_t.a, dst_t.f
    w_, s_, e_, n_ = transform_bounds("EPSG:3857", "EPSG:4326", l, b, r, t)
    return out, [[s_, w_], [n_, e_]]


def save_webp(rgba, path, quality=78):
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", quality=quality, method=6)
    return os.path.getsize(path)


def rgb_to_rgba(rgb01):
    """rgb01: (3, H, W) in 0–1 with NaN → RGBA uint8, transparent where NaN."""
    valid = np.all(np.isfinite(rgb01), axis=0)
    rgba = np.zeros(rgb01.shape[1:] + (4,), np.uint8)
    rgba[..., :3] = (np.clip(np.nan_to_num(np.moveaxis(rgb01, 0, -1)), 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0)
    return rgba


def stretch(rgb, lo=2, hi=98):
    out = np.empty_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[0]):
        v = rgb[i][np.isfinite(rgb[i])]
        a, b = np.percentile(v, [lo, hi]) if v.size else (0, 1)
        out[i] = np.where(np.isfinite(rgb[i]), (rgb[i] - a) / max(b - a, 1e-6), np.nan)
    return out


def ramp(values, vmin, vmax, cmap, alpha=1.0, mask=None):
    v = (values - vmin) / (vmax - vmin)
    rgba = (colormaps[cmap](np.clip(np.nan_to_num(v), 0, 1)) * 255).astype(np.uint8)
    ok = np.isfinite(values) if mask is None else (np.isfinite(values) & mask)
    rgba[..., 3] = np.where(ok, int(alpha * 255), 0)
    return rgba


def num(v, nd=2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else round(f, nd)


def main(run_dir, out):
    os.makedirs(out, exist_ok=True)
    site = pickle.load(open("outputs/cache/saarlouis_site.pkl", "rb"))
    ck = pickle.load(open(f"{run_dir}/results_checkpoint.pkl", "rb"))
    rep = json.load(open(f"{run_dir}/report.json"))
    g, u = site["grid"], site["uav"]
    tr = from_origin(g.x0, g.y1, g.res, g.res)
    layers = {}

    # 1. site orthophoto (UAV 2025-05-01), 1 m
    rgb = np.stack([u["red"], u["green"], u["blue"]]).astype(np.float32)
    rgb = np.where(np.all(np.isfinite(rgb), 0) & (rgb.sum(0) > 0), rgb, np.nan) / 255.0
    w, bounds = warp(rgb, tr, 1.0)
    layers["ortho_site"] = {"file": "ortho_site.webp", "bounds": bounds, "date": C.UAV_DATE,
                            "label": "Drone orthophoto 2025-05-01", "bytes": save_webp(rgb_to_rgba(w), f"{out}/ortho_site.webp")}

    # 2. port imagery: drone 0.33 m, SPOT 2025/2026, Pléiades Neo 2026-08-03
    P = pickle.load(open("outputs/cache/port_redetect.pkl", "rb"))
    gp = P["grid"]
    po = np.load("outputs/cache/port_ortho_033.npy")
    po = np.where(po.sum(axis=2, keepdims=True) > 0, po, np.nan)
    w, bounds = warp(np.moveaxis(po, -1, 0).astype(np.float32), from_origin(gp.x0, gp.y1, gp.res / 3, gp.res / 3), 0.4)
    layers["ortho_port"] = {"file": "ortho_port.webp", "bounds": bounds, "date": C.UAV_DATE,
                            "label": "Drone orthophoto, port (0.33 m)", "bytes": save_webp(rgb_to_rgba(w), f"{out}/ortho_port.webp")}
    import ee
    from stockpile_engine.auth import initialize_earth_engine
    from stockpile_engine.grid import Grid
    from stockpile_engine.sources_ee import fetch
    initialize_earth_engine(log=lambda *a: None)
    R = C.ASSET_ROOT + "/"
    sats = [("spot_20250327", "spot_20250327_area3", [2, 1, 0], 1.5, "SPOT 6/7 · 2025-03-27", "2025-03-27"),
            ("spot_20260407", "spot_20260407_area3", [2, 1, 0], 1.5, "SPOT 6/7 · 2026-04-07", "2026-04-07"),
            ("pleiades_20260803", "Saarlouis/pleiades_20260803", [0, 1, 2], 0.5, "Pléiades Neo · 2026-08-03", "2026-08-03")]
    for key, asset, idx, res, label, date in sats:
        f = gp.res / res
        gs = Grid(gp.x0, gp.y1, res, int(gp.width * f), int(gp.height * f), gp.crs)
        px = fetch(ee.Image(R + asset).select(idx, ["r", "g", "b"]), gs, ["r", "g", "b"], log=lambda *a: None)
        rgb = np.stack([px["r"], px["g"], px["b"]]).astype(np.float32)
        rgb = np.where(np.all(rgb > 0, 0), rgb, np.nan)
        w, bounds = warp(stretch(rgb), from_origin(gs.x0, gs.y1, res, res), res)
        layers[key] = {"file": f"{key}.webp", "bounds": bounds, "date": date, "label": label,
                       "bytes": save_webp(rgb_to_rgba(w), f"{out}/{key}.webp")}

    # 3. height above ground (UAV) and downscaled LST, colour-ramped
    h = ck["ref"]["ndsm"].astype(np.float32)
    w, bounds = warp(h[None], tr, 1.0)
    layers["height"] = {"file": "height.webp", "bounds": bounds, "label": "Height above ground (drone)",
                        "min": 0, "max": 20, "unit": "m", "cmap": "viridis",
                        "bytes": save_webp(ramp(w[0], 0, 20, "viridis", 0.9, w[0] > 0.5), f"{out}/height.webp", 85)}
    import rasterio
    with rasterio.open(f"{run_dir}/rasters/lst_downscaled_3m.tif") as r:
        lst, lt = r.read(1).astype(np.float32), r.transform
    lo, hi = [float(x) for x in np.nanpercentile(lst, [2, 98])]
    w, bounds = warp(lst[None], lt, 3.0)
    layers["lst"] = {"file": "lst.webp", "bounds": bounds, "label": "Land surface temperature (Landsat/ECOSTRESS composite, downscaled 3 m)",
                     "min": round(lo, 1), "max": round(hi, 1), "unit": "°C", "cmap": "inferno",
                     "bytes": save_webp(ramp(w[0], lo, hi, "inferno", 0.8), f"{out}/lst.webp", 85)}
    json.dump({"cmaps": {c: [list((np.array(colormaps[c](x)[:3]) * 255).astype(int).tolist()) for x in np.linspace(0, 1, 9)]
                         for c in ("viridis", "inferno")}}, open(f"{out}/ramps.json", "w"))

    # 4. surface grid for the draw tools: DSM at 3 m, uint16 = (z − 150 m)·100, 0 = no data
    k = 3
    Hk, Wk = g.height // k, g.width // k
    d = u["dsm"][:Hk * k, :Wk * k].reshape(Hk, k, Wk, k)
    d = np.nanmean(d, axis=(1, 3))
    enc = np.where(np.isfinite(d), np.clip((d - 150.0) * 100, 1, 65535), 0).astype("<u2")
    import base64
    json.dump({"b64": base64.b64encode(enc.tobytes()).decode("ascii")}, open(f"{out}/dsm_3m.json", "w"))
    grid_meta = {"crs": SRC_CRS, "x0": g.x0, "y1": g.y1, "res": g.res * k, "width": Wk, "height": Hk,
                 "encoding": "base64 uint16 little-endian (dsm_3m.json), z = v/100 + 150 m, 0 = no data"}

    # 5. piles
    # third base for the range: min(UAV DTM, DSM-derived ground) outside the port (flat level in the port)
    from stockpile_engine import terrain
    rv = pickle.load(open("outputs/cache/site_review.pkl", "rb"))
    v_high = {v["label"]: v["volume_m3"] for v in terrain.pile_volumes(u["dsm"] - rv["ground"], ck["ref"]["labels"], g)}
    lab_of = {p["pile_id"]: p["label"] for p in ck["ref"]["piles"]}
    # Area 3 headline: height above the quay deck (booms removed), not above the lowest land level
    flat_lvl = (ck["ref"].get("dtm_check") or {}).get("levels_m", {}).get("DSM_3", 179.37)
    hq = np.clip(ck["ref"]["ndsm"] - (QUAY_LEVEL_M - flat_lvl), 0, None)
    v_quay = {v["label"]: v["volume_m3"] for v in terrain.pile_volumes(hq, ck["ref"]["labels"], g)}
    # alerts with SWIR hits pooled across adjacent piles (same rule as the pipeline now uses)
    from stockpile_engine.pipeline import _alert_level, pool_swir_neighbours
    stats = [dict(p) for p in rep["piles"]]
    pool_swir_neighbours(stats, ck["ref"]["labels"], g, C.SWIR_POOL_M)
    for s_ in stats:
        s_["alert"] = _alert_level(s_)
    rows = {p["pile_id"]: p for p in stats}
    ps = {}
    if os.path.exists(f"{run_dir}/port_spot/port_spot.json"):
        ps = {r["pile_id"]: r for r in json.load(open(f"{run_dir}/port_spot/port_spot.json"))["piles"]}
    gj = json.load(open(f"{run_dir}/piles.geojson"))
    pile_feats = []
    for f in gj["features"]:
        pid = f["properties"].get("pile_id")
        p = rows.get(pid, {})
        com = p.get("commodity") or "unknown"
        vol = p.get("volume_toe_base_m3") if p.get("uav_measured") else None
        dens = DENSITY_DE.get(com)
        port_pile = p.get("survey_block") == "DSM_3"
        head = (v_quay.get(lab_of.get(pid)) if port_pile else vol) if p.get("uav_measured") else None
        s = ps.get(pid, {})
        props = {
            "id": pid, "block": p.get("survey_block"), "area": {"DSM_1": "Area 1", "DSM_2": "Area 2", "DSM_3": "Area 3 · port"}.get(p.get("survey_block"), "outside survey"),
            "measured": bool(p.get("uav_measured")),
            "vol": num(head, 0), "vol_base": "quay deck 179.75 m" if port_pile else "own toe plane",
            "source": p.get("client_source") or p.get("source"),
            "material": com, "material_label": LABEL.get(com, com), "material_basis": p.get("commodity_source"),
            "area_m2": num(p.get("footprint_area_m2"), 0), "occupied_m2": num(p.get("occupied_area_m2"), 0),
            "h_max": num(p.get("max_height_m"), 1), "h_mean": num(p.get("mean_height_m"), 2),
            "vol_toe": num(vol, 0), "vol_ground": num(p.get("volume_m3"), 0), "vol_sigma": num(p.get("volume_sigma_m3"), 0),
            "vol_high": num(p.get("volume_m3") if p.get("survey_block") == "DSM_3" else v_high.get(lab_of.get(pid)), 0) if p.get("uav_measured") else None,
            "density": dens, "t": num(head * dens, 0) if head and dens else None,
            "alert": p.get("alert", "none"), "alert_reason": p.get("alert_reason"),
            "dT_mean": num(p.get("mean_delta_t_c"), 2), "dT_max": num(p.get("max_delta_t_c"), 2),
            "persistence": num(p.get("persistence"), 2), "n_scenes": p.get("n_scenes"),
            "night": [p.get("n_night_anomalous"), p.get("n_night_scenes")], "p_night": num(p.get("p_value_night"), 3),
            "trend": num(p.get("delta_t_trend_c_per_month"), 2),
            "hotspot5": num(p.get("hotspot_equiv_5m_c_median"), 0), "hotspot5_min": num(p.get("min_detectable_5m_hotspot_c"), 0),
            "swir_dates": p.get("swir_pool_dates") or p.get("swir_hot_dates") or [],
            "chg_s2_0813": num(p.get("change_2026-08-13_pct"), 1), "chg_s2_0922": num(p.get("change_2026-09-22_pct"), 1),
            "port_spot_2604": num(s.get("uav_scaled_2026_m3"), 0), "port_ple_2608": num(s.get("pleiades_2026-08-03_m3"), 0),
            "port_class": s.get("material_class"),
        }
        pile_feats.append({"type": "Feature", "geometry": f["geometry"], "properties": props})
    json.dump({"type": "FeatureCollection", "features": pile_feats}, open(f"{out}/piles.geojson", "w"), separators=(",", ":"))

    # 6. thermal assets
    bj = json.load(open(f"{run_dir}/buildings_thermal.geojson"))
    keep = ["object_id", "class", "area_m2", "max_height_m", "n_scenes", "mean_delta_t_c", "max_delta_t_c", "persistence",
            "n_night_scenes", "n_night_anomalous", "p_value_night", "day_persistence", "persistent_anomaly",
            "delta_t_trend_c_per_month", "hotspot_equiv_5m_c_median", "min_detectable_5m_hotspot_c",
            "max_lst_composite_c", "max_lst_downscaled_c", "alert", "alert_reason", "top10pct_hottest", "significant_vs_noise"]
    bfe = [{"type": "Feature", "geometry": f["geometry"],
            "properties": {k: num(f["properties"].get(k), 2) if not isinstance(f["properties"].get(k), (list, str, bool)) else f["properties"].get(k)
                           for k in keep}} for f in bj["features"]]
    json.dump({"type": "FeatureCollection", "features": bfe}, open(f"{out}/assets.geojson", "w"), separators=(",", ":"))

    # 7. summary: KPIs, validation, time series, method
    meas = [f["properties"] for f in pile_feats if f["properties"]["measured"]]
    by_mat = {}
    for p in meas:
        m = by_mat.setdefault(p["material"], {"n": 0, "vol": 0.0, "t": 0.0, "area": 0.0, "density": p["density"]})
        m["n"] += 1; m["vol"] += p["vol"] or 0; m["area"] += p["area_m2"] or 0; m["t"] += p["t"] or 0
    val = {}
    for sname, sres in rep["sensor_study"].items():
        if not isinstance(sres, dict) or "methods" not in sres:
            continue
        val[sname] = {"grid_m": sres.get("grid_m"), "native_m": sres.get("native_m"), "methods": {}}
        for mname, mres in sres["methods"].items():
            if not mres.get("available"):
                continue
            val[sname]["methods"][mname] = {c: {kk: num(mres[c].get(kk), 3) for kk in ("wape_pct", "bias_pct", "r2", "n")}
                                            for c in ("pile_volume_known_footprint", "pile_volume_bias_corrected",
                                                      "pile_volume_power_corrected") if mres.get(c)}
    # per-pile out-of-sample predictions against the drone volume they were scored on (survey ground)
    scatter = []
    for p in rep["piles"]:
        if not p.get("volume_m3"):
            continue
        row = {"id": p["pile_id"], "block": p.get("survey_block"), "uav": num(p["volume_m3"], 0)}
        for sname in ("pleiades", "s2_sr_3m", "s2_sr_1m", "s2_10m"):
            for suf, key in (("", "raw"), ("_bc", "bc"), ("_pc", "pc")):
                v_ = p.get(f"vol_{sname}_gbm_pile{suf}_m3")
                if v_:
                    row[f"{sname}:{key}"] = num(v_, 0)
        scatter.append(row)
    if os.path.exists(f"{run_dir}/port_spot/port_spot.json"):
        for r_ in json.load(open(f"{run_dir}/port_spot/port_spot.json"))["piles"]:
            for row in scatter:
                if row["id"] == r_["pile_id"] and r_.get("loo_height_2025_m3"):
                    row["spot:loo"] = num(r_["loo_height_2025_m3"], 0)
                    row["uav_port"] = num(r_["uav_2025_05_01_m3"], 0)
    port = json.load(open(f"{run_dir}/port_spot/port_spot.json")) if os.path.exists(f"{run_dir}/port_spot/port_spot.json") else None
    therm = ck.get("therm", {})
    scenes = [{"date": str(s.get("date")), "sensor": s.get("sensor"), "night": bool(s.get("night"))}
              for s in therm.get("scenes_meta", [])] if therm.get("scenes_meta") else []
    summary = {
        "site": "Saarlouis", "uav_date": C.UAV_DATE, "generated": rep.get("generated_utc"),
        "grid": grid_meta, "layers": layers,
        "kpi": {"n_piles": len(pile_feats), "n_measured": len(meas),
                "vol_toe": sum(p["vol_toe"] or 0 for p in meas), "vol_ground": sum(p["vol_ground"] or 0 for p in meas),
                "vol_high": sum(p["vol_high"] or 0 for p in meas),
                "vol": sum(p["vol"] or 0 for p in meas), "t": sum(p["t"] or 0 for p in meas),
                "vol_unclassified": sum(p["vol"] or 0 for p in meas if not p["t"]),
                "vol_sigma": float(np.sqrt(sum((p["vol_sigma"] or 0) ** 2 for p in meas))),
                "area": sum(p["area_m2"] or 0 for p in meas), "by_material": by_mat,
                "alerts": {a: sum(1 for p in pile_feats if p["properties"]["alert"] == a) for a in ("critical", "warning", "watch")},
                "assets": len(bfe), "assets_top10": sum(1 for f in bfe if f["properties"].get("top10pct_hottest")),
                "assets_night": sum(1 for f in bfe if f["properties"].get("persistent_anomaly")),
                "assets_alerts": {a: sum(1 for f in bfe if f["properties"].get("alert") == a) for a in ("critical", "warning", "watch")},
                "top10_threshold_c": num(rep["summary"].get("building_top10pct_threshold_c"), 2),
                "n_thermal_scenes": rep["summary"].get("n_thermal_scenes")},
        "validation": val, "scatter": scatter,
        "s2_recent": rep.get("recent_sentinel"), "pleiades_recent": rep.get("recent_pleiades"),
        "port": {"totals": port["totals"], "transfer": port.get("transfer"), "study": {k: port["study_2025"].get(k) for k in
                 ("height_chain_raw", "height_chain_power", "footprint_chain", "sensitivity")},
                 "chosen": port["chosen_chain"]} if port else None,
        "density": DENSITY_DE, "quay_level_m": QUAY_LEVEL_M, "labels": LABEL, "company": COMPANY, "place": PLACE, "caveats": rep.get("caveats"), "scenes": scenes,
        "dtm_check": rep.get("dtm_check_port_ground"), "asset_note": rep.get("asset_note"),
        "asset_source": ck.get("therm", {}).get("asset_source"),
    }
    json.dump(summary, open(f"{out}/summary.json", "w"), default=lambda o: None, separators=(",", ":"))
    for k, v in layers.items():
        print(f"{k:20s} {v['bytes'] / 1e6:5.2f} MB")
    print("piles", len(pile_feats), "assets", len(bfe))


if __name__ == "__main__":
    COMPANY = sys.argv[3] if len(sys.argv) > 3 else ""
    PLACE = sys.argv[4] if len(sys.argv) > 4 else ""
    main(sys.argv[1], sys.argv[2])
