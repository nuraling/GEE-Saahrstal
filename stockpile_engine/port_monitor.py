"""Port stock over time from Sentinel-2 alone.

No UAV flights after 01.05.2025. The model is fixed once on that survey and
then run on every cloud-free Sentinel-2 scene:

  Sentinel-2 10 m ──radiometric match to the 2025 reference scene──►
  material classifier (trained on the steel-works piles' UAV footprints)
  ──► footprint area per pile (soft, sub-pixel) inside a 20 m search zone
  ──► volume = c · area^b, fitted on the 7 port piles (UAV 2025)

Why area and not image height: on the port, Depth Anything's relief on the
super-resolved 3 m image is flat (±0.002). Its calibrated height was the same
constant everywhere, so its "volume" was polygon area × constant, and it did
not change when a pile was erased from the image. Area is what a 10 m sensor
measures; the 2025 UAV survey shows port volume follows area closely (V ∝ A^b,
leave-one-out WAPE 17 %, R² 0.91). A pile growing, shrinking or being cleared
changes its footprint, so this chain sees change; it cannot see a pile
getting taller on the same footprint.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from . import terrain
from .config import StockpileConfig
from .grid import Grid, resample, to_grid
from .metrics import object_metrics

CLEAR_SCL = (4, 5, 6, 7)          # vegetation, bare, water, unclassified
MIN_CLEAR_PORT = 0.98


# ── Earth Engine: cloud-free scenes over the port ────────────────────────────

def list_clear_scenes(port_geom_lonlat: dict, start: str, end: str, log=print) -> List[Dict]:
    """[(system:index, date, clear share over the port)], one per date."""
    import ee
    geom = ee.Geometry(port_geom_lonlat)
    col = (ee.ImageCollection(StockpileConfig.S2_COLLECTION).filterBounds(geom).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))

    def clear(img):
        scl = img.select("SCL")
        ok = scl.eq(CLEAR_SCL[0])
        for v in CLEAR_SCL[1:]:
            ok = ok.Or(scl.eq(v))
        share = ok.reduceRegion(ee.Reducer.mean(), geom, 20, maxPixels=1e8).get("SCL")
        return img.set("port_clear", share)
    info = (col.map(clear).filter(ee.Filter.gte("port_clear", MIN_CLEAR_PORT))
            .reduceColumns(ee.Reducer.toList(3), ["system:index", "system:time_start", "port_clear"])
            .get("list").getInfo())
    by_date = {}
    for idx, t, c in info:
        d = dt.datetime.utcfromtimestamp(t / 1000).date().isoformat()
        if d not in by_date or c > by_date[d]["clear"]:
            by_date[d] = {"index": idx, "date": d, "clear": c}
    out = [by_date[d] for d in sorted(by_date)]
    log(f"Sentinel-2 {start} → {end}: {len(out)} dates with ≥ {MIN_CLEAR_PORT:.0%} clear sky over the port")
    return out


def fetch_scene(index: str, g10: Grid, log=print) -> Dict[str, np.ndarray]:
    import ee
    from .sources_ee import fetch
    bm = StockpileConfig.S2_BAND_MAP
    img = (ee.ImageCollection(StockpileConfig.S2_COLLECTION).filter(ee.Filter.eq("system:index", index))
           .first().select(list(bm.values()), list(bm.keys())).divide(10000))
    return fetch(img, g10, list(bm.keys()), log=log)


# ── the fixed model ──────────────────────────────────────────────────────────

class PortModel:
    """Everything fitted once on 2025 (S2 2025-05-01 + UAV 2025-05-01)."""

    SEARCH_M = 20.0

    def __init__(self, site: Dict, ref: Dict, cfg=StockpileConfig, log=print):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from .pipeline import _normalise_to_reference, _power_fit
        self.cfg, self.log = cfg, log
        self.g: Grid = site["grid"]
        self.g10: Grid = site["s2_raw"]["grid"]
        self.labels = ref["labels"]
        self.piles = ref["piles"]
        self.port = np.array([p.get("survey_block") == f"DSM_{cfg.PORT_SURVEY_BLOCK}" for p in self.piles])
        self.ref_bands = {k: v for k, v in site["s2_raw"]["bands"].items()}
        self.labels10 = np.rint(resample(self.labels.astype(np.float32), self.g, self.g10, order=0)).astype(np.int32)
        self._norm = _normalise_to_reference

        # search zones: each pile's polygon grown by SEARCH_M, split between neighbours
        dist, (ri, ci) = ndimage.distance_transform_edt(self.labels10 == 0, return_indices=True)
        self.zone10 = np.where(dist * self.g10.res <= self.SEARCH_M, self.labels10[ri, ci], 0).astype(np.int32)

        # material fraction per 10 m pixel from the UAV (the target)
        occ = to_grid(ref["occupied"].astype(np.float32), self.g, self.g10)
        X = self._features({k: self.ref_bands[k] for k in cfg.OPTICAL_BANDS + ["swir1", "swir2"]
                            if k in self.ref_bands})
        port_labels = [p["label"] for p, q in zip(self.piles, self.port) if q]
        valid = (self.zone10 > 0) & np.isfinite(occ) & np.all(np.isfinite(X), axis=2)
        yall = (occ > 0.5).astype(int)
        uav_area = np.array([np.nan if p["occupied_area_m2"] is None else p["occupied_area_m2"] for p in self.piles])
        uav = np.array([np.nan if p["volume_m3"] is None else p["volume_m3"] for p in self.piles])

        def fit(mask):
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=cfg.RAND_SEED)
            c.fit(X[mask], yall[mask])
            return c
        # leave-one-pile-out on the port: pile i's area comes from a classifier
        # trained on every zone except pile i's (port materials included)
        idx = np.array([i for i, p in enumerate(self.piles)
                        if self.port[i] and np.isfinite(uav[i]) and p["label"] in port_labels])
        a_loo = np.full(len(uav), np.nan)
        for i in idx:
            self.clf = fit(valid & (self.zone10 != self.piles[i]["label"]))
            a_loo[i] = self._areas(X)[i]
        self.area_check = object_metrics(uav_area[idx], a_loo[idx])
        # area → volume, fitted on satellite areas (absorbs any systematic area
        # bias); leave-one-out end to end, then the final fit
        loo = np.full(len(uav), np.nan)
        for i in idx:
            o = np.setdiff1d(idx, [i])
            fn, _ = _power_fit(uav[o], a_loo[o])
            loo[i] = fn(a_loo[i]) if fn else np.nan
        self.accuracy = object_metrics(uav[idx], loo[idx])
        # production: classifier on every zone, calibration on its own port areas
        self.clf = fit(valid)
        a_ref = self._areas(X)
        self.correct, self.exponent = _power_fit(uav[idx], a_ref[idx])
        self.port_idx, self.uav, self.uav_area, self.ref_area = idx, uav, uav_area, a_ref
        log(f"Port footprint, leave-one-pile-out: area WAPE {self.area_check['wape_pct']:.0f}%, "
            f"bias {self.area_check['bias_pct']:+.0f}%")
        log(f"Port area→volume: V = c·A^{self.exponent:.2f}; end-to-end leave-one-out volume WAPE "
            f"{self.accuracy['wape_pct']:.0f}%, R² {self.accuracy.get('r2') or 0:.2f}")

    def _features(self, b: Dict[str, np.ndarray]) -> np.ndarray:
        bright = (b["red"] + b["green"] + b["blue"]) / 3
        ndvi = (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 1e-6)
        ndwi = (b["green"] - b["nir"]) / (b["green"] + b["nir"] + 1e-6)
        red = (b["red"] - b["blue"]) / (b["red"] + b["blue"] + 1e-6)
        fs = [b[k] for k in ("blue", "green", "red", "nir")] + [bright, ndvi, ndwi, red]
        if "swir1" in b and "swir2" in b:
            fs += [b["swir1"], b["swir2"], (b["swir2"] - b["swir1"]) / (b["swir2"] + b["swir1"] + 1e-6)]
        m = ndimage.uniform_filter(np.nan_to_num(bright), 3)
        fs.append(np.sqrt(np.clip(ndimage.uniform_filter(np.nan_to_num(bright) ** 2, 3) - m * m, 0, None)))
        return np.dstack(fs).astype(np.float32)

    def _areas(self, X: np.ndarray) -> np.ndarray:
        ok = np.all(np.isfinite(X), axis=2) & (self.zone10 > 0)
        prob = np.zeros(self.zone10.shape, np.float32)
        prob[ok] = self.clf.predict_proba(X[ok])[:, 1]
        n = len(self.piles)
        return ndimage.sum(prob, self.zone10, np.arange(1, n + 1)) * self.g10.pixel_area

    def scene_volumes(self, bands10: Dict[str, np.ndarray]) -> Dict:
        excl = ndimage.binary_dilation(self.labels10 > 0, iterations=2)
        keys = [k for k in self.cfg.OPTICAL_BANDS + ["swir1", "swir2"] if k in bands10 and k in self.ref_bands]
        nb = self._norm({k: bands10[k] for k in self.cfg.OPTICAL_BANDS},
                        {k: self.ref_bands[k] for k in self.cfg.OPTICAL_BANDS}, excl)
        b = dict(nb["bands"])
        for k in ("swir1", "swir2"):                  # match SWIR the same way
            if k in keys:
                x, yv = bands10[k][~excl], self.ref_bands[k][~excl]
                ok = np.isfinite(x) & np.isfinite(yv)
                a, c = np.polyfit(x[ok], yv[ok], 1) if ok.sum() > 50 else (1.0, 0.0)
                b[k] = a * bands10[k] + c
        area = self._areas(self._features(b))
        return {"area": area, "corrected": self.correct(area), "radiometric_fit": nb["fit"]}


def sensitivity_check(model, min_response: float = 0.2) -> Dict:
    """Erase the largest port pile from the reference image (its pixels set to
    the surrounding ground) and see whether its volume responds. A model that
    cannot see a pile disappear cannot track stock over time, however good its
    leave-one-out score — the Depth Anything chain scored 18 % that way while
    measuring only polygon area."""
    base = model.scene_volumes(model.ref_bands)["corrected"]
    i = max(model.port_idx, key=lambda j: model.uav[j])
    k = model.piles[i]["label"]
    m = model.labels10 == k
    ring = ndimage.binary_dilation(m, iterations=3) & (model.labels10 == 0)
    b = {kk: v.copy() for kk, v in model.ref_bands.items()}
    for kk in b:
        b[kk][m] = np.nanmedian(b[kk][ring])
    v = model.scene_volumes(b)["corrected"]
    resp = float(1 - v[i] / base[i]) if base[i] > 0 else 0.0
    return {"pile_id": model.piles[i]["pile_id"], "volume_with_pile_m3": float(base[i]),
            "volume_pile_erased_m3": float(v[i]), "response": resp, "passed": resp >= min_response}


# ── run ──────────────────────────────────────────────────────────────────────

def run_port_monitor(site: Dict, ref: Dict, outdir: str, start: str, end: str,
                     cfg=StockpileConfig, log=print) -> Dict:
    from .grid import crs_geom_to_lonlat, vectorize_labels
    os.makedirs(outdir, exist_ok=True)
    model = PortModel(site, ref, cfg, log)
    sens = sensitivity_check(model)
    log(f"Sensitivity check: erasing {sens['pile_id']} from the image changes its volume by "
        f"{-100 * sens['response']:+.0f}% → {'PASSED' if sens['passed'] else 'FAILED: not usable for change'}")
    g = model.g
    piles = [p for p, q in zip(model.piles, model.port) if q]
    labs = [p["label"] for p in piles]

    # scrap vs bulk (UAV surface roughness + Sentinel-2 brightness patchiness, 2025)
    for p in piles:
        p.update(terrain.material_class(site["uav"]["dsm"], model.labels, p["label"],
                                        model.ref_bands, model.labels10, g.res))
        p["bulk_density_t_m3"] = cfg.BULK_DENSITY_T_M3["scrap_metal" if p["material_class"] == "scrap_metal"
                                                       else "unknown"]

    # port footprint (lon/lat) for the scene search
    port_mask = np.isin(model.labels, labs)
    geoms = vectorize_labels(ndimage.binary_dilation(port_mask, iterations=30).astype(np.int32), g)
    port_geom = crs_geom_to_lonlat(geoms[1], g.crs)
    scenes = list_clear_scenes(port_geom, start, end, log)

    rows, series = [], []
    ref_date = site["s2_raw"].get("date")
    ref_corr = model.correct(model.ref_area)
    series.append({"date": ref_date, "source": "Sentinel-2 (model, reference date)",
                   "per_pile": {p["pile_id"]: float(ref_corr[p["label"] - 1]) for p in piles}})
    series.append({"date": cfg.UAV_DATE, "source": "UAV survey",
                   "per_pile": {p["pile_id"]: float(model.uav[p["label"] - 1]) for p in piles}})
    for sc in scenes:
        try:
            b = fetch_scene(sc["index"], model.g10, log=lambda *_: None)
            r = model.scene_volumes(b)
        except Exception as exc:
            log(f"  {sc['date']}: skipped ({exc})")
            continue
        per = {p["pile_id"]: float(r["corrected"][p["label"] - 1]) for p in piles}
        area = {p["pile_id"]: float(r["area"][p["label"] - 1]) for p in piles}
        fit_r2 = float(np.nanmean([f["r2"] for f in r["radiometric_fit"].values() if f.get("r2") is not None]))
        series.append({"date": sc["date"], "source": "Sentinel-2", "per_pile": per, "area": area,
                       "clear": sc["clear"], "radiometric_r2": fit_r2})
        log(f"  {sc['date']}: port total {sum(per.values()):,.0f} m³ (radiometric match R² {fit_r2:.2f})")

    # tables
    for s in series:
        for p in piles:
            v = s["per_pile"][p["pile_id"]]
            rows.append({"date": s["date"], "source": s["source"], "pile_id": p["pile_id"],
                         "material_class": p["material_class"], "volume_m3": v,
                         "footprint_m2": (s.get("area") or {}).get(p["pile_id"]),
                         "tonnage_t": v * p["bulk_density_t_m3"]})
    totals = []
    for s in series:
        t = {"date": s["date"], "source": s["source"],
             "total_m3": sum(s["per_pile"].values()),
             "scrap_m3": sum(v for k, v in s["per_pile"].items()
                             if next(p for p in piles if p["pile_id"] == k)["material_class"] == "scrap_metal"),
             "clear": s.get("clear"), "radiometric_r2": s.get("radiometric_r2")}
        t["bulk_m3"] = t["total_m3"] - t["scrap_m3"]
        totals.append(t)
    from .outputs import write_csv, _jsonable
    write_csv(os.path.join(outdir, "port_piles_timeseries.csv"), _jsonable(rows))
    write_csv(os.path.join(outdir, "port_totals_timeseries.csv"), _jsonable(totals))
    result = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "window": [start, end], "n_scenes": sum(1 for s in series if s["source"] == "Sentinel-2"),
              "model": {"chain": "S2 10 m (matched to 2025-05-01) → material classifier trained on the "
                                 "2025 UAV footprints → footprint area per pile → V = c·A^b fitted on the "
                                 "port piles (UAV 2025)",
                        "power_exponent": model.exponent,
                        "loo_accuracy_2025_port": model.accuracy,
                        "port_area_check": model.area_check,
                        "why_not_depth_anything": "On the port, Depth Anything relief on the 3 m image is flat "
                                                  "(±0.002): its volume was polygon area × constant and did not "
                                                  "change when a pile was erased from the image."},
              "piles": [{k: p.get(k) for k in ("pile_id", "material_class", "surface_roughness_m",
                                               "brightness_patchiness", "bulk_density_t_m3", "volume_m3")}
                        for p in piles],
              "totals": totals, "sensitivity_check": sens, "usable_for_change": sens["passed"]}
    with open(os.path.join(outdir, "port_monitor.json"), "w") as fh:
        json.dump(_jsonable(result), fh, indent=2)
    figs = port_figures(site, model, piles, series, totals, outdir)
    write_port_html(os.path.join(outdir, "port_report.html"), result, figs)
    return result


# ── figures and report ───────────────────────────────────────────────────────

def port_figures(site, model, piles, series, totals, outdir) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from .outputs import SERIES, SURFACE, INK, INK2, GRID, _style, _background, _step, _xy
    figs = {}
    s2 = [t for t in totals if t["source"] == "Sentinel-2"]
    ref = [t for t in totals if t["source"] != "Sentinel-2"]
    dates = [dt.date.fromisoformat(t["date"]) for t in s2]

    # 1 · port total over time, split scrap / bulk
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.plot(dates, [t["total_m3"] / 1000 for t in s2], color=INK, lw=2, marker="o", ms=4, label="port total")
    ax.plot(dates, [t["bulk_m3"] / 1000 for t in s2], color=SERIES[0], lw=2, marker="o", ms=4, label="bulk material")
    ax.plot(dates, [t["scrap_m3"] / 1000 for t in s2], color=SERIES[1], lw=2, marker="o", ms=4, label="scrap metal")
    uav = next(t for t in ref if t["source"] == "UAV survey")
    for key, col in (("total_m3", INK), ("bulk_m3", SERIES[0]), ("scrap_m3", SERIES[1])):
        ax.scatter([dt.date.fromisoformat(uav["date"])], [uav[key] / 1000], marker="s", s=60,
                   facecolor="none", edgecolor=col, linewidth=1.8, zorder=5)
    ax.annotate("UAV survey 2025-05-01", (dt.date.fromisoformat(uav["date"]), uav["total_m3"] / 1000),
                textcoords="offset points", xytext=(8, 6), fontsize=8, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_ylabel("Volume (1000 m³)", color=INK2, fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
    ax.set_title("Port stock (DSM_3) — Sentinel-2 footprint area → volume, port-calibrated", loc="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    figs["totals"] = os.path.join(outdir, "port_totals.png")
    fig.savefig(figs["totals"], dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # 2 · per pile over time (small multiples)
    n = len(piles)
    cols = 4
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(12, 2.6 * rws), facecolor=SURFACE, squeeze=False)
    s2s = [s for s in series if s["source"] == "Sentinel-2"]
    for ax, p in zip(axes.ravel(), piles):
        _style(ax)
        col = SERIES[1] if p["material_class"] == "scrap_metal" else SERIES[0]
        ax.plot(dates, [s["per_pile"][p["pile_id"]] / 1000 for s in s2s], color=col, lw=1.8, marker="o", ms=3)
        ax.scatter([dt.date.fromisoformat(uav["date"])], [model.uav[p["label"] - 1] / 1000], marker="s", s=40,
                   facecolor="none", edgecolor=INK, linewidth=1.5)
        ax.set_title(f"{p['pile_id']} · {p['material_class'].replace('_', ' ')}", fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, None)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Per pile (1000 m³); square = UAV 2025", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    figs["piles"] = os.path.join(outdir, "port_piles.png")
    fig.savefig(figs["piles"], dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # 3 · map: scrap vs bulk with the latest volume
    g = model.g
    labs = model.labels
    rows_, cols_ = np.nonzero(np.isin(labs, [p["label"] for p in piles]))
    x0, x1 = g.x0 + cols_.min() * g.res - 60, g.x0 + cols_.max() * g.res + 60
    y0, y1 = g.y1 - rows_.max() * g.res - 60, g.y1 - rows_.min() * g.res + 60
    fig, ax = plt.subplots(figsize=(9, 9 * (y1 - y0) / (x1 - x0)), facecolor=SURFACE)
    _background(ax, site)
    k = _step(g.shape)
    X, Y = _xy(g, k)
    latest = s2s[-1] if s2s else None
    import matplotlib.patheffects as pe
    for p in piles:
        m = labs[::k, ::k] == p["label"]
        col = SERIES[1] if p["material_class"] == "scrap_metal" else SERIES[0]
        ax.contour(X, Y, m.astype(np.float32), levels=[0.5], colors=[col], linewidths=2.4)
        rr, cc = np.nonzero(m)
        txt = f"{p['pile_id'].replace('Pile_', '')} · {p['material_class'].replace('_', ' ')}"
        if latest:
            txt += f"\n{latest['per_pile'][p['pile_id']]:,.0f} m³ ({latest['date']})"
        ax.annotate(txt, (g.x0 + (cc.mean() * k + .5) * g.res, g.y1 - (rr.mean() * k + .5) * g.res),
                    ha="center", fontsize=8, color=INK, path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=SERIES[0], lw=2.5, label="bulk material"),
                       Line2D([0], [0], color=SERIES[1], lw=2.5, label="scrap metal")],
              loc="lower right", fontsize=8, facecolor=SURFACE)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Port piles — scrap metal vs bulk (UAV roughness + Sentinel-2 patchiness)", loc="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    figs["map"] = os.path.join(outdir, "port_map.png")
    fig.savefig(figs["map"], dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return figs


def write_port_html(path, result, figs):
    import html
    from .outputs import _img_tag, _fmt
    acc = result["model"]["loo_accuracy_2025_port"]
    tot = result["totals"]
    s2 = [t for t in tot if t["source"] == "Sentinel-2"]
    uav = next(t for t in tot if t["source"] == "UAV survey")
    ref = next(t for t in tot if t["source"].startswith("Sentinel-2 (model"))
    trows = "".join(
        f"<tr><td>{html.escape(t['date'])}</td><td>{html.escape(t['source'])}</td>"
        f"<td class=n><b>{_fmt(t['total_m3'], 0)}</b></td><td class=n>{_fmt(t['bulk_m3'], 0)}</td>"
        f"<td class=n>{_fmt(t['scrap_m3'], 0)}</td>"
        f"<td class=n>{_fmt(100 * (t['total_m3'] - uav['total_m3']) / uav['total_m3'], 0)}</td>"
        f"<td class=n>{_fmt(t.get('radiometric_r2'), 2)}</td></tr>" for t in [uav, ref] + s2)
    prow = "".join(
        f"<tr><td>{html.escape(p['pile_id'])}</td><td>{html.escape(p['material_class'].replace('_', ' '))}</td>"
        f"<td class=n>{_fmt(p['surface_roughness_m'], 2)}</td><td class=n>{_fmt(p['brightness_patchiness'], 3)}</td>"
        f"<td class=n>{_fmt(p['volume_m3'], 0)}</td><td class=n>{_fmt(p['bulk_density_t_m3'], 2)}</td></tr>"
        for p in result["piles"])
    first, last = (s2[0], s2[-1]) if s2 else (None, None)
    kpis = ""
    if s2:
        vals = [t["total_m3"] for t in s2]
        kpis = (f"<div class=kpi><b>{len(s2)}</b><span>cloud-free dates {html.escape(result['window'][0])} → "
                f"{html.escape(result['window'][1])}</span></div>"
                f"<div class=kpi><b>{_fmt(last['total_m3'], 0)} m³</b><span>port total, {html.escape(last['date'])}</span></div>"
                f"<div class=kpi><b>{_fmt(last['scrap_m3'], 0)} m³</b><span>of which scrap metal</span></div>"
                f"<div class=kpi><b>{_fmt(min(vals), 0)}–{_fmt(max(vals), 0)} m³</b><span>range over the year</span></div>"
                f"<div class=kpi><b>{_fmt(acc.get('wape_pct'), 0)} %</b><span>per-pile error (2025, leave-one-out, n={acc.get('n')})</span></div>")
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Port Stock Monitor</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#383835;--card:#242423}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px 16px}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin-top:28px}}
.sub{{color:var(--ink2)}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}} .kpi b{{display:block;font-size:20px}}
.kpi span{{color:var(--ink2);font-size:12px}} .tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}} figure{{margin:14px 0}}
</style></head><body><main>
<h1>Port Stock Monitor — Saarlouis port (DSM_3)</h1>
<div class=sub>Sentinel-2 only after the 01.05.2025 UAV survey · generated {html.escape(result['generated_utc'])}</div>
{"" if result.get("usable_for_change", True) else
  "<div style='border:2px solid #d03b3b;border-radius:8px;padding:12px;margin:16px 0'><b>Not usable for tracking change.</b> "
  + html.escape(f"Sensitivity check: erasing {result['sensitivity_check']['pile_id']} from the reference image changed its "
                f"volume by {-100 * result['sensitivity_check']['response']:+.0f}%. The model does not see piles appear or "
                "disappear at this resolution; the series below reflects the calibration, not the port.") + "</div>"}
<div class=kpis>{kpis}</div>
<figure>{_img_tag(figs.get('totals'))}</figure>
<h2>Totals per date</h2>
<div class=tw><table><thead><tr><th>Date</th><th>Source</th><th>Port total m³</th><th>Bulk m³</th><th>Scrap metal m³</th>
<th>vs UAV 2025 %</th><th>Radiometric match R²</th></tr></thead><tbody>{trows}</tbody></table></div>
<h2>Scrap metal vs bulk</h2>
<p class=sub>Scrap heaps are jagged and patchy: UAV DSM roughness ≥ 0.6 m (detrended over 3 m) <i>and</i> Sentinel-2 brightness
spread ≥ 0.015 inside the pile. Bulk material is smooth and uniform. Density for tonnage: scrap 0.90 t/m³, bulk unknown (1.0) until the
operator names the material.</p>
<div class=tw><table><thead><tr><th>Pile</th><th>Class</th><th>Roughness m</th><th>Patchiness</th><th>UAV 2025 m³</th><th>Density t/m³</th></tr></thead>
<tbody>{prow}</tbody></table></div>
<figure>{_img_tag(figs.get('map'))}</figure>
<figure>{_img_tag(figs.get('piles'))}</figure>
<h2>How the numbers are made — and how far to trust them</h2>
<ul>
<li>{html.escape(result['model']['chain'])}; power exponent b = {_fmt(result['model']['power_exponent'], 2)}.</li>
<li>Accuracy is measured on 2025 only: satellite footprint → volume, leave-one-out over the 7 port piles,
WAPE {_fmt(acc.get('wape_pct'), 0)} %, R² {_fmt(acc.get('r2'), 2)}. The footprint classifier is trained on the 2025 UAV footprints; its port area
error, leave-one-pile-out, is {_fmt(result['model']['port_area_check'].get('wape_pct'), 0)} %. With seven piles this is an indication, not a guarantee; there is no reference for 2026.</li>
<li>What this sees: piles growing, shrinking or being cleared change their footprint. What it cannot see: a pile getting taller on the
same footprint. That needs height, i.e. stereo imagery or a UAV flight.</li>
<li>Depth Anything was tested for this and dropped: on the port its relief is flat, and erasing a pile from the image did not change
its volume. The earlier 18 % came from polygon area, not from the image.</li>
<li>Footprints are searched within 20 m of the 2025 polygons. Material placed elsewhere in the port is not counted.</li>
<li>Every scene is matched to the 2025-05-01 reference scene on unchanged pixels (R² per scene in the table); scenes with a poor
match (R² &lt; 0.6) are flagged and should be ignored.</li>
<li>A UAV flight over the port once or twice a year would re-anchor the calibration and measure the error directly.</li>
</ul>
</main></body></html>"""
    with open(path, "w") as fh:
        fh.write(doc)
    return path
