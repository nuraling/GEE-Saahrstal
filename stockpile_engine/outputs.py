"""Everything a run leaves behind: rasters, vectors, tables, maps, a report.

Local first (``outdir``); GCS upload is a separate optional step so a run
never depends on a bucket existing.
"""

from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import json
import os
from typing import Dict, List

import numpy as np
from scipy import ndimage

from .config import StockpileConfig
from .grid import Grid, crs_geom_to_lonlat, vectorize_labels

# reference palette (dataviz skill): categorical slots, status, surfaces
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"none": "#0ca30c", "watch": "#fab219", "warning": "#ec835a", "critical": "#d03b3b"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d3268"]
ORANGE_RAMP = ["#fde3d6", "#f5a27d", "#eb6834", "#b8431a", "#6e2208"]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()
                if not isinstance(v, np.ndarray) and not isinstance(v, Grid)}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_tif(path: str, arr: np.ndarray, grid: Grid, name: str) -> str:
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height, count=1,
                       dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
                       compress="deflate", tiled=True) as ds:
        ds.write(np.asarray(arr, np.float32)[:grid.height, :grid.width], 1)
        ds.set_band_description(1, name)
    return path


def write_csv(path: str, rows: List[Dict]) -> str:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    return path


def _cmap(ramp):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("r", ramp)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


DISPLAY_MAX_PX = 2000


def _step(shape) -> int:
    """Decimation for display: maps are drawn at ≤ DISPLAY_MAX_PX, not at the
    13-megapixel analysis grid (drawing at full size ran out of memory)."""
    return max(1, int(np.ceil(max(shape) / DISPLAY_MAX_PX)))


def _xy(g, k):
    xs = (g.x0 + (np.arange(0, g.width, k) + 0.5) * g.res).astype(np.float32)
    ys = (g.y1 - (np.arange(0, g.height, k) + 0.5) * g.res).astype(np.float32)
    return np.meshgrid(xs, ys)


def _rgb(site, grid):
    u = site["uav"]
    k = _step(grid.shape)
    if all(k_ in u for k_ in ("red", "green", "blue")):
        chans = []
        for b in ("red", "green", "blue"):
            a = u[b][::k, ::k].astype(np.float32)
            lo, hi = np.nanpercentile(a, [2, 98])
            chans.append(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1))
        return np.nan_to_num(np.dstack(chans))
    return None


def _pile_bbox(ref, g, margin_m=60.0):
    rows, cols = np.nonzero(ref["labels"] > 0)
    if rows.size == 0:
        return (g.x0, g.x1, g.y0, g.y1)
    return (g.x0 + cols.min() * g.res - margin_m, g.x0 + (cols.max() + 1) * g.res + margin_m,
            g.y1 - (rows.max() + 1) * g.res - margin_m, g.y1 - rows.min() * g.res + margin_m)


def map_overview(path, site, ref, therm, top_labels=15, bbox=None, title=None):
    """Pile outlines coloured by alert, zoomed to the piles. Only the largest
    piles carry a text label — 58 labels on one map is unreadable; the table
    in the report carries every pile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    g = site["grid"]
    x0, x1, y0, y1 = bbox or _pile_bbox(ref, g)
    asp = (y1 - y0) / max(x1 - x0, 1)
    fig, ax = plt.subplots(figsize=(11, min(max(11 * asp, 4), 14) + 0.6), facecolor=SURFACE)
    _background(ax, site)
    if bbox is not None:
        top_labels = 999                     # a block map labels every pile
    alerts = {p["object_id"]: p.get("alert", "none") for p in therm.get("piles", [])}
    kd = _step(g.shape)
    labels = ref["labels"][::kd, ::kd]
    X, Y = _xy(g, kd)
    size = lambda p: p.get("volume_m3") if p.get("volume_m3") is not None else (p.get("volume_satellite_estimate_m3") or 0)
    labelled = {p["pile_id"] for p in sorted(ref["piles"], key=size, reverse=True)[:top_labels]}
    labelled |= {p["pile_id"] for p in ref["piles"] if alerts.get(p["pile_id"], "none") in ("warning", "critical")}
    halo = [pe.withStroke(linewidth=3, foreground=SURFACE)]
    objs = ndimage.find_objects(labels)
    for p in ref["piles"]:
        sl = objs[p["label"] - 1] if p["label"] - 1 < len(objs) else None
        if sl is None:
            continue
        # outline from a small window around the pile: a full-grid contour per
        # pile kept 58 full-size copies alive (5 GB)
        sl = (slice(max(sl[0].start - 2, 0), sl[0].stop + 2), slice(max(sl[1].start - 2, 0), sl[1].stop + 2))
        m = labels[sl] == p["label"]
        col = STATUS.get(alerts.get(p["pile_id"], "none"), STATUS["none"])
        ax.contour(X[sl], Y[sl], m.astype(np.float32), levels=[0.5], colors=[col], linewidths=2.2)
        cy, cx = np.argwhere(m).mean(axis=0)
        cy, cx = cy + sl[0].start, cx + sl[1].start
        xy = (g.x0 + (cx * kd + .5) * g.res, g.y1 - (cy * kd + .5) * g.res)
        short = p["pile_id"].replace("Pile_", "")
        if p["pile_id"] in labelled:
            vol_txt = (f"{p['volume_m3']:,.0f} m³" if p.get("volume_m3") is not None else
                       f"≈{p['volume_satellite_estimate_m3']:,.0f} m³ sat." if p.get("volume_satellite_estimate_m3")
                       else "no data")
            ax.annotate(f"{short} · {vol_txt}", xy, ha="center", va="center", fontsize=7.5,
                        color=INK, path_effects=halo)
        else:
            ax.annotate(short, xy, ha="center", va="center", fontsize=6.5, color=INK, path_effects=halo)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2.5, label=f"thermal: {k}") for k, c in STATUS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, facecolor=SURFACE)
    ax.set_title(title or (f"Stockpile inventory — UAV volumes; outline = thermal alert "
                           f"(labels: {top_labels} largest + alerts)"), loc="left", fontsize=11, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def map_raster(path, arr, grid, title, unit, ramp, vmin=None, vmax=None, site=None, ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10 * grid.height / grid.width + 0.6), facecolor=SURFACE)
    ext = (grid.x0, grid.x1, grid.y0, grid.y1)
    if site is not None:
        rgb = _rgb(site, site["grid"])
        if rgb is not None:
            g = site["grid"]
            ax.imshow(rgb.mean(axis=2), cmap="gray", extent=(g.x0, g.x1, g.y0, g.y1), alpha=0.6)
    ka = _step(arr.shape)
    arr = arr[::ka, ::ka]
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=_cmap(ramp), extent=ext, vmin=vmin, vmax=vmax,
                   alpha=0.85, interpolation="nearest")
    if ref is not None and site is not None:
        bx0, bx1, by0, by1 = _pile_bbox(ref, site["grid"], 150.0)
        ax.set_xlim(max(bx0, ext[0]), min(bx1, ext[1])); ax.set_ylim(max(by0, ext[2]), min(by1, ext[3]))
    if ref is not None:
        g = site["grid"]
        kd = _step(g.shape)
        X, Y = _xy(g, kd)
        ax.contour(X, Y, (ref["labels"][::kd, ::kd] > 0).astype(np.float32), levels=[0.5], colors=[INK],
                   linewidths=0.8)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label(unit, color=INK2)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


METHOD_LABELS = {"original_rf": "v1: RF on bands, whole site",
                 "gbm_site": "GBM, whole site",
                 "gbm_pile": "GBM inside pile polygons (+DA)",
                 "depth_quadratic": "Depth Anything, quadratic + mean-matched"}


def chart_sensor_wape(path, study):
    """Grouped bars: bias-corrected volume WAPE per sensor and method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sensors = list(study["results"])
    methods = [m for m in METHOD_LABELS
               if any(study["results"][s]["methods"].get(m, {}).get("available") for s in sensors)]
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    _style(ax)
    w = 0.8 / max(len(methods), 1)
    for j, m in enumerate(methods):
        vals = [study["results"][s]["methods"].get(m, {}).get("pile_volume_bias_corrected", {}).get("wape_pct")
                if study["results"][s]["methods"].get(m, {}).get("available") else None for s in sensors]
        xs = np.arange(len(sensors)) + (j - (len(methods) - 1) / 2) * w
        ok = [v is not None for v in vals]
        bars = ax.bar(xs[ok], [v for v in vals if v is not None], w - 0.03, color=SERIES[j],
                      label=METHOD_LABELS[m], edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, [v for v in vals if v is not None]):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(sensors)))
    ax.set_xticklabels([f"{s}\n({study['results'][s]['grid_m']:g} m grid)" for s in sensors], fontsize=8)
    ax.set_ylabel("Pile volume WAPE % (bias-corrected)", color=INK2, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax.set_title("Satellite volume vs UAV — 5-fold spatial cross-validation", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def _scatter_panel(ax, act, raw, bc, title, top, qc=None, pc=None, pl=None):
    _style(ax)
    ax.plot([0, top], [0, top], color=INK2, lw=1, ls="--", zorder=1)
    ok = np.isfinite(act) & np.isfinite(raw)
    ax.scatter(act[ok], raw[ok], s=34, facecolor="none", edgecolor=SERIES[0], linewidth=1.5,
               label="raw", zorder=3)
    okb = np.isfinite(act) & np.isfinite(bc)
    ax.scatter(act[okb], bc[okb], s=34, color=SERIES[1], edgecolor=SURFACE, linewidth=1.2,
               label="linear-corrected", zorder=4)
    mq = {}
    if qc is not None:
        okq = np.isfinite(act) & np.isfinite(qc)
        ax.scatter(act[okq], qc[okq], s=30, marker="D", color=SERIES[2], edgecolor=SURFACE, linewidth=1.2,
                   label="quadratic-corrected", zorder=5)

    def _m(a, p):
        from .metrics import object_metrics
        return object_metrics(a, p)
    mp = {}
    if pc is not None:
        okp = np.isfinite(act) & np.isfinite(pc)
        ax.scatter(act[okp], pc[okp], s=36, marker="^", color=SERIES[6], edgecolor=SURFACE, linewidth=1.0,
                   label="power-corrected", zorder=6)
    ml = {}
    if pl is not None:
        okl = np.isfinite(act) & np.isfinite(pl)
        ax.scatter(act[okl], pl[okl], s=60, marker="*", color=SERIES[7], edgecolor=SURFACE, linewidth=0.8,
                   label="power, port-calibrated (LOO)", zorder=7)
    mr, mb = _m(act[ok], raw[ok]), _m(act[okb], bc[okb])
    if qc is not None:
        mq = _m(act[okq], qc[okq])
    if pc is not None:
        mp = _m(act[okp], pc[okp])
    if pl is not None:
        ml = _m(act[okl], pl[okl])
    r2 = lambda m: f"{m['r2']:.2f}" if m.get("r2") is not None else "–"
    ax.set_title(f"{title}\nraw WAPE {mr.get('wape_pct', 0):.0f}%  R² {r2(mr)}\n"
                 f"linear WAPE {mb.get('wape_pct', 0):.0f}%  R² {r2(mb)}"
                 + (f"\nquadratic WAPE {mq.get('wape_pct', 0):.0f}%  R² {r2(mq)}" if mq else "")
                 + (f"\npower WAPE {mp.get('wape_pct', 0):.0f}%  R² {r2(mp)}" if mp else "")
                 + (f"\npower port-cal. WAPE {ml.get('wape_pct', 0):.0f}%  bias {ml.get('bias_pct', 0):+.0f}%" if ml else "")
                 + f"  (n={mb.get('n', 0)})", fontsize=8.5, color=INK, loc="left")
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel("UAV volume (1000 m³)", fontsize=8, color=INK2)
    from matplotlib.ticker import FuncFormatter
    k = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")
    ax.xaxis.set_major_formatter(k)
    ax.yaxis.set_major_formatter(k)


def chart_volume_scatter(path, ref, study, method="gbm_pile"):
    """Validation scatter per sensor: UAV vs satellite, raw and bias-corrected."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sensors = [s for s in study["results"] if study["results"][s]["methods"].get(method, {}).get("available")]
    if not sensors:
        return None
    fig, axes = plt.subplots(1, len(sensors), figsize=(3.8 * len(sensors), 4.7), facecolor=SURFACE,
                             squeeze=False)
    act = np.array([np.nan if p["volume_m3"] is None else p["volume_m3"] for p in ref["piles"]])
    top = max(np.nanmax(act) if np.isfinite(act).any() else 1, 1) * 1.15
    for ax, s in zip(axes[0], sensors):
        raw = np.array([p.get(f"vol_{s}_{method}_m3") or np.nan for p in ref["piles"]], float)
        bc = np.array([p.get(f"vol_{s}_{method}_bc_m3") or np.nan for p in ref["piles"]], float)
        qc = np.array([p.get(f"vol_{s}_{method}_qc_m3") or np.nan for p in ref["piles"]], float)
        pc = np.array([p.get(f"vol_{s}_{method}_pc_m3") or np.nan for p in ref["piles"]], float)
        _scatter_panel(ax, act, raw, bc, s, top, qc, pc)
    axes[0][0].set_ylabel("Satellite volume (1000 m³)", fontsize=8, color=INK2)
    axes[0][0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(f"Validation — {METHOD_LABELS.get(method, method)}, out-of-fold", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_port_scatter(path, ref, study, keep=(("s2_10m", "original_rf"), ("s2_10m", "gbm_pile"),
                                                ("s2_sr_3m", "gbm_pile"), ("s2_sr_3m", "depth_quadratic"))):
    """Port hold-out (DSM_3): trained on the other blocks, tested on the port."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    for s, sr in study["results"].items():
        for m, r in (sr.get("port_holdout") or {}).items():
            if isinstance(r, dict) and r.get("n_port_piles_seen") and (s, m) in keep:
                panels.append((s, m))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 5.1), facecolor=SURFACE,
                             squeeze=False)
    port = np.array(study.get("in_port", [False] * len(ref["piles"])))
    act = np.array([np.nan if (p["volume_m3"] is None or not port[i]) else p["volume_m3"]
                    for i, p in enumerate(ref["piles"])])
    top = max(np.nanmax(act) if np.isfinite(act).any() else 1, 1) * 1.3
    for ax, (s, m) in zip(axes[0], panels):
        raw = np.array([p.get(f"port_{s}_{m}_m3") or np.nan for p in ref["piles"]], float)
        bc = np.array([p.get(f"port_{s}_{m}_bc_m3") or np.nan for p in ref["piles"]], float)
        qc = np.array([p.get(f"port_{s}_{m}_qc_m3") or np.nan for p in ref["piles"]], float)
        pc = np.array([p.get(f"port_{s}_{m}_pc_m3") or np.nan for p in ref["piles"]], float)
        pl = np.array([p.get(f"port_{s}_{m}_pl_m3") or np.nan for p in ref["piles"]], float)
        _scatter_panel(ax, act, raw, bc, f"{s} · {m}", top, qc, pc, pl)
    axes[0][0].set_ylabel("Satellite volume (1000 m³)", fontsize=8, color=INK2)
    axes[0][0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Port hold-out (DSM_3) — model never saw the port", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


ASSET_COLOURS = {"roof": SERIES[0], "chimney_stack": SERIES[1], "pipe_conveyor": SERIES[2], "building": SERIES[0]}


def map_assets(path, site, therm, bbox=None, title=None):
    """Thermal assets: outline colour = type, filled red = top 10% hottest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    labs = therm.get("building_labels")
    if labs is None or labs.max() == 0:
        return None
    g = site["grid"]
    x0, x1, y0, y1 = bbox or (g.x0, g.x1, g.y0, g.y1)
    asp = (y1 - y0) / max(x1 - x0, 1)
    fig, ax = plt.subplots(figsize=(11, min(max(11 * asp, 4), 14) + 0.6), facecolor=SURFACE)
    _background(ax, site)
    info = {b["label"]: b for b in therm["buildings"]}
    kd = _step(g.shape)
    labs_full = labs
    labs = labs[::kd, ::kd]
    top = np.isin(labs, [k for k, b in info.items() if b.get("top10pct_hottest")])
    ax.imshow(np.ma.masked_where(~top, top), cmap=_cmap(["#d03b3b", "#d03b3b"]),
              extent=(g.x0, g.x1, g.y0, g.y1), alpha=0.75, interpolation="nearest")
    X, Y = _xy(g, kd)
    for cls, col in ASSET_COLOURS.items():
        m = np.isin(labs, [k for k, b in info.items() if b.get("class") == cls])
        if m.any():
            ax.contour(X, Y, m.astype(float), levels=[0.5], colors=[col], linewidths=1.0)
    def _inside(b):
        rr, cc = np.nonzero(labs == b["label"])
        if rr.size == 0:
            return False
        x = g.x0 + (cc.mean() * kd + .5) * g.res
        y = g.y1 - (rr.mean() * kd + .5) * g.res
        return x0 <= x <= x1 and y0 <= y <= y1
    hot = sorted([b for b in therm["buildings"] if b.get("max_lst_composite_c") is not None and
                  (bbox is None or _inside(b))], key=lambda b: -b["max_lst_composite_c"])[:10]
    import matplotlib.patheffects as pe
    for b in hot:
        rr, cc = np.nonzero(labs == b["label"])
        if rr.size:
            ax.annotate(f"{b['object_id'].replace('Asset_', '#')} {b['max_lst_composite_c']:.1f}°C",
                        (g.x0 + (cc.mean() * kd + .5) * g.res, g.y1 - (rr.mean() * kd + .5) * g.res), fontsize=7,
                        color=INK, ha="center", path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])
    handles = [Line2D([0], [0], color=c, lw=2, label=k.replace("_", " ")) for k, c in ASSET_COLOURS.items()
               if k != "building"] + [Patch(color="#d03b3b", label="top 10% hottest (max LST)")]
    ax.legend(handles=handles, loc="lower right", fontsize=8, facecolor=SURFACE)
    ax.set_title(title or "Thermal assets (building footprints + UAV stacks/pipes) — top 10% hottest by max LST",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def _img_tag(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b = base64.b64encode(fh.read()).decode()
    return f'<img alt="{html.escape(os.path.basename(path))}" src="data:image/png;base64,{b}">'


def _fmt(v, nd=1):
    if v is None:
        return "–"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "–"
        return f"{v:,.{nd}f}"
    return html.escape(str(v))


def _recent_section(rec: Dict, piles: List[Dict], s: Dict, key: str, sensor: str, title: str) -> str:
    """Report table: a validated model applied to recent scenes of one sensor."""
    if not rec.get("scenes"):
        return ""
    dates = [sc["date"] for sc in rec["scenes"]]
    ref_d = rec.get("reference_date")
    rrows = ""
    for p in piles:
        if all(p.get(f"{key}_{d}_m3") is None for d in dates):
            continue
        cells = "".join(f"<td class=n>{_fmt(p.get(f'{key}_{d}_m3'), 0)}</td>" for d in dates)
        rrows += (f"<tr><td>{_fmt(p['pile_id'])}</td><td class=n>{_fmt(p['volume_m3'], 0)}</td>"
                  f"<td class=n>{_fmt(p.get(f'{key}_{ref_d}_m3'), 0)}</td>{cells}</tr>")
    tot = "".join(f"<td class=n><b>{_fmt(sc['total_m3'], 0)}</b></td>" for sc in rec["scenes"])
    if all(sc["total_m3"] is None for sc in rec["scenes"]):
        return (f"<h2>{html.escape(title)}</h2><p class=sub>No pile was imaged on both the reference and the "
                f"newer scene{'s' if len(dates) > 1 else ''} ({html.escape(', '.join(dates))}), so this model gives no "
                f"estimate for {'them' if len(dates) > 1 else 'it'}.</p>")
    rrows += (f"<tr><td><b>Total (UAV-measured piles)</b></td><td class=n><b>{_fmt(s['total_volume_m3'], 0)}</b></td>"
              f"<td class=n><b>{_fmt(rec['scenes'][0]['total_ref_m3'], 0)}</b></td>{tot}</tr>")
    rhead = ("<th>Pile</th><th>UAV 2025-05-01 m³</th><th>Reference m³</th>"
             + "".join(f"<th>{sensor} {html.escape(d)} m³</th>" for d in dates))
    rfit = "; ".join(f"{sc['date']}: R² " + ", ".join(
        f"{b} {_fmt(f.get('r2'), 2)}" for b, f in sc["radiometric_fit"].items()) for sc in rec["scenes"])
    notes = ""
    for sc in rec["scenes"]:
        if "n_piles_seen" in sc:
            notes += (f" {html.escape(sc['date'])}: {sc['n_piles_seen']} of {sc['n_piles']} piles imaged on both dates; "
                      f"totals and change are over those piles only.")
        chk = sc.get("band_check")
        if chk and not chk.get("ok"):
            notes += f" <b>Band check failed for {html.escape(sc['date'])}: {html.escape(str(chk.get('note')))}.</b>"
    return (f"<h2>{html.escape(title)}</h2>"
            f"<p class=sub>Model: {html.escape(str(rec.get('model')))}, correction factor "
            f"{_fmt(rec.get('correction_factor'), 2)} (used only for piles without UAV). Each pile now = UAV volume × (model now ÷ model on the reference scene), so the model's own bias cancels. Recent scenes are radiometrically matched to the "
            f"reference scene on unchanged pixels ({html.escape(rfit)}). Volumes are inside the 2025 pile "
            f"polygons: material moved outside them is not counted.{notes}</p>"
            f"<div class=tw><table><thead><tr>{rhead}</tr></thead><tbody>{rrows}</tbody></table></div>")


def write_html(path, report, figs):
    s = report["summary"]
    rows = "".join(
        f"<tr><td>{_fmt(p['pile_id'])}</td><td>{_fmt(p['commodity'])}</td>"
        f"<td class=n>{_fmt(p['occupied_area_m2'], 0)}</td><td class=n>{_fmt(p['max_height_m'])}</td>"
        f"<td class=n>{_fmt(p['volume_m3'], 0)} ± {_fmt(p['volume_sigma_m3'], 0)}</td>"
        f"<td class=n>{_fmt(p.get('volume_satellite_estimate_m3'), 0)}</td>"
        f"<td class=n>{_fmt(p['tonnage_t'], 0)}</td>"
        f"<td><span class='st st-{p.get('alert', 'none')}'>{p.get('alert', 'none')}</span>"
        f"<br><small>{html.escape('; '.join(p.get('alert_reason') or []))}</small></td>"
        f"<td class=n>{_fmt(p.get('max_delta_t_c'))}</td><td class=n>{_fmt(p.get('persistence'), 2)}</td></tr>"
        for p in report["piles"])
    srows = ""
    for sname, sr in report["sensor_study"].items():
        for m, mr in sr["methods"].items():
            if not mr.get("available"):
                srows += f"<tr><td>{sname}</td><td>{m}</td><td colspan=7>{_fmt(mr.get('reason'))}</td></tr>"
                continue
            v, vb, px = mr["pile_volume_known_footprint"], mr["pile_volume_bias_corrected"], mr["pixel_height_oof"]
            vq = mr.get("pile_volume_quadratic_corrected", {})
            vp = mr.get("pile_volume_power_corrected", {})
            srows += (f"<tr><td>{sname}</td><td>{html.escape(METHOD_LABELS.get(m, m))}</td>"
                      f"<td class=n>{_fmt(px.get('rmse_m'), 2)}</td>"
                      f"<td class=n>{_fmt(v.get('wape_pct'), 0)}</td><td class=n>{_fmt(v.get('r2'), 2)}</td>"
                      f"<td class=n>{_fmt(vb.get('wape_pct'), 0)}</td><td class=n>{_fmt(vb.get('r2'), 2)}</td>"
                      f"<td class=n>{_fmt(vq.get('wape_pct'), 0)}</td><td class=n>{_fmt(vq.get('r2'), 2)}</td>"
                      f"<td class=n><b>{_fmt(vp.get('wape_pct'), 0)}</b></td><td class=n>{_fmt(vp.get('r2'), 2)}</td>"
                      f"<td class=n>{_fmt(mr.get('power_exponent'), 2)}</td>"
                      f"<td class=n>{vb.get('n', 0)}</td></tr>")
    prows = ""
    for sname, sr in report["sensor_study"].items():
        ph = sr.get("port_holdout") or {}
        if "note" in ph:
            prows += f"<tr><td>{sname}</td><td colspan=7>{html.escape(ph['note'])}</td></tr>"
            continue
        if ph and all(r.get("n_port_piles_seen", 0) == 0 for r in ph.values()):
            prows += f"<tr><td>{sname}</td><td colspan=12>no imagery over the port — not evaluated</td></tr>"
            continue
        for m, r in ph.items():
            prows += (f"<tr><td>{sname}</td><td>{html.escape(METHOD_LABELS.get(m, m))}</td>"
                      f"<td class=n>{r['n_port_piles_seen']}</td>"
                      f"<td class=n>{_fmt(r['pixel_height'].get('rmse_m'), 2)}</td>"
                      f"<td class=n>{_fmt(r['raw'].get('wape_pct'), 0)}</td>"
                      f"<td class=n>{_fmt(r['bias_corrected'].get('wape_pct'), 0)}</td>"
                      f"<td class=n>{_fmt(r.get('quadratic_corrected', {}).get('wape_pct'), 0)}</td>"
                      f"<td class=n>{_fmt(r.get('power_corrected', {}).get('wape_pct'), 0)}</td>"
                      f"<td class=n>{_fmt(r.get('power_corrected', {}).get('bias_pct'), 0)}</td>"
                      f"<td class=n><b>{_fmt(r.get('power_corrected_port_loo', {}).get('wape_pct'), 0)}</b></td>"
                      f"<td class=n>{_fmt(r.get('power_corrected_port_loo', {}).get('bias_pct'), 0)}</td>"
                      f"<td class=n>{_fmt(r.get('power_corrected_port_loo', {}).get('r2'), 2)}</td>"
                      f"<td class=n>{_fmt(r['correction_factor'], 2)}</td></tr>")
    recent_html = (_recent_section(report.get("recent_sentinel") or {}, report["piles"], s, "s2", "S2",
                                   "5 · Sentinel-2 now — the validated model on recent scenes")
                   + _recent_section(report.get("recent_pleiades") or {}, report["piles"], s, "ple", "Pléiades",
                                     "5b · Pléiades now — the validated Pléiades model on newer scenes"))
    arows = ""
    for b in sorted([b for b in report.get("buildings", []) if b.get("max_lst_composite_c") is not None],
                    key=lambda b: -b["max_lst_composite_c"])[:15]:
        arows += (f"<tr><td>{_fmt(b['object_id'])}</td><td>{_fmt(b.get('class'))}</td>"
                  f"<td class=n>{_fmt(b.get('area_m2'), 0)}</td><td class=n>{_fmt(b.get('max_height_m'), 1)}</td>"
                  f"<td class=n><b>{_fmt(b.get('max_lst_composite_c'), 1)}</b></td>"
                  f"<td class=n>{_fmt(b.get('max_lst_downscaled_c'), 1)}</td>"
                  f"<td class=n>{_fmt(b.get('max_delta_t_c'), 1)}</td>"
                  f"<td class=n>{b.get('n_night_anomalous', 0)}/{b.get('n_night_scenes', 0)}</td>"
                  f"<td class=n>{_fmt(b.get('hotspot_equiv_5m_c_median'), 0)}</td>"
                  f"<td>{'yes' if b.get('top10pct_hottest') else ''}</td></tr>")
    gaps = "".join(f"<li>{html.escape(x)}</li>" for x in report["data_recommendations"])
    notes = "".join(f"<li>{html.escape(x)}</li>" for x in report["caveats"])
    def fig_html(key):
        if key.endswith("*"):
            return "".join(f"<figure>{_img_tag(f)}</figure>" for k, f in figs.items() if k.startswith(key[:-1]) and f)
        if key == "rest":
            used = {"assets", "overview", "wape", "scatter", "port", "scatter_depth"}
            if True:
                return "".join(f"<figure>{_img_tag(f)}</figure>" for k, f in figs.items()
                               if k not in used and not k.startswith("block_") and f)
            return "".join(f"<figure>{_img_tag(f)}</figure>" for k, f in figs.items() if k not in used and f)
        return f"<figure>{_img_tag(figs[key])}</figure>" if figs.get(key) else ""
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Stockpile Intelligence Report</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#383835;--card:#242423}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px 16px}}
h1{{margin:0 0 4px;font-size:24px}} h2{{margin-top:32px;font-size:18px}}
.sub{{color:var(--ink2)}} .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}}
.kpi b{{display:block;font-size:22px}} .kpi span{{color:var(--ink2);font-size:12px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.st{{padding:1px 8px;border-radius:10px;font-size:12px;color:#0b0b0b}}
.st-none{{background:#0ca30c33}}.st-watch{{background:#fab21955}}.st-warning{{background:#ec835a66}}.st-critical{{background:#d03b3b;color:#fff}}
figure{{margin:16px 0}} img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}}
</style></head><body><main>
<h1>Stockpile Intelligence — {html.escape(report['project_uuid'])}</h1>
<div class=sub>Generated {html.escape(report['generated_utc'])} · inventory source: {html.escape(s['inventory_source'])}</div>
<div class=kpis>
<div class=kpi><b>{s['n_piles']}</b><span>stockpiles ({s['n_uav_measured']} with UAV)</span></div>
<div class=kpi><b>{_fmt(s['total_volume_m3'], 0)} m³</b><span>total volume (UAV)</span></div>
<div class=kpi><b>{_fmt(s['total_tonnage_t'], 0)} t</b><span>indicative tonnage</span></div>
<div class=kpi><b>{_fmt(s['occupied_area_m2'], 0)} m²</b><span>area under material</span></div>
<div class=kpi><b>{s.get('n_thermal_assets', 0)}</b><span>thermal assets (roofs, stacks, pipes)</span></div>
<div class=kpi><b>{s['n_thermal_scenes']}</b><span>thermal scenes analysed</span></div>
</div>
<h2>1 · Thermal — building assets (roofs, chimneys/stacks, pipes/conveyors)</h2>
<p class=sub>{html.escape(report.get('asset_note', ''))}</p>
<div class=tw><table><thead><tr><th>Asset</th><th>Type</th><th>Area m²</th><th>Height m</th><th>Max LST °C (Landsat, 30 m)</th>
<th>Max LST °C (modelled 3 m)</th><th>Max ΔT vs surroundings °C</th><th>Warm nights</th><th>Equiv. 5×5 m hot-spot °C</th><th>Top 10%</th></tr></thead>
<tbody>{arows}</tbody></table></div>
{fig_html('assets')}
<h3>By area</h3>
{fig_html('block_assets_*')}
<h2>2 · Stockpile inventory (UAV reference)</h2>
{fig_html('overview')}
<h3>By area</h3>
{fig_html('block_piles_*')}
<div class=tw><table><thead><tr><th>Pile</th><th>Commodity</th><th>Area m²</th><th>Max h m</th>
<th>Volume m³ (UAV)</th><th>Satellite est. m³</th><th>Tonnage t</th><th>Pile thermal</th><th>Max ΔT °C</th><th>Persistence</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>3 · Validation — satellite volume vs UAV (5-fold spatial cross-validation)</h2>
<p class=sub>Corrections are fitted on the piles of the <i>other</i> folds only — no pile corrects itself. Linear: Σ actual / Σ predicted.
Quadratic: actual ≈ a·v + b·v². Power: actual ≈ c·v<sup>b</sup> (b = 2 is a square relation). Both are then scaled so the corrected total equals the actual total (mean matched).</p>
<div class=tw><table><thead><tr><th>Sensor</th><th>Method</th><th>Height RMSE m</th><th>WAPE % raw</th><th>R² raw</th>
<th>WAPE % linear corr.</th><th>R² linear</th><th>WAPE % quadratic corr.</th><th>R² quadratic</th>
<th>WAPE % power corr.</th><th>R² power</th><th>exponent b</th><th>n piles</th></tr></thead><tbody>{srows}</tbody></table></div>
{fig_html('wape')}{fig_html('scatter')}{fig_html('scatter_depth')}
<h2>4 · Port hold-out (DSM_3) — trained on the other survey blocks, tested on the port</h2>
<p class=sub>{html.escape(report.get('port_reference_note', ''))}</p>
<div class=tw><table><thead><tr><th>Sensor</th><th>Method</th><th>Port piles</th><th>Height RMSE m</th><th>WAPE % raw</th>
<th>WAPE % linear</th><th>WAPE % quadratic</th><th>WAPE % power</th><th>Bias % power</th>
<th>WAPE % power, port-calibrated</th><th>Bias % port-cal.</th><th>R² port-cal.</th><th>Linear factor</th></tr></thead><tbody>{prows}</tbody></table></div>
<p class=sub>"Power" is fitted on the steel-works piles only. "Power, port-calibrated" is fitted on the other port piles, leave-one-out:
each port pile is corrected by a curve that never saw it, i.e. what a few UAV-measured piles in a new area buy.</p>
{fig_html('port')}
{recent_html}
{fig_html('rest')}
<h2>What would raise accuracy</h2><ul>{gaps}</ul>
<h2>Read this before quoting a number</h2><ul>{notes}</ul>
</main></body></html>"""
    with open(path, "w") as fh:
        fh.write(doc)
    return path


def recommendations(study, therm) -> List[str]:
    rec = [
        "Pléiades (or Pléiades Neo) tri-stereo acquisition over the yard → photogrammetric DSM at "
        "0.5–1 m with ~1 m vertical accuracy. This is what turns satellite volumes from 'modelled' "
        "into 'measured'; single-image methods (spectral RF, Depth Anything) cannot recover absolute height.",
        "SPOT 6/7 stereo pair as a cheaper alternative: 1.5 m DSM, ~3 m vertical — adequate for large "
        "piles (> 5 m high), not for flat piles.",
        "A bare-ground DTM of the pads (UAV flight when yards are emptied, or from the site's as-built "
        "survey) so every later volume is surface-minus-pad, not surface-minus-interpolated-ground.",
        "Repeat UAV flights (≥ 3 dates) aligned with satellite acquisitions to train and validate "
        "across stock levels, not one snapshot.",
        "Operator weighbridge / stock-book tonnages per pile and date, and measured bulk densities "
        "per commodity — without them tonnage is volume × a textbook density.",
        "Pile polygons with commodity labels from the operator (replaces the spectral commodity guess).",
        "Night-time thermal: ECOSTRESS night passes (free, ~70 m) and, for real pile-scale detection, "
        "commercial high-resolution thermal (e.g. Hydrosat, SatVu, OroraTech: 3.5–10 m class) or a "
        "UAV radiometric thermal flight — Landsat's 100 m footprint only sees a smoulder once it is large.",
        "In-situ probe temperatures inside two or three piles (the operator's own spontaneous-"
        "combustion checks) to calibrate what a surface anomaly means for this material.",
    ]
    if therm.get("piles"):
        md = [p.get("min_detectable_5m_hotspot_c") for p in therm["piles"]
              if p.get("min_detectable_5m_hotspot_c") is not None]
        if md:
            rec.insert(6, f"With the current thermal data, a 5 × 5 m hot-spot must be about "
                          f"{np.nanmedian(md):.0f} °C before it rises above the background scatter "
                          f"(median over piles). That is the argument for higher-resolution thermal.")
    return rec


CAVEATS = [
    "UAV volumes are the reference; their ± is area × σz (5 cm) and excludes base-surface error.",
    "Every satellite score is out-of-fold under spatial block cross-validation: a pile is only "
    "predicted by a model that never saw its block. Scores on the training pixels (reported for the "
    "original RF as 'in_sample') are optimistic by construction.",
    "Tonnage uses indicative bulk densities; commodity is a colour rule unless the pile polygon "
    "carries a 'commodity' property.",
    "Sentinel-2 at 1 m or 3 m is super-resolved, not measured. It sharpens footprints; it cannot "
    "see a feature smaller than its 10 m pixels recorded.",
    "The downscaled LST map is modelled from optical bands; only the per-pile contrasts use the "
    "thermal measurement directly.",
    "Hot-spot temperatures are the temperature a 5 × 5 m source would need to explain the observed "
    "excess (Planck mixing in the 100 m / 70 m footprint) — an equivalent, not a measurement.",
]


def ground_method_check(site: Dict, ref: Dict) -> Dict:
    """Volumes with DSM-derived ground vs the UAV DTM, on piles that have both:
    how far the port's reference (derived ground) can be trusted."""
    from . import terrain
    from .metrics import object_metrics
    g = site["grid"]
    dsm, dtm = site["uav"]["dsm"], site["uav"]["dtm"]
    labels = ref["labels"]
    both = [i for i, p in enumerate(ref["piles"]) if p.get("ground_source") == "uav_dtm" and p.get("volume_m3")]
    if not both:
        return {}
    d = terrain.derive_dtm(dsm, g.res)
    h = terrain.remove_thin_structures(terrain.ndsm(dsm, d), g.res)
    vd = terrain.pile_volumes(h, labels, g)
    a = np.array([ref["piles"][i]["volume_m3"] for i in both])
    b = np.array([vd[i]["volume_m3"] for i in both])
    return object_metrics(a, b)


def _port_note(ref, gchk) -> str:
    chk = ref.get("dtm_check") or {}
    if chk.get("method") == "flat_min":
        lv = ", ".join(f"{k} {v:.2f} m" for k, v in chk.get("levels_m", {}).items())
        return (f"The port has no UAV DTM. It is flat, so its ground is one level per block: {lv} "
                f"(the {chk.get('percentile', 5):g}th percentile of the land surface; open water excluded "
                f"with Sentinel-2 NDWI — the absolute minimum is the Saar surface, ~4 m lower).")
    if gchk.get("n"):
        return (f"The port has no UAV DTM; its ground is derived from the DSM, which gives volumes "
                f"{gchk['bias_pct']:+.0f}% vs the UAV DTM on the steel-works piles (n={gchk['n']}).")
    return ""


def _hillshade(z, res, az=315.0, alt=45.0):
    gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmedian(z)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = np.radians(az), np.radians(alt)
    hs = np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def _background(ax, site):
    """Ortho where there is one, DSM hillshade elsewhere (the port has no ortho)."""
    g = site["grid"]
    k = _step(g.shape)
    ext = (g.x0, g.x1, g.y0, g.y1)
    dsm = site["uav"]["dsm"][::k, ::k]
    if np.isfinite(dsm).any():
        hs = _hillshade(dsm, g.res * k)
        ax.imshow(np.ma.masked_where(~np.isfinite(dsm), hs), cmap="gray", extent=ext, alpha=0.9)
    rgb = _rgb(site, g)
    if rgb is not None:
        has = rgb.sum(axis=2) > 0.02
        ax.imshow(np.dstack([rgb, has.astype(np.float32)]), extent=ext)


def block_bboxes(site, ref, margin_m=40.0) -> Dict[str, tuple]:
    """Map extent per UAV survey block: around that block's piles."""
    g = site["grid"]
    ids = site["uav"].get("dsm_id")
    out = {}
    if ids is None:
        return out
    for p in ref["piles"]:
        blk = p.get("survey_block")
        if not blk:
            continue
        rr, cc = np.nonzero(ref["labels"] == p["label"])
        if rr.size == 0:
            continue
        b = (g.x0 + cc.min() * g.res, g.x0 + (cc.max() + 1) * g.res,
             g.y1 - (rr.max() + 1) * g.res, g.y1 - rr.min() * g.res)
        o = out.get(blk)
        out[blk] = b if o is None else (min(o[0], b[0]), max(o[1], b[1]), min(o[2], b[2]), max(o[3], b[3]))
    return {k: (v[0] - margin_m, v[1] + margin_m, v[2] - margin_m, v[3] + margin_m) for k, v in sorted(out.items())}


BLOCK_NAMES = {"DSM_1": "north-east yard (DSM_1)", "DSM_2": "steel-works yards (DSM_2)", "DSM_3": "port (DSM_3)"}


def write_all(outdir: str, project_uuid: str, site: Dict, ref: Dict, study: Dict, therm: Dict,
              cfg=StockpileConfig, timings=None) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    rd = os.path.join(outdir, "rasters")
    fd = os.path.join(outdir, "figures")
    os.makedirs(rd, exist_ok=True)
    os.makedirs(fd, exist_ok=True)
    g: Grid = site["grid"]
    files = {"rasters": [], "vectors": [], "tables": [], "figures": []}

    files["rasters"].append(write_tif(os.path.join(rd, "ndsm_uav.tif"), ref["ndsm"], g, "height_m"))
    files["rasters"].append(write_tif(os.path.join(rd, "pile_labels.tif"), ref["labels"].astype(np.float32), g, "label"))
    for s, m in study.get("maps", {}).items():
        files["rasters"].append(write_tif(os.path.join(rd, f"height_pred_{s}.tif"), m["height_oof"], m["grid"], "height_m"))
        files["rasters"].append(write_tif(os.path.join(rd, f"pile_prob_{s}.tif"), m["prob_oof"], m["grid"], "p_pile"))
    if therm.get("composite_lst") is not None:
        files["rasters"].append(write_tif(os.path.join(rd, "lst_composite_30m.tif"), therm["composite_lst"],
                                          site["thermal_grid"], "lst_c"))
    if therm.get("downscaled_lst"):
        d = therm["downscaled_lst"]
        files["rasters"].append(write_tif(os.path.join(rd, "lst_downscaled_3m.tif"), d["lst"], d["grid"], "lst_c"))
    if therm.get("swir_hot_frequency") is not None:
        files["rasters"].append(write_tif(os.path.join(rd, "swir_hot_frequency.tif"), therm["swir_hot_frequency"],
                                          site["swir_grid"], "share_of_scenes"))

    # merge thermal into pile records
    tmap = {t["label"]: t for t in therm.get("piles", [])}
    piles = []
    for p in ref["piles"]:
        t = tmap.get(p["label"], {})
        q = dict(p)
        for k in ("n_scenes", "mean_delta_t_c", "max_delta_t_c", "surface_delta_t_equiv_c_median", "max_z", "persistence", "n_anomalous",
                  "persistent_anomaly", "delta_t_trend_c_per_month", "hotspot_equiv_5m_c_median",
                  "min_detectable_5m_hotspot_c", "swir_hot_scenes", "swir_hot_dates", "alert",
                  "alert_reason", "p_value_vs_noise", "n_night_scenes", "n_night_anomalous",
                  "day_persistence"):
            if k in t:
                q[k] = t[k]
        q.setdefault("alert", "none")
        piles.append(q)

    geoms = vectorize_labels(ref["labels"], g, simplify_m=g.res * 0.5)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": crs_geom_to_lonlat(geoms[p["label"]], g.crs),
         "properties": _jsonable({k: v for k, v in p.items() if k != "label"})}
        for p in piles if p["label"] in geoms]}
    pth = os.path.join(outdir, "piles.geojson")
    with open(pth, "w") as fh:
        json.dump(fc, fh)
    files["vectors"].append(pth)
    if therm.get("buildings") and therm.get("building_labels") is not None:
        bg = vectorize_labels(therm["building_labels"], g)
        bfc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": crs_geom_to_lonlat(bg[b["label"]], g.crs),
             "properties": _jsonable(b)} for b in therm["buildings"] if b["label"] in bg]}
        pth = os.path.join(outdir, "buildings_thermal.geojson")
        with open(pth, "w") as fh:
            json.dump(bfc, fh)
        files["vectors"].append(pth)
        files["tables"].append(write_csv(os.path.join(outdir, "buildings_thermal.csv"), _jsonable(therm["buildings"])))

    files["tables"].append(write_csv(os.path.join(outdir, "piles.csv"), _jsonable(piles)))
    srows = []
    for s, sr in study["results"].items():
        for m, mr in sr["methods"].items():
            if not mr.get("available"):
                continue
            row = {"sensor": s, "method": m, "grid_m": sr["grid_m"], "native_m": sr["native_m"],
                   "depth_anything": sr["depth_anything"]}
            for grp in ("pixel_height_oof", "pile_volume_known_footprint", "pile_volume_bias_corrected",
                        "pile_volume_satellite_only", "pile_area"):
                for k, v in mr.get(grp, {}).items():
                    row[f"{grp}.{k}"] = v
            row["footprint_iou"] = mr.get("footprint_iou")
            srows.append(row)
    if srows:
        files["tables"].append(write_csv(os.path.join(outdir, "sensor_comparison.csv"), _jsonable(srows)))
    if therm.get("per_scene"):
        names = {p["label"]: p["pile_id"] for p in piles}
        files["tables"].append(write_csv(os.path.join(outdir, "thermal_scenes.csv"),
                                         _jsonable([{"pile_id": names.get(r["label"]), **r} for r in therm["per_scene"]])))

    figs = {"assets": map_assets(os.path.join(fd, "thermal_assets.png"), site, therm),
            "overview": map_overview(os.path.join(fd, "overview.png"), site, ref, therm)}
    for blk, bb in block_bboxes(site, ref).items():
        nm = BLOCK_NAMES.get(blk, blk)
        figs[f"block_piles_{blk}"] = map_overview(os.path.join(fd, f"piles_{blk}.png"), site, ref, therm,
                                                  bbox=bb, title=f"Stockpiles — {nm}: UAV volume per pile; outline = thermal alert")
        figs[f"block_assets_{blk}"] = map_assets(os.path.join(fd, f"thermal_assets_{blk}.png"), site, therm,
                                                 bbox=(bb[0] - 150, bb[1] + 150, bb[2] - 150, bb[3] + 150),
                                                 title=f"Thermal assets — {nm}; red = top 10% hottest site-wide")
    if study["results"]:
        figs["wape"] = chart_sensor_wape(os.path.join(fd, "sensor_volume_wape.png"), study)
        figs["scatter"] = chart_volume_scatter(os.path.join(fd, "validation_scatter_cv.png"), ref, study)
        figs["scatter_depth"] = chart_volume_scatter(os.path.join(fd, "validation_scatter_depth.png"), ref, study,
                                                     method="depth_quadratic")
        figs["port"] = chart_port_scatter(os.path.join(fd, "validation_scatter_port.png"), ref, study)
    hmax = float(np.nanpercentile(ref["ndsm"], 99.5)) if np.isfinite(ref["ndsm"]).any() else 10
    figs["ndsm"] = map_raster(os.path.join(fd, "ndsm_uav.png"),
                              np.where(ref["ndsm"] > cfg.MIN_PILE_HEIGHT_M, ref["ndsm"], np.nan),
                              g, "UAV nDSM (height above ground)", "m", BLUE_RAMP, 0, hmax, site, ref)
    if therm.get("downscaled_lst"):
        d = therm["downscaled_lst"]
        lo, hi = np.nanpercentile(d["lst"], [2, 99.5])
        figs["lst"] = map_raster(os.path.join(fd, "lst_downscaled.png"), d["lst"], d["grid"],
                                 "Surface temperature, day composite (modelled to 3 m)", "°C", ORANGE_RAMP, lo, hi)
    figs = {k: v for k, v in figs.items() if v}
    files["figures"] = list(figs.values())
    gchk = ground_method_check(site, ref) if any(
        p.get("ground_source") == "derived_from_dsm" for p in ref["piles"]) else {}
    port_rows = []
    for sname, sr in study["results"].items():
        for m, r in (sr.get("port_holdout") or {}).items():
            if isinstance(r, dict) and "raw" in r:
                port_rows.append({"sensor": sname, "method": m, "n": r["n_port_piles_seen"],
                                  **{f"raw.{k}": v for k, v in r["raw"].items()},
                                  **{f"corrected.{k}": v for k, v in r["bias_corrected"].items()},
                                  "correction_factor": r["correction_factor"]})
    if port_rows:
        files["tables"].append(write_csv(os.path.join(outdir, "validation_port_holdout.csv"), _jsonable(port_rows)))

    report = {
        "project_uuid": project_uuid,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "inventory_source": ref["source"],
            "n_piles": len(piles),
            "n_uav_measured": sum(1 for p in piles if p.get("uav_measured", True)),
            "total_volume_m3": float(sum(p["volume_m3"] or 0 for p in piles)),
            "total_volume_sigma_m3": float(sum(p["volume_sigma_m3"] or 0 for p in piles)),
            "total_tonnage_t": float(sum(p["tonnage_t"] or 0 for p in piles)),
            "occupied_area_m2": float(sum(p["occupied_area_m2"] or 0 for p in piles)),
            "satellite_estimate_unmeasured_piles_m3": float(sum(
                p.get("volume_satellite_estimate_m3") or 0 for p in piles if p.get("volume_m3") is None)),
            "tonnage_by_commodity": {c: float(sum(p["tonnage_t"] or 0 for p in piles if p["commodity"] == c))
                                     for c in sorted({p["commodity"] for p in piles})},
            "n_thermal_scenes": therm.get("n_scenes", 0),
            "n_thermal_alerts": sum(1 for p in piles if p.get("alert", "none") != "none"),
            "n_thermal_assets": len(therm.get("buildings", [])),
            "building_top10pct_threshold_c": therm.get("building_p90_c"),
        },
        "piles": piles,
        "sensor_study": study["results"],
        "recent_sentinel": study.get("recent") or {},
        "recent_pleiades": study.get("recent_pleiades") or {},
        "dtm_check_port_ground": ref.get("dtm_check"),
        "ground_method_volume_check": gchk,
        "port_reference_note": _port_note(ref, gchk),
        "asset_note": (f"{len(therm.get('buildings', []))} assets "
                       f"({'client polygons' if therm.get('asset_source') == 'client_polygons' else 'detected from the UAV surface: roofs, chimneys/stacks, pipes/conveyors'}); "
                       f"{therm.get('n_scenes', 0)} thermal scenes (Landsat 8/9 + ECOSTRESS). Top 10% by max LST, as in v1. "
                       "Landsat measures at 100 m and ECOSTRESS at 70 m: a chimney is a small part of one pixel, so "
                       "max LST is the pixel it dominates; the 5×5 m hot-spot column is the temperature needed to explain the excess."),
        "sensors_skipped": study.get("skipped", {}),
        "buildings": therm.get("buildings", []),
        "data_recommendations": recommendations(study, therm),
        "caveats": CAVEATS,
        "files": files,
    }
    report = _jsonable(report)
    with open(os.path.join(outdir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    files["report_html"] = write_html(os.path.join(outdir, "report.html"), report, figs)
    report["files"]["report_html"] = files["report_html"]
    return report
