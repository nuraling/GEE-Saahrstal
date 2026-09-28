#!/usr/bin/env python3
"""Blocks 1–2 (UAV DTM available): audit the hand-drawn outlines and find piles
they miss. Writes contact sheets for visual review (../review/site_*.png) and
outputs/cache/site_review.pkl.

  python tools/site_outlines_review.py
"""
import pickle
import sys

import ee
import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from stockpile_engine import port_piles, terrain  # noqa: E402
from stockpile_engine.auth import initialize_earth_engine  # noqa: E402
from stockpile_engine.config import StockpileConfig as C  # noqa: E402
from stockpile_engine.grid import Grid  # noqa: E402
from stockpile_engine.sources_ee import fetch  # noqa: E402

MIN_HEIGHT_M = 1.0       # above the ground surface
MAX_OVERLAP = 0.3        # a candidate overlapping an existing outline by more is not "new"
CROP_RES = 0.4


def main():
    initialize_earth_engine(log=lambda *a: None)
    site = pickle.load(open("outputs/cache/saarlouis_site.pkl", "rb"))
    ck = pickle.load(open("outputs/saarlouis/results_checkpoint.pkl", "rb"))
    g, u, ref = site["grid"], site["uav"], ck["ref"]
    blocks = np.isin(u["dsm_id"], [1, 2])
    dsm = np.where(blocks, u["dsm"], np.nan)
    derived = terrain.derive_dtm(dsm, g.res)
    ground = np.where(np.isfinite(u["dtm"]), np.fmin(u["dtm"], derived), derived)
    ortho = {k: u[k] for k in ("red", "green", "blue")}
    ndvi = ref.get("ndvi")
    det = port_piles.detect(dsm, ground, g, ortho, ndvi, deck_m=MIN_HEIGHT_M)

    old = ref["labels"]
    h = dsm - ground
    slope = terrain.slope_deg(np.where(np.isfinite(dsm), dsm, np.nanmedian(dsm)), g.res)
    audit = []
    for p in ref["piles"]:
        if p.get("survey_block") not in ("DSM_1", "DSM_2"):
            continue
        m = old == p["label"]
        hv = h[m & np.isfinite(h)]
        if hv.size == 0:
            continue
        top = np.nanpercentile(hv, 95)
        edge = m & ~ndimage.binary_erosion(m, iterations=2)
        audit.append({"pile_id": p["pile_id"], "label": p["label"], "area_m2": float(m.sum() * g.pixel_area),
                      "flat_top_pct": float(np.mean(hv > top - 0.5) * 100),
                      "steep_edge_pct": float(np.mean(slope[edge] > 60) * 100),
                      "median_slope": float(np.median(slope[m])), "hmax": float(hv.max())})

    lab = det["labels"]
    new = []
    for o in det["objects"]:
        if o["class"] != "pile":
            continue
        m = lab == o["label"]
        ov = float((m & (old > 0)).sum() / m.sum())
        if ov <= MAX_OVERLAP:
            new.append({**o, "overlap_old": ov})
    new.sort(key=lambda o: -o["area_m2"])
    for i, o in enumerate(new, 1):
        o["cid"] = f"N{i:03d}"

    # contact sheets from the orthophoto at 0.4 m, one EE call per candidate
    img = ee.Image(C.ASSET_ROOT + "/Saarlouis/Ortho_010525").select([0, 1, 2], ["r", "g", "b"])
    per = 16
    for s in range(0, len(new), per):
        fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=70)
        for ax, o in zip(axes.ravel(), new[s:s + per]):
            m = lab == o["label"]
            rr, cc = np.nonzero(m)
            r0, r1 = max(rr.min() - 15, 0), min(rr.max() + 15, m.shape[0])
            c0, c1 = max(cc.min() - 15, 0), min(cc.max() + 15, m.shape[1])
            f = int(round(g.res / CROP_RES))
            gc = Grid(float(g.x0 + c0 * g.res), float(g.y1 - r0 * g.res), CROP_RES, int((c1 - c0) * f), int((r1 - r0) * f), g.crs)
            px = fetch(img, gc, ["r", "g", "b"], log=lambda *a: None)
            rgb = np.dstack([px["r"], px["g"], px["b"]]).astype(np.float32)
            rgb = np.clip(np.nan_to_num(rgb) / 255.0, 0, 1)
            e = np.kron((m & ~ndimage.binary_erosion(m))[r0:r1, c0:c1], np.ones((f, f), bool))
            e = e[:rgb.shape[0], :rgb.shape[1]]
            rgb[e] = (1, 0, 1)
            ax.imshow(rgb)
            ax.set_title(f"{o['cid']} {o['material']} {o['area_m2']:.0f}m² h{o['max_height_m']:.0f} r{o['roughness_m']:.1f}", fontsize=12)
            ax.axis("off")
        for ax in axes.ravel()[len(new[s:s + per]):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"../review/site_sheet_{s // per + 1}.png")
        plt.close()

    pickle.dump({"det": det, "ground": ground, "audit": audit, "new": new}, open("outputs/cache/site_review.pkl", "wb"))
    print("audit of hand-drawn outlines (flagged = steep edges ≥ 8 % and flat top ≥ 35 %):")
    for a in sorted(audit, key=lambda a: -a["steep_edge_pct"]):
        flag = "FLAG" if a["steep_edge_pct"] >= 8 and a["flat_top_pct"] >= 35 else ""
        print(f"  {a['pile_id']} area {a['area_m2']:6.0f} flat {a['flat_top_pct']:3.0f}% steep {a['steep_edge_pct']:3.0f}% slope {a['median_slope']:3.0f} hmax {a['hmax']:4.1f} {flag}")
    print(len(new), "new candidates;", sum(o["area_m2"] for o in new), "m²; sheets:", (len(new) + per - 1) // per)


if __name__ == "__main__":
    main()
