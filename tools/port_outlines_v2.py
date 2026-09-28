#!/usr/bin/env python3
"""Build the reviewed port (DSM_3) pile outlines → data/port_piles_v2.geojson.

Inputs are the caches written while reviewing (outputs/cache/port_redetect.pkl,
port_piles_v1.pkl, port_ortho_033.npy). Candidates come from
stockpile_engine.port_piles; the ACCEPT list is the visual review of every
candidate against the 2025 orthophoto at 0.33 m (geo-solain, 2026-09-28).
Three candidates merged a heap with a hall roof; the roof is cut out where the
orthophoto is smooth and the UAV surface is smooth (roof planes, not heaps).

  python tools/port_outlines_v2.py
"""
import json
import pickle
import sys

import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from scipy import ndimage

sys.path.insert(0, ".")
from stockpile_engine import terrain  # noqa: E402

PIECE_ROUGH_M = 0.12     # small (< 300 m²) pieces left after cutting a roof must be heap-rough
ROOF_ROUGH_M = 0.07      # roofs 0.05–0.06 m; flat-topped scrap 0.08–0.09 m
GROUND_M = 179.35          # flat level used for DSM_3 (lowest land level, water excluded)

# visual review: candidate id → material. Everything else was rejected
# (buildings, sheds, silo, trees, wagons, a coal train, containers, coils, trucks, cranes).
ACCEPT = {
    "P01": "bulk_grey", "P02": "coal", "P03": "coal", "P07": "coal", "P16": "light_bulk",
    "P17": "coal", "P21": "coal", "P23": "coal", "P25": "coal", "P29": "coal", "P33": "coal",
    "P04": "scrap_metal", "P05": "scrap_metal", "P06": "scrap_metal", "P12": "scrap_metal",
    "P18": "scrap_metal", "P19": "scrap_metal", "P22": "scrap_metal", "P31": "scrap_metal",
    "P41": "scrap_metal", "P47": "scrap_metal",
}
SPLIT = ("P04", "P05", "P06")
OLD_NAMES = {"P02": "Pile_37", "P01": "Pile_38", "P07": "Pile_39", "P03": "Pile_40"}


def split_roof(m, tex, dsm):
    rough = np.abs(dsm - ndimage.median_filter(dsm, size=5))
    low = ndimage.binary_opening(m & (tex < 0.09), iterations=1)
    cl, n = ndimage.label(low)
    roof = np.zeros_like(m)
    for k in range(1, n + 1):
        c = cl == k
        if c.sum() >= 80 and np.median(rough[c]) < ROOF_ROUGH_M:
            roof |= c
    roof = ndimage.binary_closing(roof, iterations=3)
    keep = ndimage.binary_opening(m & ~ndimage.binary_dilation(roof, iterations=3), iterations=3)
    cl2, n2 = ndimage.label(keep)
    sz = ndimage.sum(np.ones_like(cl2), cl2, range(1, n2 + 1))
    keep_ids = [i + 1 for i, s in enumerate(sz)
                if s >= 60 and np.median(rough[cl2 == i + 1]) >= (PIECE_ROUGH_M if s < 300 else ROOF_ROUGH_M)]
    # small leftovers along a cut roof (edges, lean-tos: 0.08 m) must be heap-rough; large heaps
    # only have to be rougher than a roof (white scrap is fairly smooth)
    return np.isin(cl2, keep_ids), roof


def main():
    P = pickle.load(open("outputs/cache/port_redetect.pkl", "rb"))
    res = pickle.load(open("outputs/cache/port_piles_v1.pkl", "rb"))
    tex = pickle.load(open("outputs/cache/port_splits.pkl", "rb"))["tex"]
    g, lab = P["grid"], res["labels"]
    dsm = np.nan_to_num(P["dsm"], nan=GROUND_M)
    by = {x["pid"]: x for x in res["objects"] if x.get("pid")}
    order = sorted(ACCEPT, key=lambda p: -by[p]["area_m2"])
    final = np.zeros_like(lab, dtype=np.int32)
    meta = []
    for i, pid in enumerate(order, 1):
        m = lab == by[pid]["label"]
        if pid in SPLIT:
            m, _ = split_roof(m, tex, dsm)
        final[m & (final == 0)] = i
        meta.append({"name": f"Port_{i:02d}", "candidate": pid, "material": ACCEPT[pid],
                     "old_name": OLD_NAMES.get(pid), "roof_removed": pid in SPLIT})
    h_flat = P["dsm"] - GROUND_M
    v_flat = terrain.pile_volumes(h_flat, final, g)
    v_toe = terrain.pile_volumes(h_flat, final, g, P["dsm"], "toe")
    for m_, a, b in zip(meta, v_flat, v_toe):
        m_.update({"area_m2": a["area_m2"], "volume_toe_m3": b["volume_m3"], "volume_flat_m3": a["volume_m3"],
                   "max_height_toe_m": b["max_height_m"], "volume_sigma_m3": b["volume_sigma_m3"]})
    tr = from_origin(g.x0, g.y1, g.res, g.res)
    feats = []
    for geom, val in features.shapes(final, mask=final > 0, transform=tr):
        m_ = meta[int(val) - 1]
        feats.append({"type": "Feature", "geometry": transform_geom(g.crs, "EPSG:4326", geom),
                      "properties": {**m_, "commodity": {"scrap_metal": "scrap_metal", "coal": "coal"}.get(m_["material"], "unknown"),
                                     "source": "uav_dsm_auto + visual review (ortho 0.33 m), geo-solain 2026-09-28"}})
    json.dump({"type": "FeatureCollection", "features": feats,
               "properties": {"ground_m": GROUND_M, "volume_base": "toe plane per pile (volume_flat_m3: flat ground)"}},
              open("data/port_piles_v2.geojson", "w"), indent=1)
    pickle.dump({"labels": final, "meta": meta, "grid": g}, open("outputs/cache/port_piles_v2.pkl", "wb"))
    for m_ in meta:
        print(f"{m_['name']} {m_['candidate']:4s} {m_['material']:12s} {m_['area_m2']:7.0f} m²  toe {m_['volume_toe_m3']:8.0f}  flat {m_['volume_flat_m3']:8.0f}  {m_['old_name'] or ''}")
    for mat in sorted({m_['material'] for m_ in meta}):
        s = [m_ for m_ in meta if m_["material"] == mat]
        print(f"TOTAL {mat:12s} n={len(s):2d} toe {sum(x['volume_toe_m3'] for x in s):9.0f}  flat {sum(x['volume_flat_m3'] for x in s):9.0f}")
    print(f"TOTAL all          n={len(meta)} toe {sum(x['volume_toe_m3'] for x in meta):9.0f}  flat {sum(x['volume_flat_m3'] for x in meta):9.0f}")


if __name__ == "__main__":
    main()
