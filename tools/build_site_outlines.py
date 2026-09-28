#!/usr/bin/env python3
"""Combine the reviewed outlines into data/saarlouis_stockpiles_v2.geojson.

  blocks 1–2 : the client outlines outside the port (audited: none is a roof)
               + new piles from tools/site_outlines_review.py that passed the
               visual review (ACCEPT_NEW; candidates < MIN_NEW_AREA_M2 not reviewed)
  port (DSM_3): data/port_piles_v2.geojson (tools/port_outlines_v2.py)

  python tools/build_site_outlines.py
"""
import json
import pickle
import sys

import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

sys.path.insert(0, ".")

MIN_NEW_AREA_M2 = 120.0
# visual review against the 2025 orthophoto at 0.4 m (geo-solain, 2026-09-28).
# Rejected: merged plant blobs, buildings, trees/vegetated banks, conveyors,
# stackers, trains/rails, bunker walls, vehicles.
ACCEPT_NEW = ["N003", "N004", "N005", "N007", "N010", "N013", "N019", "N020", "N022", "N023",
              "N026", "N028", "N029", "N030", "N032", "N037", "N038", "N039", "N046", "N048",
              "N051", "N052", "N053", "N055", "N057", "N059", "N064", "N066", "N067", "N069",
              "N074", "N075", "N080"]
# block 1 (DSM_1) is a lime works: its light heaps are limestone (indicative).
LIGHT_IN_BLOCK1 = "limestone"
DARK = "coal"


def main():
    rv = pickle.load(open("outputs/cache/site_review.pkl", "rb"))
    site = pickle.load(open("outputs/cache/saarlouis_site.pkl", "rb"))
    g, dsm_id = site["grid"], site["uav"]["dsm_id"]
    old = json.load(open("data/saarlouis_stockpiles.geojson"))["features"]
    port = json.load(open("data/port_piles_v2.geojson"))["features"]
    port_ids = {"Pile_%02d" % i for i in range(37, 44)}

    feats = []
    for f in old:
        if f["properties"]["name"] in port_ids:
            continue
        feats.append({"type": "Feature", "geometry": f["geometry"],
                      "properties": {"name": f["properties"]["name"], "source": "client outline (audited)"}})

    lab = rv["det"]["labels"]
    tr = from_origin(g.x0, g.y1, g.res, g.res)
    new = {o["cid"]: o for o in rv["new"]}
    n_added = 0
    for cid in ACCEPT_NEW:
        o = new[cid]
        if o["area_m2"] < MIN_NEW_AREA_M2:
            continue
        m = lab == o["label"]
        block = int(np.median(dsm_id[m]))
        commodity = None
        if o["material"] == "light_bulk" and block == 1:
            commodity = LIGHT_IN_BLOCK1
        elif o["material"] == "dark_bulk":
            commodity = DARK
        geoms = [geom for geom, v in features.shapes(m.astype(np.uint8), mask=m, transform=tr) if v == 1]
        geom = max(geoms, key=lambda gm: len(gm["coordinates"][0]))
        props = {"name": f"New_{cid[1:]}", "source": "uav auto-detection + visual review", "review_id": cid}
        if commodity:
            props["commodity"] = commodity
        feats.append({"type": "Feature", "geometry": transform_geom(g.crs, "EPSG:4326", geom), "properties": props})
        n_added += 1

    # a port pile cut into several parts (roof removed between them) → Port_05a, Port_05b, …
    by_name = {}
    for f in port:
        by_name.setdefault(f["properties"]["name"], []).append(f)
    for name, fs in by_name.items():
        fs = sorted(fs, key=lambda f: -len(f["geometry"]["coordinates"][0]))
        for j, f in enumerate(fs):
            p = f["properties"]
            feats.append({"type": "Feature", "geometry": f["geometry"],
                          "properties": {"name": name + ("abcdefgh"[j] if len(fs) > 1 else ""),
                                         "commodity": p["commodity"] if p["commodity"] != "unknown" else None,
                                         "material_review": p["material"], "old_name": p.get("old_name"),
                                         "source": "port rebuild: uav + visual review"}})
    for f in feats:
        f["properties"] = {k: v for k, v in f["properties"].items() if v is not None}
    json.dump({"type": "FeatureCollection", "features": feats}, open("data/saarlouis_stockpiles_v2.geojson", "w"))
    n_port = sum(len(v) for v in by_name.values())
    print(f"{len(feats)} outlines: {len(feats) - n_added - n_port} client (blocks 1–2), {n_added} new, {n_port} port polygons ({len(by_name)} piles)")


if __name__ == "__main__":
    main()
