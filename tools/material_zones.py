#!/usr/bin/env python3
"""Site-wide material split (blocks 1–2) from material zones + orthophoto colour.

Rifky's cues (2026-10-01, from the 2025-05-01 drone survey):
  * iron ore piles are brown / rust-coloured; dark black open piles are coal;
  * the yard is laid out in separate zones by material (coal, ore, lime),
    not mixed piles — so the zone is the primary label, colour the check.

Zones (training polygons) = the union of the member pile outlines, buffered by
ZONE_BUFFER_M, written to data/material_zones.geojson. Membership was set by
visual review of the 2025 orthophoto at 1 m (geo-solain, 2026-10-01).
The zone labels the pile; where the colour disagrees, the colour wins:
light = limestone, dark = coal, brown = ore (Rifky 2026-10-01) and the pile carries `material_check`. Slag zone and the
skipped EAF construction area (Piles 54–58) are also Rifky's calls.

Colour features (median over the upper half of each pile, to avoid ground and
toe): brightness = mean(R,G,B); redness = (R − B) / (R + G + B). Coal is blue-
grey (redness ≤ −0.08), ore rust (≥ −0.03), lime bright (> 140).
Check: leave-one-pile-out nearest-centroid on the two standardised features,
compared with the zone label.

Port piles (DSM_3) keep their 2026-09-28 visual-review labels.

  PYTHONPATH=. python tools/material_zones.py
"""
import json
import pickle
import sys

import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

sys.path.insert(0, ".")

ZONE_BUFFER_M = 15.0
REVIEW = "zone + ortho colour, Rifky cue 2026-10-01 (geo-solain)"
ZONES = {
    # rust-coloured bedding yard north of the rail lines, and the two long brown windrows
    "ore_yard":     ("iron_ore",  ["Pile_44", "Pile_45", "Pile_46", "Pile_47", "Pile_48", "Pile_49",
                                   "Pile_50", "Pile_51", "Pile_52"]),
    # (Pile_15, a small brown fibrous heap in the west yard, is not in it: unzoned)
    "ore_windrows": ("iron_ore",  ["Pile_16", "Pile_17", "Pile_19"]),
    # dark open stockpiles: main coal yard, west yard, south windrows, Pile_53
    "coal_yard":    ("coal",      ["Pile_01", "Pile_02", "Pile_03", "Pile_04", "Pile_05", "Pile_06",
                                   "Pile_07", "Pile_08", "Pile_09", "Pile_10", "Pile_11", "Pile_12",
                                   "Pile_13", "Pile_14", "Pile_18", "Pile_20", "Pile_21", "Pile_53",
                                   "New_019", "New_046", "New_048", "New_053"]),
    # light-grey heaps east of the ore yard, by the quarry: Rifky 2026-10-01 "could be furnace slag"
    # (+ Pile_15, the small brown heap in the west yard)
    "slag_area":    ("slag",      ["Pile_22", "New_003", "New_005", "New_010", "New_032", "New_069",
                                   "Pile_15"]),
    # block 1 (DSM_1) is the lime works: every pile in it
    "lime_works":   ("limestone", None),
}
# new EAF construction site (Rifky 2026-10-01): skip, not stock
NO_COLOUR_CHECK = {"slag_area"}
# piles whose median colour misleads (patchy): visual call on the 1 m ortho, geo-solain 2026-10-01
VISUAL_OVERRIDE = {"New_020": "coal"}    # dark grey-black heap with light patches (median br 104)
SKIP = {"Pile_54", "Pile_55", "Pile_56", "Pile_57", "Pile_58"}
DARK_MAX = 75.0           # brightness: coal below, others above (coal 31–75, ore 76–115)
LIGHT_MIN = 140.0         # brightness: limestone above
RUST_MIN = -0.05          # redness: ore above, coal below


def colour_class(bright, red):
    if not np.isfinite(bright):
        return None
    if bright >= LIGHT_MIN:
        return "limestone"
    return "iron_ore" if red >= RUST_MIN else "coal"


def main():
    site = pickle.load(open("outputs/cache/saarlouis_site.pkl", "rb"))
    g, u = site["grid"], site["uav"]
    tr = from_origin(g.x0, g.y1, g.res, g.res)
    path = "data/saarlouis_stockpiles_v2.geojson"
    fc = json.load(open(path))
    feats = fc["features"]

    geoms = [transform_geom("EPSG:4326", g.crs, f["geometry"]) for f in feats]
    lab = features.rasterize([(gm, i + 1) for i, gm in enumerate(geoms)], out_shape=(g.height, g.width),
                             transform=tr, fill=0, dtype="int32")
    rgb = np.dstack([u["red"], u["green"], u["blue"]])
    h = u["dsm"] - u["dtm"]

    rows = []
    for i, f in enumerate(feats):
        name = f["properties"]["name"]
        m = (lab == i + 1) & np.isfinite(rgb).all(2) & np.isfinite(h)
        block = int(np.median(u["dsm_id"][lab == i + 1])) if (lab == i + 1).any() else 0
        bright = red = np.nan
        if m.sum() >= 20:
            top = m.copy()
            top[m] = h[m] >= np.median(h[m])
            r, gr, b = (float(np.median(rgb[..., k][top])) for k in range(3))
            bright, red = (r + gr + b) / 3, (r - b) / (r + gr + b)
        zone = next((z for z, (_, mem) in ZONES.items() if mem and name in mem), None)
        if zone is None and block == 1:
            zone = "lime_works"
        rows.append({"i": i, "name": name, "block": block, "zone": zone, "bright": bright, "red": red,
                     "colour": colour_class(bright, red)})

    # zone polygons (training polygons) from member outlines
    zf = []
    for z, (mat, _) in ZONES.items():
        mem = [shape(geoms[r["i"]]) for r in rows if r["zone"] == z]
        poly = unary_union([p.buffer(ZONE_BUFFER_M) for p in mem])
        zf.append({"type": "Feature", "geometry": transform_geom(g.crs, "EPSG:4326", mapping(poly)),
                   "properties": {"zone": z, "material": mat, "n_piles": len(mem), "basis": REVIEW}})
    json.dump({"type": "FeatureCollection", "features": zf}, open("data/material_zones.geojson", "w"))

    # independent colour check: leave-one-out nearest centroid vs zone label
    # slag has no colour class (Rifky's call, not a colour rule): left out of the check
    lr = [r for r in rows if r["zone"] and r["zone"] not in NO_COLOUR_CHECK and np.isfinite(r["bright"])]
    X = np.array([[r["bright"], r["red"]] for r in lr])
    X = (X - X.mean(0)) / X.std(0)
    y = np.array([ZONES[r["zone"]][0] for r in lr])
    hit = 0
    for k in range(len(lr)):
        keep = np.arange(len(lr)) != k
        cents = {c: X[keep & (y == c)].mean(0) for c in set(y[keep])}
        pred = min(cents, key=lambda c: np.sum((X[k] - cents[c]) ** 2))
        lr[k]["loo"] = pred
        hit += pred == y[k]
    print(f"colour vs zone, leave-one-out nearest centroid: {hit}/{len(lr)} agree")

    # write labels into the outlines (blocks 1–2 only; port untouched)
    flags = []
    for r in rows:
        p = feats[r["i"]]["properties"]
        if r["block"] == 3 or p["name"].startswith("Port_"):
            continue
        zmat = ZONES[r["zone"]][0] if r["zone"] else None
        p.pop("material_check", None)
        if zmat and r["zone"] not in NO_COLOUR_CHECK and r["colour"] and r["colour"] != zmat:
            flags.append(r["name"])
            # Rifky 2026-10-01: where they disagree, go by colour —
            # light = limestone, dark = coal, brown = ore
            mat = VISUAL_OVERRIDE.get(r["name"], r["colour"])
            basis = f"colour ({mat}) over zone {r['zone']} (Rifky: light lime, dark coal, brown ore)"
            p["material_check"] = "colour and zone disagree; colour used"
        elif zmat:
            mat, basis = zmat, f"zone {r['zone']}"
        elif r["name"] in SKIP:
            mat, basis = None, "skipped: new EAF construction area"
        else:
            mat, basis = None, "no zone (outside the coal/ore/lime yards)" if r["colour"] else "no orthophoto"
        p.pop("commodity", None)
        if mat:
            p["commodity"] = mat
        p["material_zone"] = r["zone"]
        p["material_basis"] = f"{basis}; {REVIEW}"
        if np.isfinite(r["bright"]):
            p["ortho_brightness"] = round(r["bright"], 1)
            p["ortho_redness"] = round(r["red"], 3)
    json.dump(fc, open(path, "w"))

    for r in sorted(rows, key=lambda r: (str(r["zone"]), r["name"])):
        if r["block"] == 3:
            continue
        print(f"{r['name']:9s} b{r['block']} {str(r['zone']):13s} colour {str(r['colour']):10s} "
              f"loo {r.get('loo', '-'):10s} br {r['bright']:6.1f} red {r['red']:+.3f}")
    print("flagged (colour ≠ zone):", flags)
    json.dump(rows, open("outputs/cache/material_zones_rows.json", "w"), default=float)


if __name__ == "__main__":
    main()
