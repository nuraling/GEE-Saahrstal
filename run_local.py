#!/usr/bin/env python3
"""Run the stockpile pipeline from the command line.

  python run_local.py --synthetic                      # no credentials needed
  python run_local.py --ee --stockpiles piles.geojson  # real Saarlouis run
      [--buildings buildings.geojson] [--thermal-start 2025-03-01 --thermal-end 2025-08-31]
      [--sensors pleiades,s2_10m,s2_sr_3m] [--out outputs/saarlouis]
"""

import argparse
import json
import os
import sys

from stockpile_engine.config import StockpileConfig
from stockpile_engine.pipeline import run_pipeline


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--ee", action="store_true")
    ap.add_argument("--project", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stockpiles", help="GeoJSON of pile polygons (lon/lat)")
    ap.add_argument("--buildings", help="GeoJSON of building polygons (lon/lat)")
    ap.add_argument("--aoi", help="GeoJSON AOI to clip the UAV footprint to")
    ap.add_argument("--sensors", default=",".join(StockpileConfig.SENSORS))
    ap.add_argument("--thermal-start")
    ap.add_argument("--thermal-end")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--cache", help="pickle of the fetched site: written after an --ee fetch, "
                                    "reused if it exists (skips the Earth Engine download)")
    a = ap.parse_args(argv)

    if a.no_depth:
        StockpileConfig.ENABLE_DEPTH = False
    load = lambda p: json.load(open(p)) if p else None
    sensors = [s.strip() for s in a.sensors.split(",") if s.strip()]

    if a.synthetic:
        from stockpile_engine.synthetic import make_site, s2_from_pleiades
        site = make_site()
        site["s2_raw"] = s2_from_pleiades(site)
        project = a.project or "synthetic_demo"
    else:
        import pickle
        if a.cache and os.path.exists(a.cache):
            print(f"Using cached site {a.cache}")
            site = pickle.load(open(a.cache, "rb"))
        else:
            from stockpile_engine.auth import initialize_earth_engine
            from stockpile_engine.sources_ee import load_site
            initialize_earth_engine()
            aoi = load(a.aoi)
            site = load_site(aoi_geojson=(aoi or {}).get("geometry", aoi) if aoi else None,
                             thermal_start=a.thermal_start, thermal_end=a.thermal_end, sensors=sensors)
            if a.cache:
                os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
                pickle.dump(site, open(a.cache, "wb"))
        project = a.project or "saarlouis"
    out = a.out or os.path.join("outputs", project)
    big = site["grid"].width * site["grid"].height > 5e6 and a.cache
    report = run_pipeline(site, project, out, load(a.stockpiles), load(a.buildings), sensors,
                          defer_outputs=bool(big))
    if report.get("deferred"):
        import subprocess
        del site
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "rebuild_report.py"), out, a.cache, project], check=True)
        report = json.load(open(os.path.join(out, "report.json")))
    s = report["summary"]
    print(json.dumps(s, indent=2))
    print(f"\nReport: {os.path.join(out, 'report.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
