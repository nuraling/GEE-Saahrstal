#!/usr/bin/env python3
"""Pléiades 2026-08-03 (and any STOCKPILE_PLEIADES_RECENT scene): apply the
validated Pléiades model from a finished run, then rebuild the report.

  python pleiades_recent.py outputs/saarlouis outputs/cache/saarlouis_site.pkl
"""
import pickle
import sys

from stockpile_engine import outputs
from stockpile_engine.auth import initialize_earth_engine
from stockpile_engine.config import StockpileConfig
from stockpile_engine.pipeline import Telemetry, recent_pleiades
from stockpile_engine.sources_ee import fetch_pleiades_recent

outdir = sys.argv[1] if len(sys.argv) > 1 else "outputs/saarlouis"
site_pkl = sys.argv[2] if len(sys.argv) > 2 else "outputs/cache/saarlouis_site.pkl"
initialize_earth_engine()
ck = pickle.load(open(f"{outdir}/results_checkpoint.pkl", "rb"))
site = pickle.load(open(site_pkl, "rb"))
site["pleiades_recent"] = fetch_pleiades_recent(site["optical"]["pleiades"]["grid"], StockpileConfig)
ck["study"]["recent_pleiades"] = recent_pleiades(site, ck["ref"], ck["study"], StockpileConfig,
                                                 Telemetry("pleiades_recent", print))
with open(f"{outdir}/results_checkpoint.pkl", "wb") as fh:
    pickle.dump(ck, fh, protocol=4)
rep = outputs.write_all(outdir, "saarlouis", site, ck["ref"], ck["study"], ck["therm"])
for sc in ck["study"]["recent_pleiades"].get("scenes", []):
    tot = "no piles imaged on both dates" if sc["total_m3"] is None else \
        f"{sc['total_m3']:,.0f} m³ ({sc['change_pct']:+.1f}%)"
    print(sc["date"], tot, f"{sc.get('n_piles_seen')}/{sc.get('n_piles')} piles", sc.get("band_check"))
print(rep["files"]["report_html"])
