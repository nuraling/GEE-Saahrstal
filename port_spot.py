#!/usr/bin/env python3
"""Port (area 3) from SPOT: calibrate 2025, apply 2026. Needs a finished run's checkpoint + site cache.

  python port_spot.py [outdir] [site_cache]
"""
import pickle
import sys

from stockpile_engine.auth import initialize_earth_engine
from stockpile_engine.port_spot import run_spot_port

outdir = sys.argv[1] if len(sys.argv) > 1 else "outputs/saarlouis"
site_pkl = sys.argv[2] if len(sys.argv) > 2 else "outputs/cache/saarlouis_site.pkl"
initialize_earth_engine()
ck = pickle.load(open(f"{outdir}/results_checkpoint.pkl", "rb"))
site = pickle.load(open(site_pkl, "rb"))
res = run_spot_port(site, ck["ref"], f"{outdir}/port_spot")
print(res["totals"], res["chosen_chain"])
for t in res.get("transfer", []):
    print(t["sensor"], t["date"], f"{t['total_m3']:,.0f} m³", f"{t['n_piles_seen']}/{t['n_piles']} piles", f"change {t['change_pct']:+.1f}%")
