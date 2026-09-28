#!/usr/bin/env python3
"""Port stock over a period from Sentinel-2, calibrated on the 2025 UAV survey.

  python port_monitor.py [start] [end]      (default: 1 Jan this year → today)
Needs a finished run: outputs/saarlouis/results_checkpoint.pkl and the site cache.
"""
import datetime as dt
import pickle
import sys

from stockpile_engine.auth import initialize_earth_engine
from stockpile_engine.port_monitor import run_port_monitor

today = dt.date.today()
start = sys.argv[1] if len(sys.argv) > 1 else f"{today.year}-01-01"
end = sys.argv[2] if len(sys.argv) > 2 else today.isoformat()
initialize_earth_engine()
ck = pickle.load(open("outputs/saarlouis/results_checkpoint.pkl", "rb"))
site = pickle.load(open("outputs/cache/saarlouis_site.pkl", "rb"))
res = run_port_monitor(site, ck["ref"], f"outputs/saarlouis/port_{start[:4]}", start, end)
print(f"{res['n_scenes']} scenes; report: outputs/saarlouis/port_{start[:4]}/port_report.html")
