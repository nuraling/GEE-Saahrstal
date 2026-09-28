#!/usr/bin/env python3
"""Re-run only the thermal stage on a finished run, then rebuild the report.

  python rerun_thermal.py outputs/saarlouis outputs/cache/saarlouis_site.pkl [buildings.geojson]
"""
import json
import pickle
import sys

from stockpile_engine import outputs
from stockpile_engine.config import StockpileConfig
from stockpile_engine.pipeline import Telemetry, thermal_stage

outdir, site_pkl = sys.argv[1], sys.argv[2]
bfc = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
ck = pickle.load(open(f"{outdir}/results_checkpoint.pkl", "rb"))
site = pickle.load(open(site_pkl, "rb"))
tel = Telemetry("saarlouis")
ck["therm"] = thermal_stage(site, ck["ref"], bfc, StockpileConfig, tel)
pickle.dump(ck, open(f"{outdir}/results_checkpoint.pkl", "wb"), protocol=4)
rep = outputs.write_all(outdir, "saarlouis", site, ck["ref"], ck["study"], ck["therm"])
print(rep["files"]["report_html"])
