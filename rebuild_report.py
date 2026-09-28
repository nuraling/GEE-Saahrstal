#!/usr/bin/env python3
"""Rebuild outputs from a run's checkpoint without re-modelling.

  python rebuild_report.py outputs/saarlouis outputs/cache/saarlouis_site.pkl
"""
import pickle
import sys

from stockpile_engine import outputs

outdir, site_pkl = sys.argv[1], sys.argv[2]
ck = pickle.load(open(f"{outdir}/results_checkpoint.pkl", "rb"))
site = pickle.load(open(site_pkl, "rb"))
rep = outputs.write_all(outdir, sys.argv[3] if len(sys.argv) > 3 else "saarlouis", site,
                        ck["ref"], ck["study"], ck["therm"])
print(rep["files"]["report_html"])
