"""The full chain on the synthetic yard: every stage runs and the reference
numbers are right. (Satellite scores on synthetic data prove the plumbing,
not the sensors.)"""
import json
import os

import numpy as np

from stockpile_engine.config import StockpileConfig
from stockpile_engine.pipeline import run_pipeline
from stockpile_engine.synthetic import make_site, s2_from_pleiades


def test_end_to_end(tmp_path):
    StockpileConfig.N_TRAIN_PX, old_n = 40_000, StockpileConfig.N_TRAIN_PX
    StockpileConfig.GBM_MAX_ITER, old_it = 80, StockpileConfig.GBM_MAX_ITER
    try:
        site = make_site()
        site["s2_raw"] = s2_from_pleiades(site)
        # a newer Pléiades scene: other radiometry, and the west half not imaged
        ple = site["optical"]["pleiades"]
        newb = {k: (1.1 * v).astype(np.float32) for k, v in ple["bands"].items()}
        for v in newb.values():
            v[:, : v.shape[1] // 2] = np.nan
        site["pleiades_recent"] = [{"grid": ple["grid"], "date": "2026-08-03", "bands": newb}]
        rep = run_pipeline(site, "e2e", str(tmp_path), sensors=["pleiades", "s2_10m", "spot"],
                           log=lambda *_: None)
    finally:
        StockpileConfig.N_TRAIN_PX, StockpileConfig.GBM_MAX_ITER = old_n, old_it

    s = rep["summary"]
    assert s["n_piles"] == 8
    truth = site["truth"]["object_height"]
    assert s["total_volume_m3"] > 0.9 * 55000
    assert "spot" in rep["sensors_skipped"]
    for sensor in ("pleiades", "s2_10m"):
        m = rep["sensor_study"][sensor]["methods"]
        assert m["gbm_pile"]["available"] and m["original_rf"]["available"] and m["gbm_site"]["available"]
        assert not m["depth_quadratic"]["available"]           # no model in tests
        # the original method looks better on its own training pixels than held out
        ins = m["original_rf"]["pixel_height_in_sample_(original_method)"]["rmse_m"]
        assert ins < m["original_rf"]["pixel_height_oof"]["rmse_m"]
        # knowing the footprint beats guessing it
        assert (m["gbm_pile"]["pile_volume_bias_corrected"]["wape_pct"]
                < m["original_rf"]["pile_volume_bias_corrected"]["wape_pct"])
    # Pléiades applied to the newer scene: only piles imaged on both dates count,
    # and an unchanged site (radiometry aside) shows little change
    rp = rep["recent_pleiades"]["scenes"][0]
    assert 0 < rp["n_piles_seen"] < rp["n_piles"]
    assert abs(rp["change_pct"]) < 25
    seen = [p for p in rep["piles"] if p.get("ple_2026-08-03_m3") is not None]
    assert len(seen) == rp["n_piles_seen"]
    alerts = {p["pile_id"]: p["alert"] for p in rep["piles"]}
    assert alerts["S1"] == "critical" and alerts["S2"] in ("warning", "critical")
    for f in ("report.html", "report.json", "piles.geojson", "piles.csv", "sensor_comparison.csv"):
        assert os.path.exists(tmp_path / f)
    gj = json.load(open(tmp_path / "piles.geojson"))
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0][0]
    assert 5 < lon < 10 and 45 < lat < 55                 # written in lon/lat


class _StubDepth:
    """Stands in for Depth Anything (no model download in CI): returns a
    blurred brightness 'disparity', exercising the tiling → features →
    calibration path end to end."""
    status = "stub"
    backend = "stub"

    def infer(self, rgb):
        from scipy import ndimage
        from stockpile_engine.depth import tiled_inference
        return tiled_inference(rgb, lambda t: ndimage.gaussian_filter(t.mean(axis=2).astype(np.float32), 4),
                               tile=256, overlap=64)


def test_depth_anything_path_runs_through_the_sensor_study(tmp_path):
    StockpileConfig.N_TRAIN_PX, old_n = 30_000, StockpileConfig.N_TRAIN_PX
    StockpileConfig.GBM_MAX_ITER, old_it = 60, StockpileConfig.GBM_MAX_ITER
    try:
        site = make_site()
        rep = run_pipeline(site, "e2e_da", str(tmp_path), sensors=["pleiades"], depth=_StubDepth(),
                           log=lambda *_: None)
    finally:
        StockpileConfig.N_TRAIN_PX, StockpileConfig.GBM_MAX_ITER = old_n, old_it
    m = rep["sensor_study"]["pleiades"]
    assert m["depth_anything"].startswith("used")
    assert m["methods"]["depth_quadratic"]["available"]
    assert "da_relief_20m" in m["methods"]["gbm_pile"]["features"]
