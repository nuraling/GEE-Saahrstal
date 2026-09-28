"""Thermal physics and anomaly logic."""
import numpy as np

from stockpile_engine import thermal
from stockpile_engine.synthetic import make_site


def test_planck_round_trip():
    t = np.array([250.0, 300.0, 400.0, 700.0])
    assert np.allclose(thermal.inverse_planck(thermal.planck_radiance(t)), t)


def test_hotspot_inverts_radiance_mixing():
    bg, hot, area, fp = 20.0, 150.0, 25.0, 100.0
    phi = area / fp ** 2
    L = phi * thermal.planck_radiance(hot + 273.15) + (1 - phi) * thermal.planck_radiance(bg + 273.15)
    t_obs = thermal.inverse_planck(L) - 273.15
    assert abs(thermal.hotspot_temperature(t_obs, bg, area, fp) - hot) < 1e-6


def test_the_original_30m_t4_unmixing_overstates_what_landsat_can_see():
    """Band 10 is measured at 100 m, not 30 m: the same 1 °C excess implies a
    much hotter spot when the footprint is the real one."""
    t30 = thermal.hotspot_temperature(21.0, 20.0, 25.0, 30.0)
    t100 = thermal.hotspot_temperature(21.0, 20.0, 25.0, 100.0)
    assert t100 > t30 + 50


def test_finer_thermal_sensors_detect_cooler_hotspots():
    md = {fp: thermal.min_detectable_hotspot(20.0, 0.8, 2.0, 25.0, fp) for fp in (100.0, 70.0, 3.5)}
    assert md[100.0] > md[70.0] > md[3.5]
    assert md[3.5] < 30.0


def test_synthetic_site_warm_pile_and_swir_fire_are_flagged():
    site = make_site(n_scenes=8)
    from stockpile_engine.grid import features_from_geojson, rasterize_labels
    g = site["grid"]
    labels = rasterize_labels([f["geometry"] for f in features_from_geojson(site["stockpiles"])], g)
    res = thermal.analyse_scenes(site["thermal"], site["thermal_grid"], labels, g, labels > 0)
    by = {s["label"]: s for s in res["summary"]}
    assert by[2]["persistent_anomaly"]            # S2: whole surface +6 °C
    assert not by[4]["persistent_anomaly"]
    sw = thermal.swir_hotspots(site["swir"], labels, g, site["swir_grid"])
    assert len(sw["per_object"][1]) == 2          # S1: the burning spot, 2 of 4 scenes
    assert all(len(v) == 0 for k, v in sw["per_object"].items() if k != 1)


def test_dark_material_is_not_a_fire():
    """Coal and scrap reflect as much at 2.2 µm as at 1.6 µm in every scene.
    On the real Saarlouis yard the plain NHI > 0 test flagged 50 of 58 piles
    in ~19 of 20 scenes. Only a departure from the pixel's own normal counts."""
    from stockpile_engine.grid import Grid
    g = Grid(0, 100, 10.0, 10, 10)
    labels = np.zeros((100, 100), np.int32)
    labels[20:60, 20:60] = 1
    g1 = Grid(0, 100, 1.0, 100, 100)
    scenes = []
    for i in range(10):
        b11 = np.full((10, 10), 0.05, np.float32)
        b12 = np.full((10, 10), 0.055, np.float32)      # NHI ≈ +0.05, every scene
        if i == 7:
            b12[4, 4] = 0.20                            # one real event
        scenes.append({"b11": b11, "b12": b12, "date": f"2025-05-{i + 1:02d}"})
    sw = thermal.swir_hotspots(scenes, labels, g1, g)
    assert [h["date"] for h in sw["per_object"][1]] == ["2025-05-08"]


def test_swir_hits_pool_across_adjacent_piles():
    import numpy as np
    from stockpile_engine.grid import Grid
    from stockpile_engine.pipeline import _alert_level, pool_swir_neighbours
    labels = np.zeros((60, 60), np.int32)
    labels[10:20, 10:20] = 1          # pile A
    labels[10:20, 25:35] = 2          # pile B, 5 m away
    labels[45:55, 45:55] = 3          # pile C, far
    g = Grid(0.0, 60.0, 1.0, 60, 60, "EPSG:25832")
    piles = [{"label": 1, "pile_id": "A", "swir_hot_scenes": 1, "swir_hot_dates": ["2025-03-27"]},
             {"label": 2, "pile_id": "B", "swir_hot_scenes": 1, "swir_hot_dates": ["2025-04-03"]},
             {"label": 3, "pile_id": "C", "swir_hot_scenes": 1, "swir_hot_dates": ["2025-05-01"]}]
    pool_swir_neighbours(piles, labels, g, 20.0)
    alerts = {p["pile_id"]: _alert_level(p) for p in piles}
    assert alerts["A"] == "critical" and alerts["B"] == "critical"   # one hot spot split between neighbours
    assert alerts["C"] != "critical"                                 # a single date elsewhere stays below
