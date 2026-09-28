import numpy as np

from stockpile_engine.config import StockpileConfig
from stockpile_engine.depth import DepthAnything, affine_calibrate, local_relief, tiled_inference
from stockpile_engine.grid import Grid
from stockpile_engine.height_model import spatial_folds
from stockpile_engine.metrics import object_metrics


def test_folds_hold_out_the_same_ground_at_every_resolution():
    g = Grid(0, 900, 1.0, 900, 900)
    f1 = spatial_folds(g, 150, 5, 42)
    f10 = spatial_folds(g.with_res(10), 150, 5, 42, ref=g)
    # centre of every 10 m pixel falls in the same fold at 1 m
    assert np.array_equal(f10, f1[5::10, 5::10])
    assert len(np.unique(f1)) == 5


def test_depth_model_id_is_a_transformers_checkpoint():
    # "depth-anything/Depth-Anything-V2-small" (the original config) is not
    # loadable by transformers.pipeline; the -hf repos are.
    assert StockpileConfig.HF_DEPTH_MODEL.endswith("-hf")


def test_tiled_inference_hides_the_tile_grid():
    H, W = 700, 900
    yy, xx = np.mgrid[0:H, 0:W]
    field = np.sin(xx / 60.0) + np.cos(yy / 70.0)

    def fake_model(rgb):                    # a model with a per-tile scale drift
        return rgb[..., 0].astype(np.float32) * np.random.uniform(0.5, 2.0) + np.random.uniform(-5, 5)

    rgb = np.repeat(((field - field.min()) / np.ptp(field) * 255).astype(np.uint8)[..., None], 3, axis=2)
    out = tiled_inference(rgb, fake_model, tile=256, overlap=64)
    assert out.shape == (H, W)
    assert np.corrcoef(out.ravel(), field.ravel())[0, 1] > 0.9


def test_affine_calibration_recovers_scale_and_offset():
    rng = np.random.default_rng(0)
    d = rng.normal(size=5000)
    h = 3.0 * d + 1.5 + rng.normal(0, 0.1, 5000)
    a, b = affine_calibrate(d, h, np.ones_like(d, bool))
    assert abs(a - 3.0) < 0.05 and abs(b - 1.5) < 0.05


def test_depth_disabled_returns_none_not_an_exception():
    StockpileConfig.ENABLE_DEPTH, old = False, StockpileConfig.ENABLE_DEPTH
    try:
        assert DepthAnything(log=lambda *_: None).infer(np.zeros((10, 10, 3), np.uint8)) is None
    finally:
        StockpileConfig.ENABLE_DEPTH = old


def test_wape_is_reported_unscaled():
    m = object_metrics([100, 100], [120, 80])
    assert m["wape_pct"] == 20.0 and m["bias_pct"] == 0.0


def test_bias_correction_never_uses_a_piles_own_fold():
    from stockpile_engine.pipeline import bias_correct
    true = np.array([100, 100, 100, 100, 1000.0])
    pred = np.array([50, 50, 50, 50, 50.0])
    fold = np.array([0, 0, 1, 1, 2])
    out = bias_correct(true, pred, fold)
    # pile 5 (fold 2) is scaled by the other folds only: 400/200 = 2 → 100, not its own 1000
    assert out[4] == 100.0


def test_quadratic_calibration_fits_a_curved_relation_and_matches_the_mean():
    from stockpile_engine.height_model import poly_calibrate
    rng = np.random.default_rng(1)
    d = rng.uniform(0, 3, 4000)
    h = 1.5 * d ** 2 + rng.normal(0, 0.2, 4000)          # curved, like DA vs UAV
    fn = poly_calibrate(d, h, np.ones_like(d, bool), 2)
    assert abs(fn(d).mean() - h.mean()) < 0.01
    lin = poly_calibrate(d, h, np.ones_like(d, bool), 1)
    assert np.mean((fn(d) - h) ** 2) < 0.5 * np.mean((lin(d) - h) ** 2)


def test_quadratic_volume_correction_is_leak_free_and_mean_matched():
    from stockpile_engine.pipeline import quadratic_correct, _quad_fit
    rng = np.random.default_rng(2)
    v = rng.uniform(500, 30000, 40)
    true = 0.6 * v + 2e-5 * v * v
    fold = np.arange(40) % 5
    out = quadratic_correct(true, v, fold)
    assert np.all(np.isfinite(out))
    fn = _quad_fit(true, v)
    assert abs(fn(v).sum() - true.sum()) / true.sum() < 1e-6
    # changing a pile's own truth must not change its own correction
    t2 = true.copy(); t2[0] *= 10
    assert quadratic_correct(t2, v, fold)[0] == out[0]


def test_power_correction_recovers_a_square_relation():
    from stockpile_engine.pipeline import power_correct
    rng = np.random.default_rng(3)
    v = rng.uniform(1000, 20000, 60)
    true = 1e-4 * v ** 2 * rng.lognormal(0, 0.05, 60)
    out, b = power_correct(true, v, np.arange(60) % 5)
    assert 1.9 < b < 2.1
    assert np.abs(out - true).sum() / true.sum() < 0.08


def test_port_calibration_never_uses_the_pile_it_corrects():
    from stockpile_engine.pipeline import port_local_correction
    act = [10000.0, 20000.0, 30000.0, 15000.0, 25000.0, 12000.0, 5000.0]
    piles = [{"volume_m3": a, "port_s_m_m3": 0.35 * a} for a in act] + [{"volume_m3": 9000.0}]
    study = {"in_port": [True] * 7 + [False],
             "results": {"s": {"port_holdout": {"m": {"n_port_piles_seen": 7}}}}}
    ref = {"piles": piles}
    port_local_correction(ref, study)
    r = study["results"]["s"]["port_holdout"]["m"]["power_corrected_port_loo"]
    assert r["wape_pct"] < 1.0                       # a clean 0.35x bias is fully removed
    piles2 = [dict(p) for p in piles]
    piles2[0]["volume_m3"] = 99999.0                  # pile 0's own truth
    ref2 = {"piles": piles2}
    port_local_correction(ref2, {"in_port": study["in_port"],
                                 "results": {"s": {"port_holdout": {"m": {"n_port_piles_seen": 7}}}}})
    assert abs(ref2["piles"][0]["port_s_m_pl_m3"] - piles[0]["port_s_m_pl_m3"]) < 1e-6


def test_sensitivity_check_flags_a_model_that_cannot_see_a_pile_vanish():
    from stockpile_engine.port_monitor import sensitivity_check

    class Blind:                       # volume from the polygon alone: ignores the image
        labels10 = np.zeros((20, 20), np.int32)
        labels10[5:10, 5:10] = 1
        piles = [{"label": 1, "pile_id": "P1"}]
        port_idx = [0]
        uav = np.array([1000.0])
        ref_bands = {k: np.full((20, 20), 0.1, np.float32) for k in ("red", "green", "blue", "nir")}

        def scene_volumes(self, b):
            return {"corrected": np.array([1000.0])}

    class Seeing(Blind):
        def scene_volumes(self, b):
            return {"corrected": np.array([10000.0 * float(np.mean(b["red"][5:10, 5:10]))])}

    assert not sensitivity_check(Blind())["passed"]
    s = Seeing()
    s.ref_bands = {k: v.copy() for k, v in s.ref_bands.items()}
    s.ref_bands["red"][5:10, 5:10] = 0.3           # the pile is brighter than its ground
    assert sensitivity_check(s)["passed"]
