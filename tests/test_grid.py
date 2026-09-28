import numpy as np

from stockpile_engine.grid import Grid, block_mean, features_from_geojson, grid_for_bounds, resample, to_grid


def test_grids_at_every_sensor_resolution_share_an_origin():
    g = grid_for_bounds((360001.3, 5469500.7, 361234.9, 5470011.2), 1.0)
    for r in (1.5, 3.0, 10.0, 30.0):
        h = g.with_res(r)
        assert (h.x0, h.y1) == (g.x0, g.y1)
        assert h.width * r >= g.width * g.res - 1e-6


def test_block_mean_is_area_average():
    g = Grid(0, 100, 1.0, 10, 10)
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    b = block_mean(a, g, g.with_res(5.0))
    assert b.shape == (2, 2)
    assert np.isclose(b[0, 0], a[:5, :5].mean())


def test_block_mean_ignores_nan():
    g = Grid(0, 10, 1.0, 2, 2)
    a = np.array([[1, np.nan], [3, np.nan]], np.float32)
    assert np.isclose(block_mean(a, g, g.with_res(2.0))[0, 0], 2.0)


def test_resample_round_trip_on_smooth_field():
    g = Grid(0, 300, 10.0, 30, 30)
    X, Y = g.xy()
    a = (np.sin(X / 80) + np.cos(Y / 90)).astype(np.float32)
    up = to_grid(a, g, g.with_res(1.0), order=3)
    back = block_mean(up, g.with_res(1.0), g)
    assert np.nanmax(np.abs(back[2:-2, 2:-2] - a[2:-2, 2:-2])) < 0.02


def test_code_editor_multipolygon_is_split_per_drawn_shape():
    mp = {"type": "MultiPolygon", "coordinates": [
        [[[0, 0], [1, 0], [1, 1], [0, 0]]], [[[2, 2], [3, 2], [3, 3], [2, 2]]]]}
    feats = features_from_geojson(mp)
    assert len(feats) == 2 and all(f["geometry"]["type"] == "Polygon" for f in feats)
