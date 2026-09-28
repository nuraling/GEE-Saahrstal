"""Volumes must match geometry, and real piles must survive extraction."""
import numpy as np

from stockpile_engine import terrain
from stockpile_engine.config import StockpileConfig
from stockpile_engine.grid import Grid


def _cone(g, cx, cy, r, h):
    X, Y = g.xy()
    d = np.hypot(X - cx, Y - cy)
    return np.clip(h * (1 - d / r), 0, None).astype(np.float32)


def test_cone_volume_matches_pi_r2_h_over_3():
    g = Grid(0, 200, 0.5, 400, 400)
    z = _cone(g, 100, 100, 40, 12)
    labels = (z > 0).astype(np.int32)
    v = terrain.pile_volumes(z, labels, g)[0]
    exact = np.pi * 40 ** 2 * 12 / 3
    assert abs(v["volume_m3"] - exact) / exact < 0.01


def test_toe_base_handles_a_sloping_pad():
    g = Grid(0, 200, 1.0, 200, 200)
    X, Y = g.xy()
    pad = (100 + 0.05 * X + 0.02 * Y).astype(np.float32)
    z = _cone(g, 100, 100, 30, 8)
    surface = pad + z
    labels = (z > 0).astype(np.int32)
    wrong_dtm = np.full_like(pad, pad.min())          # a flat, wrong base
    v_dtm = terrain.pile_volumes(surface - wrong_dtm, labels, g)[0]["volume_m3"]
    v_toe = terrain.pile_volumes(surface - wrong_dtm, labels, g, surface, "toe")[0]["volume_m3"]
    exact = np.pi * 30 ** 2 * 8 / 3
    assert abs(v_toe - exact) / exact < 0.05
    assert abs(v_dtm - exact) / exact > 0.5


def test_a_9m_coal_pile_is_kept_and_a_building_and_wagon_are_not():
    """The original extraction kept objects with mean height <= 3 m: it dropped
    every real pile and kept the wagons."""
    g = Grid(0, 300, 1.0, 300, 300)
    X, Y = g.xy()
    h = _cone(g, 80, 220, 45, 9)
    h[(X > 180) & (X < 240) & (Y > 180) & (Y < 230)] = 10.0      # building
    h[(X > 60) & (X < 74) & (Y > 60) & (Y < 63)] = 3.8           # wagon
    dtm = np.zeros_like(h)
    det = terrain.detect_piles(h, dtm + h, g, None, StockpileConfig)
    by_class = {}
    for o in det["objects"]:
        by_class.setdefault(o["class"], []).append(o)
    assert len(by_class.get(terrain.PILE, [])) == 1
    assert by_class[terrain.PILE][0]["max_height_m"] > 8
    assert len(by_class.get(terrain.BUILDING, [])) == 2


def test_tonnage_uses_commodity_density():
    assert terrain.tonnage(100, "iron_ore") > terrain.tonnage(100, "coal") > terrain.tonnage(100, "wood_chips")


def test_operator_labels_teach_the_unlabelled_piles():
    colours = {1: {"blue": .1, "green": .1, "red": .1, "nir": .1},
               2: {"blue": .6, "green": .6, "red": .6, "nir": .6},
               3: {"blue": .12, "green": .11, "red": .1, "nir": .1}}
    out = terrain.commodity_from_examples(colours, {1: "coke", 2: "limestone"})
    assert out == {3: "coke"}


def test_derived_ground_follows_the_pad_not_the_pile():
    g = Grid(0, 400, 1.0, 400, 400)
    X, Y = g.xy()
    pad = (100 + 0.01 * X).astype(np.float32)
    pile = _cone(g, 200, 200, 35, 10)
    d = terrain.derive_dtm(pad + pile, g.res)
    inside = pile > 1
    assert np.nanmax(np.abs(d[inside] - pad[inside])) < 1.0


def test_structures_keep_stacks_above_40m_and_thin_pipes():
    g = Grid(0, 300, 1.0, 300, 300)
    X, Y = g.xy()
    h = np.zeros(g.shape, np.float32)
    h[(np.hypot(X - 50, Y - 250) < 4)] = 80.0                      # stack, 80 m
    h[(X > 100) & (X < 260) & (Y > 148) & (Y < 151)] = 8.0         # pipe bridge, 3 m wide
    h[(X > 150) & (X < 230) & (Y > 30) & (Y < 90)] = 15.0          # hall
    h += _cone(g, 60, 60, 30, 8)                                   # a pile: not a structure
    st = terrain.detect_structures(h, g)
    classes = sorted(o["class"] for o in st["objects"])
    assert classes == sorted([terrain.CHIMNEY, terrain.PIPE, terrain.ROOF])
