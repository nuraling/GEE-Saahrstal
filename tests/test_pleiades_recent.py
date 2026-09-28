import numpy as np

from stockpile_engine.config import StockpileConfig
from stockpile_engine.sources_ee import band_order_check


def _scene(swap=False):
    rng = np.random.default_rng(0)
    red = rng.uniform(0.03, 0.08, (50, 50)).astype(np.float32)
    nir = rng.uniform(0.3, 0.5, (50, 50)).astype(np.float32)      # vegetation
    red[:25], nir[:25] = 0.2, 0.2                                  # bare ground
    b = {"blue": red, "green": red, "red": red, "nir": nir}
    if swap:
        b["red"], b["nir"] = b["nir"], b["red"]
    return b


def test_band_check_passes_on_correct_order():
    assert band_order_check(_scene())["ok"]


def test_band_check_flags_swapped_red_nir():
    chk = band_order_check(_scene(swap=True))
    assert not chk["ok"] and "band map" in chk["note"]


def test_pleiades_20260803_is_configured():
    assert StockpileConfig.PLEIADES_RECENT["2026-08-03"].endswith("Saarlouis/pleiades_20260803")
