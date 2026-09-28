"""Per-pixel predictors for the optical → height / footprint models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .depth import local_relief
from .terrain import pseudo_reflectance


def _local_std(a: np.ndarray, size: int) -> np.ndarray:
    m = ndimage.uniform_filter(a, size)
    m2 = ndimage.uniform_filter(a * a, size)
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def build_features(bands: Dict[str, np.ndarray], res: float,
                   disparity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """H×W×F stack.

    Beyond the raw bands (all the original Random Forest saw), the stack adds
    what actually carries height information in a single image: shading
    gradients (a pile's sunlit and shadowed flanks), texture, and multi-scale
    brightness contrast. Window sizes are in metres so 1 m and 10 m sensors
    are described consistently.
    """
    refl = pseudo_reflectance({k: v for k, v in bands.items() if k in ("blue", "green", "red", "nir")})
    feats, names = [], []
    for k in ("blue", "green", "red", "nir"):
        if k in refl:
            feats.append(np.nan_to_num(refl[k]))
            names.append(k)
    r, g, b = (np.nan_to_num(refl.get(k, np.zeros_like(next(iter(refl.values())))))
               for k in ("red", "green", "blue"))
    bright = (r + g + b) / 3.0
    feats += [bright, (r - b) / (r + b + 1e-6)]
    names += ["brightness", "redness"]
    if "nir" in refl:
        nir = np.nan_to_num(refl["nir"])
        feats.append((nir - r) / (nir + r + 1e-6))
        names.append("ndvi")

    gy, gx = np.gradient(bright, res)
    feats += [gx, gy, np.hypot(gx, gy)]
    names += ["grad_x", "grad_y", "grad_mag"]
    for w_m in (5.0, 15.0, 40.0):
        px = max(int(round(w_m / res)), 3)
        feats.append(_local_std(bright, px))
        names.append(f"tex_std_{int(w_m)}m")
        sig = max(w_m / res / 2.0, 0.5)
        feats.append(bright - ndimage.gaussian_filter(bright, sig))
        names.append(f"dog_{int(w_m)}m")
    if disparity is not None:
        d = np.nan_to_num(disparity)
        feats += [d, local_relief(d, res, 20.0), local_relief(d, res, 60.0)]
        names += ["da_disp", "da_relief_20m", "da_relief_60m"]
    X = np.dstack(feats).astype(np.float32)
    # no imagery → no features: filters above ran on zero-filled bands, but a
    # pixel outside the scene must never look like "dark ground" to a model
    nodata = np.zeros(X.shape[:2], bool)
    for k in ("blue", "green", "red", "nir"):
        if k in bands:
            nodata |= ~np.isfinite(bands[k])
    X[nodata] = np.nan
    return X, names
