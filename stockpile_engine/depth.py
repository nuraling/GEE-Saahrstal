"""Depth Anything V2 on overhead imagery, and calibrating it to metres.

Depth Anything predicts *relative* inverse depth from one image. It was trained
on ground-level photographs; on a nadir satellite image there is no
perspective, so what it responds to is shading, shadow and texture — the same
cues a photo-interpreter uses to see a pile as a mound. Two consequences drive
the design:

* its output has no unit and a different scale in every tile, so each tile is
  standardised and only *local* relief (tile minus its own low-pass) is kept;
* metres come only from calibration against the UAV nDSM, fitted on training
  blocks and scored on held-out blocks (see height_model.py).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import StockpileConfig


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def ensure_onnx_model(path: str = StockpileConfig.DEPTH_ONNX_PATH,
                      url: str = StockpileConfig.DEPTH_ONNX_URL, log=print) -> Optional[str]:
    """The ONNX weights, downloaded once from the GitHub release if missing."""
    import os
    if os.path.isfile(path) and os.path.getsize(path) > 1e7:
        return path
    try:
        import requests
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log(f"Downloading Depth Anything V2 (ONNX) from {url} ...")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path + ".part", "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        os.replace(path + ".part", path)
        return path
    except Exception as exc:
        log(f"ONNX weights unavailable: {exc}")
        return None


class DepthAnything:
    def __init__(self, model_id: str = StockpileConfig.HF_DEPTH_MODEL, log=print):
        self.model_id = model_id
        self.log = log
        self._pipe = None
        self._onnx = None
        self.status = "not_loaded"
        self.backend = None

    def _load(self) -> bool:
        if self._pipe is not None or self._onnx is not None:
            return True
        if not StockpileConfig.ENABLE_DEPTH:
            self.status = "disabled"
            return False
        try:
            import onnxruntime as ort
            path = ensure_onnx_model(log=self.log)
            if path:
                self._onnx = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                self.backend = "onnx"
                self.status = "loaded"
                self.log(f"Depth Anything V2 Small loaded (ONNX runtime, {path})")
                return True
        except ImportError:
            pass
        except Exception as exc:
            self.log(f"ONNX Depth Anything failed ({exc}); trying transformers")
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(task="depth-estimation", model=self.model_id, device=device)
            self.backend = "transformers"
            self.status = "loaded"
            self.log(f"Depth Anything loaded: {self.model_id} ({'cuda' if device == 0 else 'cpu'})")
            return True
        except Exception as exc:
            self.status = f"unavailable: {type(exc).__name__}: {str(exc)[:160]}"
            self.backend = None
            self.log(f"Depth Anything unavailable — continuing without it ({self.status})")
            return False

    def _infer_tile_onnx(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        n = StockpileConfig.DEPTH_TILE_PX
        h, w = rgb.shape[:2]
        x = np.asarray(Image.fromarray(rgb).resize((n, n), Image.BICUBIC), np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        inp = self._onnx.get_inputs()[0].name
        d = self._onnx.run(None, {inp: x.astype(np.float32)})[0][0]
        return np.asarray(Image.fromarray(d.astype(np.float32)).resize((w, h), Image.BILINEAR), np.float32)

    def _infer_tile(self, rgb: np.ndarray) -> np.ndarray:
        if self._onnx is not None:
            return self._infer_tile_onnx(rgb)
        from PIL import Image
        import torch
        out = self._pipe(Image.fromarray(rgb))
        pd = out["predicted_depth"]
        if pd.ndim == 2:
            pd = pd[None, None]
        elif pd.ndim == 3:
            pd = pd[None]
        pd = torch.nn.functional.interpolate(pd.float(), size=rgb.shape[:2],
                                             mode="bilinear", align_corners=False)
        return pd[0, 0].cpu().numpy().astype(np.float32)

    def infer(self, rgb: np.ndarray, tile: int = StockpileConfig.DEPTH_TILE_PX,
              overlap: int = StockpileConfig.DEPTH_TILE_OVERLAP) -> Optional[np.ndarray]:
        """Standardised relative disparity for an H×W×3 uint8 image, or None."""
        if not self._load():
            return None
        return tiled_inference(rgb, self._infer_tile, tile, overlap)


def _standardise(a: np.ndarray) -> np.ndarray:
    med = np.nanmedian(a)
    q75, q25 = np.nanpercentile(a, [75, 25])
    return (a - med) / max(q75 - q25, 1e-6)


def tiled_inference(rgb: np.ndarray, infer_tile, tile: int, overlap: int) -> np.ndarray:
    """Run infer_tile over overlapping tiles and feather-blend the results.

    Each tile's output is standardised first: the model's scale drifts from
    tile to tile, and blending unstandardised tiles paints the tile grid into
    the height map.
    """
    H, W = rgb.shape[:2]
    step = max(tile - overlap, 1)
    acc = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float64)
    ramp = np.minimum(ramp / max(overlap, 1), 1.0)
    w2d = np.outer(ramp, ramp)
    rows = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    cols = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if rows[-1] + tile < H:
        rows.append(max(H - tile, 0))
    if cols[-1] + tile < W:
        cols.append(max(W - tile, 0))
    for r in rows:
        for c in cols:
            sub = rgb[r:r + tile, c:c + tile]
            if sub.size == 0:
                continue
            d = _standardise(infer_tile(np.ascontiguousarray(sub)))
            h, w = d.shape
            ww = w2d[:h, :w]
            acc[r:r + h, c:c + w] += d * ww
            wsum[r:r + h, c:c + w] += ww
    out = (acc / np.maximum(wsum, 1e-9)).astype(np.float32)
    return out


def local_relief(disp: np.ndarray, res: float, window_m: float = 40.0) -> np.ndarray:
    """Disparity minus its own low-pass: what stands proud of its surroundings."""
    sigma = max(window_m / res / 2.0, 1.0)
    base = ndimage.gaussian_filter(np.nan_to_num(disp), sigma)
    return (disp - base).astype(np.float32)


def affine_calibrate(feature: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """height ≈ a·feature + b by robust least squares on mask pixels."""
    x, y = feature[mask], target[mask]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 10:
        return 0.0, float(np.nanmean(y)) if y.size else 0.0
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    keep = np.abs(r) < 2.5 * max(r.std(), 1e-6)
    if keep.sum() > 10:
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return float(coef[0]), float(coef[1])


def to_uint8_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """2–98% stretch per band → uint8 RGB for the model."""
    chans = []
    for a in (red, green, blue):
        lo, hi = np.nanpercentile(a, [2, 98]) if np.isfinite(a).any() else (0, 1)
        chans.append(np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1))
    return (np.dstack(chans) * 255).astype(np.uint8)
