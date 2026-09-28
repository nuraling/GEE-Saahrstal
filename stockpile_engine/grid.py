"""A metric raster grid, and moving arrays between grids.

Every source (UAV, Pléiades, SPOT, Sentinel-2, Landsat) is fetched onto a Grid
that shares its origin with the UAV grid, so resampling between resolutions is
pure array arithmetic with no re-registration error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Grid:
    x0: float          # west edge (m)
    y1: float          # north edge (m)
    res: float         # pixel size (m)
    width: int
    height: int
    crs: str = "EPSG:25832"

    @property
    def x1(self) -> float:
        return self.x0 + self.width * self.res

    @property
    def y0(self) -> float:
        return self.y1 - self.height * self.res

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def pixel_area(self) -> float:
        return self.res * self.res

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y1, self.res, self.res)

    def bounds(self):
        return (self.x0, self.y0, self.x1, self.y1)

    def with_res(self, res: float) -> "Grid":
        """Same ground extent at another pixel size (extent rounded outward)."""
        w = int(np.ceil(self.width * self.res / res - 1e-9))
        h = int(np.ceil(self.height * self.res / res - 1e-9))
        return replace(self, res=float(res), width=w, height=h)

    def xy(self):
        """Pixel-centre coordinate arrays (X, Y)."""
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.res
        ys = self.y1 - (np.arange(self.height) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def ee_grid(self) -> dict:
        """Pixel grid dict for ee.data.computePixels."""
        return {
            "dimensions": {"width": self.width, "height": self.height},
            "affineTransform": {"scaleX": self.res, "shearX": 0, "translateX": self.x0,
                                "shearY": 0, "scaleY": -self.res, "translateY": self.y1},
            "crsCode": self.crs,
        }

    def subgrids(self, max_px: int) -> List["Grid"]:
        """Split into tiles of at most max_px × max_px (for request limits)."""
        out = []
        for r0 in range(0, self.height, max_px):
            for c0 in range(0, self.width, max_px):
                out.append(Grid(self.x0 + c0 * self.res, self.y1 - r0 * self.res, self.res,
                                min(max_px, self.width - c0), min(max_px, self.height - r0),
                                self.crs))
        return out

    def offset_in(self, parent: "Grid"):
        """(row, col) of this grid's top-left pixel inside parent."""
        return (int(round((parent.y1 - self.y1) / self.res)),
                int(round((self.x0 - parent.x0) / self.res)))


def grid_for_bounds(bounds, res: float, crs: str = "EPSG:25832") -> Grid:
    minx, miny, maxx, maxy = bounds
    # snap to a multiple of 30 m so 1, 1.5, 3, 10 and 30 m grids nest exactly
    snap = 30.0
    x0 = np.floor(minx / snap) * snap
    y1 = np.ceil(maxy / snap) * snap
    x1 = np.ceil(maxx / snap) * snap
    y0 = np.floor(miny / snap) * snap
    return Grid(float(x0), float(y1), float(res),
                int(round((x1 - x0) / res)), int(round((y1 - y0) / res)), crs)


# ── resampling ────────────────────────────────────────────────────────────────

def block_mean(arr: np.ndarray, src: Grid, dst: Grid) -> np.ndarray:
    """Area-average a fine array onto a coarser grid (NaN-aware).

    This is how a sensor with bigger pixels would have seen the same ground.
    """
    f = dst.res / src.res
    k = int(round(f))
    if abs(f - k) > 1e-6 or k < 1:
        # non-integer ratio: fall back to zoom-based resample
        return resample(arr, src, dst, order=1)
    H, W = dst.height * k, dst.width * k
    pad = np.full((H, W), np.nan, dtype=np.float32)
    h, w = min(H, arr.shape[0]), min(W, arr.shape[1])
    pad[:h, :w] = arr[:h, :w]
    blocks = pad.reshape(dst.height, k, dst.width, k)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(blocks).sum(axis=(1, 3))
        s = np.nansum(blocks, axis=(1, 3))
        out = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float32)


def resample(arr: np.ndarray, src: Grid, dst: Grid, order: int = 1) -> np.ndarray:
    """Interpolate onto dst (order 0 nearest, 1 bilinear, 3 cubic). NaN-aware."""
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    fill = float(np.nanmean(arr)) if valid.any() else 0.0
    filled = np.where(valid, arr, fill)
    ys = (src.y1 - (dst.y1 - (np.arange(dst.height) + 0.5) * dst.res)) / src.res - 0.5
    xs = ((dst.x0 + (np.arange(dst.width) + 0.5) * dst.res) - src.x0) / src.res - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    out = ndimage.map_coordinates(filled, [yy, xx], order=order, mode="nearest")
    vmask = ndimage.map_coordinates(valid.astype(np.float32), [yy, xx], order=1, mode="nearest")
    out[vmask < 0.5] = np.nan
    return out.astype(np.float32)


def to_grid(arr: np.ndarray, src: Grid, dst: Grid, order: int = 1) -> np.ndarray:
    """Coarser → block mean; finer → interpolate; same → copy."""
    if abs(src.res - dst.res) < 1e-9 and src.shape == dst.shape:
        return np.asarray(arr, dtype=np.float32).copy()
    if dst.res > src.res:
        return block_mean(arr, src, dst)
    return resample(arr, src, dst, order=order)


# ── vector ↔ raster ──────────────────────────────────────────────────────────

def lonlat_geom_to_crs(geom: dict, crs: str) -> dict:
    from rasterio.warp import transform_geom
    return transform_geom("EPSG:4326", crs, geom)


def crs_geom_to_lonlat(geom: dict, crs: str) -> dict:
    from rasterio.warp import transform_geom
    return transform_geom(crs, "EPSG:4326", geom)


def rasterize_labels(geoms: Iterable[dict], grid: Grid, all_touched: bool = False) -> np.ndarray:
    """Label raster: 0 background, k+1 for the k-th geometry (grid CRS)."""
    from rasterio.features import rasterize
    shapes = [(g, i + 1) for i, g in enumerate(geoms)]
    if not shapes:
        return np.zeros(grid.shape, dtype=np.int32)
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                     fill=0, dtype="int32", all_touched=all_touched)


def vectorize_labels(labels: np.ndarray, grid: Grid, simplify_m: float = 0.0) -> dict:
    """{label: GeoJSON geometry in grid CRS} for every positive label."""
    from rasterio.features import shapes
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    parts = {}
    for geom, val in shapes(labels.astype(np.int32), mask=labels > 0, transform=grid.transform):
        parts.setdefault(int(val), []).append(shape(geom))
    out = {}
    for k, polys in parts.items():
        g = unary_union(polys)
        if simplify_m:
            g = g.simplify(simplify_m, preserve_topology=True)
        out[k] = mapping(g)
    return out


def split_multipolygon(geom: dict) -> List[dict]:
    """A Code-Editor MultiPolygon → one Polygon per drawn shape."""
    if geom["type"] == "MultiPolygon":
        return [{"type": "Polygon", "coordinates": c} for c in geom["coordinates"]]
    if geom["type"] == "Polygon":
        return [geom]
    if geom["type"] == "GeometryCollection":
        out = []
        for g in geom["geometries"]:
            out.extend(split_multipolygon(g))
        return out
    raise ValueError(f"unsupported geometry type {geom['type']}")


def features_from_geojson(obj: Optional[dict]) -> List[dict]:
    """Accept Feature / FeatureCollection / bare geometry → list of Features
    whose geometries are single Polygons (properties preserved)."""
    if not obj:
        return []
    if obj.get("type") == "FeatureCollection":
        feats = obj.get("features", [])
    elif obj.get("type") == "Feature":
        feats = [obj]
    else:
        feats = [{"type": "Feature", "geometry": obj, "properties": {}}]
    out = []
    for f in feats:
        for poly in split_multipolygon(f["geometry"]):
            out.append({"type": "Feature", "geometry": poly,
                        "properties": dict(f.get("properties") or {})})
    return out
