# -*- coding: utf-8 -*-
"""
Préparation des données pour l'inférence / entraînement GNO.
Lecture raster/LAS, normalisation, tiling, graphes k-NN.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import xy as rasterio_xy
except ImportError:
    rasterio = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


@dataclass
class PointCloud:
    xy: np.ndarray
    values: np.ndarray
    nodata_mask: Optional[np.ndarray] = None


@dataclass
class Tile:
    xy: np.ndarray
    values: np.ndarray
    core_indices: np.ndarray
    global_indices: np.ndarray


@dataclass
class NormStats:
    vmin: float
    vmax: float

    def normalize(self, x):
        denom = self.vmax - self.vmin
        if denom == 0 or not np.isfinite(denom):
            return np.zeros_like(x, dtype=np.float32)
        return ((x - self.vmin) / denom).astype(np.float32)

    def denormalize(self, x):
        return (x * (self.vmax - self.vmin) + self.vmin).astype(np.float32)

    def to_dict(self):
        return {"vmin": float(self.vmin), "vmax": float(self.vmax)}

    @classmethod
    def from_dict(cls, d):
        return cls(vmin=float(d["vmin"]), vmax=float(d["vmax"]))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def read_raster_as_points(path, band=1):
    if rasterio is None:
        raise ImportError("rasterio n'est pas installé.")
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        meta = src.meta.copy()
        rows, cols = np.where(
            np.isfinite(data) if nodata is None
            else (data != nodata) & np.isfinite(data)
        )
        if len(rows) == 0:
            raise ValueError(f"Aucune donnée valide dans le raster {path}")
        xs, ys = rasterio_xy(transform, rows, cols, offset="center")
        xy = np.column_stack([xs, ys]).astype(np.float64)
        values = data[rows, cols]
        cloud = PointCloud(xy=xy, values=values)
        meta["valid_rows"] = rows
        meta["valid_cols"] = cols
        meta["shape"] = data.shape
        return cloud, meta


def read_las_as_points(path, dimension="z"):
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy n'est pas installé. pip install laspy lazrs")
    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    dim = dimension.lower().strip()
    if dim == "z":
        values = np.asarray(las.z, dtype=np.float32)
    elif dim == "intensity":
        values = np.asarray(las.intensity, dtype=np.float32)
    elif dim == "classification":
        values = np.asarray(las.classification, dtype=np.float32)
    elif hasattr(las, dim):
        values = np.asarray(getattr(las, dim), dtype=np.float32)
    else:
        extra_names = [d.name for d in las.point_format.extra_dimensions] if hasattr(las.point_format, "extra_dimensions") else []
        if dim in extra_names:
            values = np.asarray(las[dim], dtype=np.float32)
        else:
            raise ValueError(f"Dimension '{dimension}' introuvable dans {path}.")
    valid = np.isfinite(values)
    if not np.any(valid):
        raise ValueError(f"Aucune valeur valide pour '{dimension}'")
    xy = np.column_stack([x[valid], y[valid]])
    return PointCloud(xy=xy, values=values[valid], nodata_mask=valid), las


def compute_norm_stats(values):
    return NormStats(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))


def load_norm_stats_beside_model(model_path):
    for path in (model_path + ".norm.json", os.path.splitext(model_path)[0] + ".norm.json"):
        if os.path.isfile(path):
            return NormStats.load(path)
    return None


def build_knn_graph(xy, k=8):
    if cKDTree is None:
        raise ImportError("scipy n'est pas installé.")
    n = len(xy)
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, k), dtype=np.float32)
    k_eff = min(k + 1, n)
    tree = cKDTree(xy)
    dists, idxs = tree.query(xy, k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    idxs = idxs[:, 1:]
    dists = dists[:, 1:]
    sources = np.repeat(np.arange(n), idxs.shape[1])
    targets = idxs.ravel()
    edge_index = np.stack([sources, targets], axis=0).astype(np.int64)
    return edge_index, dists.astype(np.float32)


def iter_row_tiles(cloud, meta, core_rows=64, overlap_rows=8):
    rows = meta["valid_rows"]
    n_rows_img = meta["shape"][0]
    unique_rows = np.unique(rows)
    row_to_indices = {}
    for i, r in enumerate(rows):
        row_to_indices.setdefault(int(r), []).append(i)
    step = max(1, core_rows)
    for r0 in range(0, n_rows_img, step):
        r1 = min(r0 + core_rows, n_rows_img)
        r0_ov = max(0, r0 - overlap_rows)
        r1_ov = min(n_rows_img, r1 + overlap_rows)
        tile_global, core_local = [], []
        for r in range(r0_ov, r1_ov):
            for gi in row_to_indices.get(r, []):
                local_idx = len(tile_global)
                tile_global.append(gi)
                if r0 <= r < r1:
                    core_local.append(local_idx)
        if not core_local:
            continue
        tile_global = np.array(tile_global, dtype=np.int64)
        core_local = np.array(core_local, dtype=np.int64)
        yield Tile(
            xy=cloud.xy[tile_global],
            values=cloud.values[tile_global],
            core_indices=core_local,
            global_indices=tile_global[core_local],
        )


def iter_spatial_tiles(cloud, cell_size=None, target_points_per_tile=4096, overlap_ratio=0.15):
    xy = cloud.xy
    n = len(xy)
    if n == 0:
        return
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    extent_x, extent_y = xmax - xmin, ymax - ymin
    if cell_size is None or cell_size <= 0:
        area = max(extent_x * extent_y, 1e-12)
        density = n / area
        cell_area = target_points_per_tile / max(density, 1e-12)
        cell_size = float(np.sqrt(cell_area))
    overlap = cell_size * overlap_ratio
    step = cell_size
    nx = max(1, int(np.ceil(extent_x / step)))
    ny = max(1, int(np.ceil(extent_y / step)))
    tree = cKDTree(xy) if cKDTree is not None else None
    for ix in range(nx):
        for iy in range(ny):
            cx0, cy0 = xmin + ix * step, ymin + iy * step
            cx1, cy1 = cx0 + cell_size, cy0 + cell_size
            ox0, oy0 = cx0 - overlap, cy0 - overlap
            ox1, oy1 = cx1 + overlap, cy1 + overlap
            if tree is not None:
                center = np.array([(ox0 + ox1) / 2, (oy0 + oy1) / 2])
                radius = np.sqrt(((ox1 - ox0) / 2) ** 2 + ((oy1 - oy0) / 2) ** 2)
                candidates = tree.query_ball_point(center, radius)
                if not candidates:
                    continue
                cand = np.array(candidates, dtype=np.int64)
                mask = ((xy[cand, 0] >= ox0) & (xy[cand, 0] < ox1) &
                        (xy[cand, 1] >= oy0) & (xy[cand, 1] < oy1))
                tile_global = cand[mask]
            else:
                mask = ((xy[:, 0] >= ox0) & (xy[:, 0] < ox1) &
                        (xy[:, 1] >= oy0) & (xy[:, 1] < oy1))
                tile_global = np.where(mask)[0]
            if len(tile_global) == 0:
                continue
            core_mask = ((xy[tile_global, 0] >= cx0) & (xy[tile_global, 0] < cx1) &
                         (xy[tile_global, 1] >= cy0) & (xy[tile_global, 1] < cy1))
            core_local = np.where(core_mask)[0]
            if len(core_local) == 0:
                continue
            yield Tile(
                xy=xy[tile_global],
                values=cloud.values[tile_global],
                core_indices=core_local,
                global_indices=tile_global[core_local],
            )
