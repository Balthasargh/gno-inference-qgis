# -*- coding: utf-8 -*-
"""
Écriture des résultats d'inférence :
  - Raster → GeoTIFF (même emprise, résolution, CRS)
  - Nuage de points → LAS/LAZ (nouvelle dimension gno_prediction)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def write_raster_predictions(
    path: str,
    predictions: np.ndarray,
    meta: dict,
    nodata: Optional[float] = None,
) -> None:
    """
    Réécrit un GeoTIFF à partir des prédictions sur les pixels valides.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio n'est pas installé.")

    rows = meta["valid_rows"]
    cols = meta["valid_cols"]
    shape = meta["shape"]

    out = np.full(
        shape,
        nodata if nodata is not None else (meta.get("nodata") or -9999.0),
        dtype=np.float32,
    )
    out[rows, cols] = predictions.astype(np.float32)

    out_meta = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": meta.get("crs"),
        "transform": meta.get("transform"),
        "nodata": nodata if nodata is not None else (meta.get("nodata") or -9999.0),
        "compress": "deflate",
    }

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(out, 1)


def write_las_predictions(
    path: str,
    predictions: np.ndarray,
    las,
    valid_mask: Optional[np.ndarray] = None,
    dimension_name: str = "gno_prediction",
) -> None:
    """
    Écrit un LAS/LAZ identique à l'entrée avec une dimension supplémentaire.
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy n'est pas installé.")

    header = las.header
    try:
        from laspy import ExtraBytesParams
        if dimension_name not in [d.name for d in las.point_format.extra_dimensions]:
            las.add_extra_dim(ExtraBytesParams(name=dimension_name, type=np.float32))
    except Exception:
        pass

    full_pred = np.zeros(len(las.points), dtype=np.float32)
    if valid_mask is not None:
        full_pred[valid_mask] = predictions.astype(np.float32)
    else:
        if len(predictions) == len(las.points):
            full_pred = predictions.astype(np.float32)
        else:
            raise ValueError(
                f"Taille des prédictions ({len(predictions)}) incompatible "
                f"avec le LAS ({len(las.points)})."
            )

    try:
        las[dimension_name] = full_pred
    except Exception:
        new_las = laspy.LasData(header)
        for dim in las.point_format.dimension_names:
            setattr(new_las, dim, getattr(las, dim))
        try:
            from laspy import ExtraBytesParams
            new_las.add_extra_dim(ExtraBytesParams(name=dimension_name, type=np.float32))
            new_las[dimension_name] = full_pred
            las = new_las
        except Exception as exc:
            raise RuntimeError(f"Impossible d'ajouter la dimension {dimension_name}: {exc}")

    las.write(path)
