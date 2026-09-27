# -*- coding: utf-8 -*-
"""
Moteur d'inférence GNO — SEULE implémentation du pipeline
entrée → tuiles → GNO → sortie (raster ou nuage de points).

Communication par callbacks simples, sans dépendance Qt / Processing.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch

from ..utils import data_preparation as dp
from ..utils.export_results import write_las_predictions, write_raster_predictions
from ..utils.model_loader import load_model, resolve_device


class GNOInferenceEngine:
    """Pipeline d'inférence unifié."""

    def __init__(
        self,
        model_path: str,
        k: int = 8,
        batch_size: int = 4096,
        device_str: str = "cpu",
        progress_cb: Optional[Callable[[float], None]] = None,
        message_cb: Optional[Callable[[str], None]] = None,
        warning_cb: Optional[Callable[[str], None]] = None,
        is_canceled_cb: Optional[Callable[[], bool]] = None,
    ):
        self.model_path = model_path
        self.k = k
        self.batch_size = batch_size
        self.device = resolve_device(device_str)
        self.progress_cb = progress_cb or (lambda p: None)
        self.message_cb = message_cb or (lambda m: None)
        self.warning_cb = warning_cb or (lambda w: None)
        self.is_canceled_cb = is_canceled_cb or (lambda: False)
        self._model = None
        self._norm = None

    def run(self, input_path: str, output_path: str, las_dimension: str = "z") -> str:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in (".las", ".laz"):
            return self._run_pointcloud(input_path, output_path, las_dimension)
        return self._run_raster(input_path, output_path)

    def _run_raster(self, input_path: str, output_path: str) -> str:
        self.message_cb(f"Lecture du raster : {input_path}")
        cloud, meta = dp.read_raster_as_points(input_path)
        self._setup_norm(cloud.values)
        cloud.values = self._norm.normalize(cloud.values)
        n = len(cloud.values)
        predictions = np.full(n, np.nan, dtype=np.float32)
        self.message_cb("Chargement du modèle…")
        self._load_model()
        tiles = list(dp.iter_row_tiles(cloud, meta, core_rows=max(16, self.batch_size // 64)))
        self.message_cb(f"{len(tiles)} tuile(s) à traiter ({n} points).")
        self._infer_on_tiles(tiles, predictions)
        if self.is_canceled_cb():
            raise InterruptedError("Inférence annulée.")
        out_values = self._norm.denormalize(predictions)
        self.message_cb(f"Écriture du raster de sortie : {output_path}")
        write_raster_predictions(output_path, out_values, meta)
        self.progress_cb(100.0)
        return output_path

    def _run_pointcloud(self, input_path: str, output_path: str, dimension: str = "z") -> str:
        self.message_cb(f"Lecture du nuage de points : {input_path}")
        cloud, las = dp.read_las_as_points(input_path, dimension=dimension)
        self._setup_norm(cloud.values)
        cloud.values = self._norm.normalize(cloud.values)
        n = len(cloud.values)
        predictions = np.full(n, np.nan, dtype=np.float32)
        self.message_cb("Chargement du modèle…")
        self._load_model()
        tiles = list(dp.iter_spatial_tiles(cloud, target_points_per_tile=self.batch_size))
        self.message_cb(f"{len(tiles)} tuile(s) à traiter ({n} points).")
        self._infer_on_tiles(tiles, predictions)
        if self.is_canceled_cb():
            raise InterruptedError("Inférence annulée.")
        out_values = self._norm.denormalize(predictions)
        self.message_cb(f"Écriture du nuage de points de sortie : {output_path}")
        write_las_predictions(output_path, out_values, las, valid_mask=cloud.nodata_mask)
        self.progress_cb(100.0)
        return output_path

    def _infer_on_tiles(self, tiles, predictions: np.ndarray) -> None:
        n_tiles = max(len(tiles), 1)
        self._model.eval()
        with torch.no_grad():
            for i, tile in enumerate(tiles):
                if self.is_canceled_cb():
                    return
                edge_index, _ = dp.build_knn_graph(tile.xy, k=self.k)
                x = torch.from_numpy(tile.values[:, None]).float().to(self.device)
                pos = torch.from_numpy(tile.xy).float().to(self.device)
                ei = torch.from_numpy(edge_index).long().to(self.device)
                try:
                    out = self._forward(x, pos, ei)
                except Exception as exc:
                    self.warning_cb(f"Échec sur la tuile {i + 1}/{n_tiles} : {exc}. Tuile ignorée.")
                    continue
                out_np = out.detach().cpu().numpy()
                if out_np.ndim > 1:
                    out_np = out_np[:, 0]
                core = tile.core_indices
                gidx = tile.global_indices
                predictions[gidx] = out_np[core]
                self.progress_cb(100.0 * (i + 1) / n_tiles)

    def _forward(self, x, pos, edge_index):
        model = self._model
        try:
            return model(x=x, pos=pos, edge_index=edge_index)
        except TypeError:
            pass
        try:
            return model(x, pos, edge_index)
        except TypeError:
            pass
        try:
            return model(x, pos)
        except TypeError:
            pass
        try:
            return model(input_geom=pos, output_queries=pos, x=x)
        except TypeError:
            return model(x)

    def _load_model(self) -> None:
        if self._model is None:
            self._model = load_model(self.model_path, device=self.device)
            self.message_cb(f"Modèle chargé sur {self.device}.")

    def _setup_norm(self, values: np.ndarray) -> None:
        stats = dp.load_norm_stats_beside_model(self.model_path)
        if stats is not None:
            self._norm = stats
            self.message_cb(
                f"Normalisation globale chargée (vmin={stats.vmin:.4g}, vmax={stats.vmax:.4g})."
            )
        else:
            self._norm = dp.compute_norm_stats(values)
            self.warning_cb(
                "Aucun fichier .norm.json trouvé à côté du modèle. "
                "Normalisation min-max locale calculée sur l'entrée courante "
                f"(vmin={self._norm.vmin:.4g}, vmax={self._norm.vmax:.4g})."
            )
