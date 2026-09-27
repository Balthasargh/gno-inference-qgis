# -*- coding: utf-8 -*-
"""
Moteur d'entraînement GNO — SEULE implémentation de la boucle
d'entraînement (utilisée par le script CLI et l'algorithme Processing).
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GNODataset(Dataset):
    """Dataset de paires de rasters alignés (input / target)."""

    def __init__(self, pairs, k=8, norm_input=None, norm_target=None):
        self.pairs = pairs
        self.k = k
        self.norm_input = norm_input
        self.norm_target = norm_target

    @property
    def _dp(self):
        try:
            from ..utils import data_preparation as dp
            return dp
        except ImportError:
            from utils import data_preparation as dp
            return dp

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dp = self._dp
        in_path, tgt_path = self.pairs[idx]
        cloud_in, meta = dp.read_raster_as_points(in_path)
        cloud_tgt, _ = dp.read_raster_as_points(tgt_path)
        n = min(len(cloud_in.values), len(cloud_tgt.values))
        xy = cloud_in.xy[:n]
        v_in = cloud_in.values[:n]
        v_tgt = cloud_tgt.values[:n]
        if self.norm_input is not None:
            v_in = self.norm_input.normalize(v_in)
        if self.norm_target is not None:
            v_tgt = self.norm_target.normalize(v_tgt)
        edge_index, _ = dp.build_knn_graph(xy, k=self.k)
        return {
            "x": torch.from_numpy(v_in[:, None]).float(),
            "y": torch.from_numpy(v_tgt[:, None]).float(),
            "pos": torch.from_numpy(xy).float(),
            "edge_index": torch.from_numpy(edge_index).long(),
        }


def _collate_single(batch):
    return batch[0]


class GNOTrainingEngine:
    """Boucle d'entraînement GINO sur paires de rasters."""

    def __init__(
        self, data_dir, output_path, epochs=100, lr=1e-3, k=8, batch_size=1,
        val_split=0.15, patience=15, seed=42, device_str="cpu",
        progress_cb=None, message_cb=None, warning_cb=None, is_canceled_cb=None,
    ):
        self.data_dir = data_dir
        self.output_path = output_path
        self.epochs = epochs
        self.lr = lr
        self.k = k
        self.batch_size = max(1, batch_size)
        self.val_split = val_split
        self.patience = patience
        self.seed = seed
        self.device_str = device_str
        self.progress_cb = progress_cb or (lambda p: None)
        self.message_cb = message_cb or (lambda m: None)
        self.warning_cb = warning_cb or (lambda w: None)
        self.is_canceled_cb = is_canceled_cb or (lambda: False)

    @property
    def _ml(self):
        try:
            from ..utils.model_loader import default_build_fn, resolve_device
            return default_build_fn, resolve_device
        except ImportError:
            from utils.model_loader import default_build_fn, resolve_device
            return default_build_fn, resolve_device

    @property
    def _dp(self):
        try:
            from ..utils import data_preparation as dp
            return dp
        except ImportError:
            from utils import data_preparation as dp
            return dp

    def run(self) -> str:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        default_build_fn, resolve_device = self._ml
        dp = self._dp
        device = resolve_device(self.device_str)

        pairs = self._discover_pairs()
        if len(pairs) < 2:
            raise ValueError(f"Pas assez de paires dans {self.data_dir} (trouvé {len(pairs)}).")

        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * self.val_split))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]
        if not train_pairs:
            train_pairs = val_pairs
            val_pairs = []

        self.message_cb(f"{len(train_pairs)} scène(s) train, {len(val_pairs)} validation.")

        all_in, all_tgt = [], []
        for ip, tp in train_pairs:
            c_in, _ = dp.read_raster_as_points(ip)
            c_tgt, _ = dp.read_raster_as_points(tp)
            all_in.append(c_in.values)
            all_tgt.append(c_tgt.values)
        norm_in = dp.compute_norm_stats(np.concatenate(all_in))
        norm_tgt = dp.compute_norm_stats(np.concatenate(all_tgt))

        train_ds = GNODataset(train_pairs, k=self.k, norm_input=norm_in, norm_target=norm_tgt)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=_collate_single)

        val_loader = None
        if val_pairs:
            val_ds = GNODataset(val_pairs, k=self.k, norm_input=norm_in, norm_target=norm_tgt)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate_single)

        model = default_build_fn(in_channels=1, out_channels=1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            if self.is_canceled_cb():
                self.message_cb("Entraînement annulé.")
                break

            model.train()
            train_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pos = batch["pos"].to(device)
                ei = batch["edge_index"].to(device)
                optimizer.zero_grad()
                pred = self._forward(model, x, pos, ei)
                if pred.shape != y.shape:
                    pred = pred.view_as(y)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(n_batches, 1)

            val_loss = train_loss
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                n_val_b = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["x"].to(device)
                        y = batch["y"].to(device)
                        pos = batch["pos"].to(device)
                        ei = batch["edge_index"].to(device)
                        pred = self._forward(model, x, pos, ei)
                        if pred.shape != y.shape:
                            pred = pred.view_as(y)
                        val_loss += criterion(pred, y).item()
                        n_val_b += 1
                val_loss /= max(n_val_b, 1)

            scheduler.step(val_loss)
            self.message_cb(
                f"Epoch {epoch}/{self.epochs} — train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )
            self.progress_cb(100.0 * epoch / self.epochs)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    self.message_cb(f"Arrêt anticipé (patience={self.patience}).")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)) or ".", exist_ok=True)
        torch.save(model, self.output_path)
        norm_path = self.output_path + ".norm.json"
        norm_tgt.save(norm_path)
        self.message_cb(f"Modèle sauvegardé : {self.output_path}")
        self.message_cb(f"Stats de normalisation : {norm_path}")
        self.progress_cb(100.0)
        return self.output_path

    def _forward(self, model, x, pos, edge_index):
        try:
            return model(x=x, pos=pos, edge_index=edge_index)
        except TypeError:
            pass
        try:
            return model(x, pos, edge_index)
        except TypeError:
            pass
        try:
            return model(input_geom=pos, output_queries=pos, x=x)
        except TypeError:
            return model(x)

    def _discover_pairs(self):
        files = sorted(os.listdir(self.data_dir))
        tifs = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        pairs = []
        inputs = [f for f in tifs if "_input" in f.lower()]
        for inp in inputs:
            base = inp.lower().replace("_input", "_target")
            tgt = next((t for t in tifs if t.lower() == base), None)
            if tgt:
                pairs.append((
                    os.path.join(self.data_dir, inp),
                    os.path.join(self.data_dir, tgt),
                ))
        if not pairs:
            tifs_full = [os.path.join(self.data_dir, f) for f in tifs]
            for i in range(0, len(tifs_full) - 1, 2):
                pairs.append((tifs_full[i], tifs_full[i + 1]))
        return pairs
