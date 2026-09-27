#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script autonome d'entraînement GNO.

Usage :
    python scripts/train_gno.py --data-dir ./data --epochs 100 --output ./gno_model.pt

À exécuter hors de QGIS, dans un environnement Python dédié (idéalement GPU).
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement d'un modèle GINO sur paires de rasters."
    )
    parser.add_argument("--data-dir", required=True, help="Dossier des paires input/target")
    parser.add_argument("--output", required=True, help="Chemin du modèle de sortie (.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if plugin_root not in sys.path:
        sys.path.insert(0, plugin_root)

    from core.gno_training import GNOTrainingEngine

    engine = GNOTrainingEngine(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
        k=args.k,
        batch_size=args.batch_size,
        val_split=args.val_split,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
        progress_cb=lambda p: print(f"  progression : {p:.1f} %"),
        message_cb=lambda m: print(m),
        warning_cb=lambda w: print(f"AVERTISSEMENT : {w}"),
        is_canceled_cb=lambda: False,
    )
    engine.run()


if __name__ == "__main__":
    main()
