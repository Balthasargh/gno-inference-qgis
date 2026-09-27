# -*- coding: utf-8 -*-
"""
Chargement de modèles GNO / GINO sauvegardés en .pt / .pth.

Supporte :
  - un modèle complet (torch.save(model, path))
  - un state_dict brut ou encapsulé dans {"state_dict": ...}
    → reconstruction via default_build_fn() (GINO de neuraloperator)
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch


def default_build_fn(
    in_channels: int = 1,
    out_channels: int = 1,
    gno_coord_dim: int = 2,
    fno_n_modes=(16, 16),
    fno_hidden_channels: int = 64,
    in_gno_radius: float = 0.05,
    out_gno_radius: float = 0.05,
):
    """
    Construit un GINO par défaut (neuraloperator).

    Adapter les hyperparamètres ici si l'architecture de ton modèle diffère.
    """
    try:
        from neuralop.models import GINO
    except ImportError:
        raise ImportError(
            "neuraloperator n'est pas installé. "
            "Installez-le avec : pip install neuraloperator"
        )

    model = GINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=gno_coord_dim,
        fno_n_modes=fno_n_modes,
        fno_hidden_channels=fno_hidden_channels,
        in_gno_radius=in_gno_radius,
        out_gno_radius=out_gno_radius,
    )
    return model


def load_model(
    path: str,
    device: Optional[torch.device] = None,
    build_fn: Optional[Callable] = None,
    map_location: Optional[str] = None,
) -> torch.nn.Module:
    """
    Charge un modèle depuis un fichier .pt / .pth.

    Parameters
    ----------
    path : str
        Chemin vers le fichier modèle.
    device : torch.device or None
        Device cible (cpu / cuda). Si None, utilise map_location ou cpu.
    build_fn : callable or None
        Fonction de reconstruction de l'architecture si state_dict.
        Par défaut : default_build_fn.
    map_location : str or None
        Argument map_location de torch.load.

    Returns
    -------
    torch.nn.Module
        Modèle en mode eval().
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier modèle introuvable : {path}")

    if device is None:
        device = torch.device(map_location or "cpu")

    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        state = obj.get("state_dict", obj.get("model_state_dict", obj))
        if not isinstance(state, dict):
            raise ValueError(
                f"Format de checkpoint non reconnu dans {path}. "
                "Attendu : Module, state_dict, ou dict contenant 'state_dict'."
            )
        builder = build_fn or default_build_fn
        kwargs = {}
        if "in_channels" in obj:
            kwargs["in_channels"] = obj["in_channels"]
        if "out_channels" in obj:
            kwargs["out_channels"] = obj["out_channels"]
        model = builder(**kwargs)
        model.load_state_dict(state)
    else:
        raise ValueError(
            f"Objet chargé depuis {path} n'est ni un Module ni un dict "
            f"(type={type(obj)})."
        )

    model.to(device)
    model.eval()
    return model


def resolve_device(device_str: str) -> torch.device:
    """Convertit 'cpu' / 'gpu' / 'cuda' en torch.device."""
    device_str = (device_str or "cpu").lower().strip()
    if device_str in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")
