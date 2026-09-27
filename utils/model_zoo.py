# -*- coding: utf-8 -*-
"""
Model Zoo : registre JSON, cache local, téléchargement avec vérification sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class ZooModel:
    id: str
    name: str
    description: str = ""
    task: str = "both"
    url: str = ""
    sha256: str = ""
    recommended_k: int = 8
    recommended_batch_size: int = 4096
    license: str = ""
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ZooModel":
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            task=d.get("task", "both"),
            url=d.get("url", ""),
            sha256=d.get("sha256", ""),
            recommended_k=int(d.get("recommended_k", 8)),
            recommended_batch_size=int(d.get("recommended_batch_size", 4096)),
            license=d.get("license", ""),
            tags=list(d.get("tags", [])),
        )


def default_registry_path(plugin_dir: str) -> str:
    return os.path.join(plugin_dir, "resources", "model_zoo.json")


def cache_dir() -> str:
    try:
        from qgis.core import QgsApplication
        base = QgsApplication.qgisSettingsDirPath()
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".gno_inference")
    path = os.path.join(base, "gno_inference", "model_zoo")
    os.makedirs(path, exist_ok=True)
    return path


def load_registry(path: str) -> List[ZooModel]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", [])
    return [ZooModel.from_dict(m) for m in models]


def cached_path(model: ZooModel) -> str:
    ext = ".pt"
    if model.url:
        for e in (".pt", ".pth", ".ckpt"):
            if model.url.lower().endswith(e):
                ext = e
                break
    return os.path.join(cache_dir(), f"{model.id}{ext}")


def is_cached(model: ZooModel) -> bool:
    path = cached_path(model)
    if not os.path.isfile(path):
        return False
    if model.sha256:
        return _file_sha256(path) == model.sha256.lower()
    return True


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    model: ZooModel,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    is_canceled_cb: Optional[Callable[[], bool]] = None,
) -> str:
    dest = cached_path(model)
    if is_cached(model):
        return dest
    if not model.url:
        raise ValueError(f"Modèle {model.id} sans URL de téléchargement.")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".partial", dir=os.path.dirname(dest))
    os.close(tmp_fd)
    try:
        def _reporthook(block_num, block_size, total_size):
            if is_canceled_cb and is_canceled_cb():
                raise InterruptedError("Téléchargement annulé.")
            if progress_cb and total_size > 0:
                progress_cb(min(block_num * block_size, total_size), total_size)
        urllib.request.urlretrieve(model.url, tmp_path, reporthook=_reporthook)
        if model.sha256:
            digest = _file_sha256(tmp_path)
            if digest != model.sha256.lower():
                os.remove(tmp_path)
                raise ValueError(
                    f"Hash SHA256 incorrect pour {model.id} "
                    f"(attendu {model.sha256}, obtenu {digest})."
                )
        os.replace(tmp_path, dest)
        return dest
    except Exception:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
