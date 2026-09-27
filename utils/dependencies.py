# -*- coding: utf-8 -*-
"""
Détection et installation assistée des dépendances Python du plugin.

Dépendances requises (toujours vérifiées) :
  torch, neuraloperator, rasterio, scipy, numpy

Dépendance optionnelle (vérifiée uniquement pour LAS/LAZ) :
  laspy  (+ lazrs recommandé pour la compression)
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import List

REQUIRED = [
    ("torch", "torch"),
    ("neuralop", "neuraloperator"),
    ("rasterio", "rasterio"),
    ("scipy", "scipy"),
    ("numpy", "numpy"),
]

OPTIONAL_LAS = [
    ("laspy", "laspy"),
]


def _is_available(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def missing_dependencies(include_optional: bool = False) -> List[str]:
    checks = list(REQUIRED)
    if include_optional:
        checks.extend(OPTIONAL_LAS)
    missing = []
    for import_name, pip_name in checks:
        if not _is_available(import_name):
            missing.append(pip_name)
    return missing


def missing_las_dependencies() -> List[str]:
    return [pip for imp, pip in OPTIONAL_LAS if not _is_available(imp)]


def install_packages(packages: List[str], user: bool = True, log_cb=None) -> int:
    if not packages:
        return 0
    cmd = [sys.executable, "-m", "pip", "install"]
    if user:
        cmd.append("--user")
    cmd.extend(packages)
    if log_cb:
        log_cb(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_cb:
                log_cb(line.rstrip("\n"))
        proc.wait()
        return proc.returncode
    except Exception as exc:
        if log_cb:
            log_cb(f"Erreur lors de l'installation : {exc}")
        return 1
