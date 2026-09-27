# -*- coding: utf-8 -*-
"""
Filtres de fichiers et validation des extensions.

Source unique de vérité pour :
  - les filtres QFileDialog / QgsProcessingParameterFile
  - la détection du type d'entrée (raster vs point cloud)
  - la validation des chemins avant lancement
"""

from __future__ import annotations

import os
from typing import FrozenSet, Optional, Tuple

# ---------------------------------------------------------------------------
# Extensions (minuscules, avec le point)
# ---------------------------------------------------------------------------

RASTER_EXTENSIONS: FrozenSet[str] = frozenset({
    ".tif", ".tiff", ".gtiff",
    ".img", ".vrt", ".jp2", ".j2k",
    ".asc", ".bil", ".bsq", ".bip",
    ".nc", ".hdf", ".h5",
})

POINTCLOUD_EXTENSIONS: FrozenSet[str] = frozenset({
    ".las", ".laz",
})

MODEL_EXTENSIONS: FrozenSet[str] = frozenset({
    ".pt", ".pth", ".ckpt",
})

# Extensions préférées pour l'écriture (auto-complétion)
DEFAULT_RASTER_OUT_EXT = ".tif"
DEFAULT_POINTCLOUD_OUT_EXT = ".las"
DEFAULT_MODEL_OUT_EXT = ".pt"

# ---------------------------------------------------------------------------
# Chaînes de filtre (Qt / QGIS Processing)
# ---------------------------------------------------------------------------

# QFileDialog / Processing : syntaxe "Label (*.ext1 *.ext2);;Autre (*.*)"
FILTER_RASTER_OPEN = (
    "Rasters (*.tif *.tiff *.img *.vrt *.jp2 *.asc);;"
    "GeoTIFF (*.tif *.tiff);;"
    "Tous les fichiers (*.*)"
)
FILTER_RASTER_SAVE = (
    "GeoTIFF (*.tif);;"
    "GeoTIFF compressé (*.tif);;"
    "Tous les fichiers (*.*)"
)
FILTER_POINTCLOUD_OPEN = (
    "Nuages de points LAS/LAZ (*.las *.laz);;"
    "LAS (*.las);;"
    "LAZ (*.laz);;"
    "Tous les fichiers (*.*)"
)
FILTER_POINTCLOUD_SAVE = (
    "LAS (*.las);;"
    "LAZ compressé (*.laz);;"
    "Tous les fichiers (*.*)"
)
FILTER_MODEL_OPEN = (
    "Modèles PyTorch (*.pt *.pth *.ckpt);;"
    "PyTorch (*.pt *.pth);;"
    "Checkpoint (*.ckpt);;"
    "Tous les fichiers (*.*)"
)
FILTER_MODEL_SAVE = (
    "PyTorch (*.pt);;"
    "PyTorch (*.pth);;"
    "Tous les fichiers (*.*)"
)

# Variantes Processing (souvent plus simples)
PROC_FILTER_POINTCLOUD = "LAS/LAZ (*.las *.laz);;Tous les fichiers (*.*)"
PROC_FILTER_MODEL = "PyTorch (*.pt *.pth *.ckpt);;Tous les fichiers (*.*)"
PROC_FILTER_MODEL_OUT = "PyTorch (*.pt *.pth)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extension(path: str) -> str:
    """Retourne l'extension en minuscules (avec le point), ou ''."""
    if not path:
        return ""
    return os.path.splitext(path)[1].lower()


def is_raster_path(path: str) -> bool:
    return extension(path) in RASTER_EXTENSIONS


def is_pointcloud_path(path: str) -> bool:
    return extension(path) in POINTCLOUD_EXTENSIONS


def is_model_path(path: str) -> bool:
    return extension(path) in MODEL_EXTENSIONS


def detect_input_type(path: str) -> str:
    """
    'raster' | 'pointcloud' | 'unknown'
    """
    ext = extension(path)
    if ext in POINTCLOUD_EXTENSIONS:
        return "pointcloud"
    if ext in RASTER_EXTENSIONS:
        return "raster"
    return "unknown"


def ensure_extension(path: str, default_ext: str) -> str:
    """
    Ajoute default_ext si le chemin n'a aucune extension connue
    pour ce type de sortie. Ne remplace pas une extension déjà présente.
    """
    if not path:
        return path
    ext = extension(path)
    if ext:
        return path
    if not default_ext.startswith("."):
        default_ext = "." + default_ext
    return path + default_ext


def validate_existing_file(path: str, label: str = "fichier") -> Optional[str]:
    """
    Retourne un message d'erreur, ou None si OK.
    """
    if not path or not str(path).strip():
        return f"Chemin du {label} manquant."
    path = str(path).strip()
    if not os.path.isfile(path):
        return f"{label.capitalize()} introuvable : {path}"
    return None


def validate_model_file(path: str) -> Optional[str]:
    err = validate_existing_file(path, "modèle")
    if err:
        return err
    if not is_model_path(path):
        return (
            f"Extension de modèle non reconnue ({extension(path) or 'aucune'}). "
            f"Attendu : {', '.join(sorted(MODEL_EXTENSIONS))}."
        )
    return None


def validate_input_path(
    path: str,
    expected: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Valide un chemin d'entrée.

    Parameters
    ----------
    path : str
    expected : 'raster' | 'pointcloud' | None
        Si fourni, refuse un type différent.

    Returns
    -------
    (error_message or None, detected_type)
    """
    err = validate_existing_file(path, "fichier d'entrée")
    if err:
        return err, "unknown"

    detected = detect_input_type(path)
    if detected == "unknown":
        return (
            f"Extension non reconnue ({extension(path) or 'aucune'}). "
            f"Raster : {', '.join(sorted(RASTER_EXTENSIONS)[:6])}… ; "
            f"nuage de points : {', '.join(sorted(POINTCLOUD_EXTENSIONS))}.",
            "unknown",
        )

    if expected is not None and detected != expected:
        return (
            f"Type d'entrée incorrect : attendu {expected}, "
            f"détecté {detected} ({extension(path)}).",
            detected,
        )
    return None, detected


def validate_output_path(
    path: str,
    kind: str,
) -> Tuple[Optional[str], str]:
    """
    Valide / normalise un chemin de sortie.

    kind : 'raster' | 'pointcloud' | 'model'

    Returns
    -------
    (error_message or None, normalized_path)
    """
    if not path or not str(path).strip():
        return "Chemin de sortie manquant.", path or ""

    path = str(path).strip()
    defaults = {
        "raster": DEFAULT_RASTER_OUT_EXT,
        "pointcloud": DEFAULT_POINTCLOUD_OUT_EXT,
        "model": DEFAULT_MODEL_OUT_EXT,
    }
    allowed = {
        "raster": RASTER_EXTENSIONS,
        "pointcloud": POINTCLOUD_EXTENSIONS,
        "model": MODEL_EXTENSIONS,
    }

    if kind not in defaults:
        return f"Type de sortie inconnu : {kind}", path

    path = ensure_extension(path, defaults[kind])
    ext = extension(path)

    if kind == "raster":
        # On autorise toute extension raster, mais on recommande .tif
        if ext and ext not in allowed[kind]:
            # Avertissement soft : on laisse passer (GDAL peut gérer),
            # sauf extensions clairement point-cloud / model
            if ext in POINTCLOUD_EXTENSIONS or ext in MODEL_EXTENSIONS:
                return (
                    f"Extension de sortie incompatible pour un raster : {ext}.",
                    path,
                )
    elif kind == "pointcloud":
        if ext not in POINTCLOUD_EXTENSIONS:
            return (
                f"Extension de sortie LAS/LAZ invalide ({ext or 'aucune'}). "
                f"Utilisez .las ou .laz.",
                path,
            )
    elif kind == "model":
        if ext not in MODEL_EXTENSIONS:
            return (
                f"Extension de modèle invalide ({ext or 'aucune'}). "
                f"Utilisez .pt ou .pth.",
                path,
            )

    # Répertoire parent écritable ?
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            return f"Impossible de créer le dossier de sortie : {exc}", path

    return None, path


def validate_data_dir(path: str) -> Optional[str]:
    """Valide un dossier de données d'entraînement."""
    if not path or not str(path).strip():
        return "Dossier de données manquant."
    path = str(path).strip()
    if not os.path.isdir(path):
        return f"Dossier introuvable : {path}"
    # Au moins un GeoTIFF ?
    try:
        entries = os.listdir(path)
    except OSError as exc:
        return f"Impossible de lire le dossier : {exc}"
    tifs = [f for f in entries if extension(f) in {".tif", ".tiff"}]
    if not tifs:
        return (
            f"Aucun GeoTIFF (.tif/.tiff) trouvé dans {path}. "
            "Attendu : paires *_input.tif / *_target.tif."
        )
    return None
