# GNO Inference — Plugin QGIS

Plugin QGIS encapsulant un modèle **Graph Neural Operator (GNO / GINO)**
pré-entraîné en PyTorch, exposé comme algorithme de la boîte à outils de
**Traitement (Processing)**. Fonctionne sur un **raster** (DEM,
imagerie...) ou directement sur un **nuage de points LAS/LAZ**.

## Fonctionnement

1. L'entrée est convertie en nuage de points `(x, y, valeur)` :
   - **Raster** : une valeur par pixel valide (grille régulière).
   - **Nuage de points LAS/LAZ** : une dimension LAS au choix (`z` par
     défaut, ou `intensity`, `classification`, une extra dimension...).
2. Les valeurs sont normalisées (min-max).
3. L'entrée est découpée en tuiles avec recouvrement, chaque tuile
   reconstruisant son propre graphe de voisinage k-NN
   (`scipy.spatial.cKDTree`) :
   - **Raster** : tuiling par bandes de lignes (exploite la grille
     régulière).
   - **Nuage de points** : tuiling spatial 2D générique par grille de
     cellules (pas de grille implicite dans un nuage LAS/LAZ).
4. Le modèle GNO (`.pt` / `.pth`) est chargé et exécuté par lots, sur CPU
   ou GPU.
5. Les prédictions sont dénormalisées et réécrites :
   - **Raster** : GeoTIFF, avec la même emprise, résolution et CRS que
     l'entrée.
   - **Nuage de points** : LAS/LAZ identique à l'entrée (même en-tête,
     mêmes dimensions), avec une dimension supplémentaire portant les
     prédictions (`gno_prediction`).

## Installation

### 1. Dépendances Python (dans l'environnement de QGIS)

Sous Windows (OSGeo4W Shell) :

```
python -m pip install torch neuraloperator rasterio scipy numpy
```

Sous Linux/Mac, utilise le Python de QGIS (souvent `qgis-python` ou le
Python système si QGIS l'utilise) :

```
python3 -m pip install torch neuraloperator rasterio scipy numpy
```

> Pour une installation GPU (CUDA), installe torch depuis
> https://pytorch.org/get-started/locally/ avec la commande adaptée à ta
> configuration, avant les autres dépendances.

Optionnel — uniquement si tu utilises une entrée/sortie **nuage de points
LAS/LAZ** :

```
python -m pip install laspy
# Pour les fichiers .laz compressés, ajoute un backend de (dé)compression :
python -m pip install lazrs
```

Ces paquets optionnels ne sont vérifiés/installés que lorsque tu choisis
un nuage de points dans le panneau dockable (dialogue de dépendances
dédié) — un usage 100% raster n'a pas besoin de `laspy`.

### 2. Installation du plugin

1. Compresse le dossier `GNO_Plugin/` en `GNO_Plugin.zip`
   (le zip doit contenir directement `metadata.txt`, `__init__.py`, etc.
   à sa racine, pas un sous-dossier supplémentaire).
2. Dans QGIS : **Extensions → Gérer et installer les extensions →
   Installer depuis un ZIP**.
3. Sélectionne `GNO_Plugin.zip`, installe, puis active le plugin.

## Utilisation

1. **Traitement → Boîte à outils**.
2. Cherche **GNO Inference → Inférence GNO sur raster / nuage de points**.
3. Renseigne SOIT une entrée raster, SOIT une entrée nuage de points (pas
   les deux) :
   - **Raster d'entrée** + **Raster de sortie** (GeoTIFF) ; ou
   - **Nuage de points d'entrée** (.las/.laz) + **Dimension à utiliser**
     (`z` par défaut) + **Nuage de points de sortie** (.las/.laz).
4. Renseigne les paramètres communs :
   - **Fichier du modèle** : ton `.pt` / `.pth` entraîné
   - **k** : nombre de voisins du graphe (8 par défaut)
   - **Taille de lot** : nombre de points coeur traités par tuile (4096
     par défaut)
   - **Device** : CPU ou GPU
5. Lance l'algorithme. La progression et les messages s'affichent dans le
   panneau de Traitement.

Le panneau dockable interactif (icône dans la barre d'outils) propose la
même chose avec un sélecteur **Type d'entrée : Raster / Nuage de points**,
sans passer par la boîte à outils de Traitement.

## Entraîner un modèle

Deux façons d'entraîner, sur le **même moteur** (`core/gno_training.py`,
classe `GNOTrainingEngine`), pour éviter toute divergence entre les deux :

- **Depuis QGIS** : **Traitement → Boîte à outils → GNO Inference →
  Entraînement GNO**. Pratique pour itérer rapidement sans quitter QGIS ;
  l'entraînement peut être long, et se lance comme n'importe quel
  algorithme de Traitement (progression et messages dans le panneau,
  annulation coopérative entre deux epochs).
- **Script autonome** `scripts/train_gno.py`, à exécuter **en dehors de
  QGIS**, dans un environnement Python dédié (idéalement avec GPU) — plus
  adapté à un entraînement intensif :

```
pip install -r scripts/requirements-train.txt
```

### Format des données

Un dossier avec des paires de rasters GeoTIFF alignés (même grille, CRS,
résolution, nodata) :

```
data/
├── scene_001_input.tif
├── scene_001_target.tif
├── scene_002_input.tif
├── scene_002_target.tif
└── ...
```

### Lancer l'entraînement (script autonome)

```
python scripts/train_gno.py \
    --data-dir ./data \
    --epochs 100 \
    --lr 1e-3 \
    --k 8 \
    --batch-size 4 \
    --val-split 0.15 \
    --device gpu \
    --output ./gno_model.pt
```

### Lancer l'entraînement (algorithme Processing)

Mêmes paramètres, sous forme de champs dans la boîte à outils : **Dossier
de données**, **Nombre d'epochs**, **Taux d'apprentissage**, **k**,
**Taille de lot**, **Proportion de scènes en validation**, **Patience**,
**Graine aléatoire**, **Device**, **Modèle de sortie (.pt)**.

Dans les deux cas, l'entraînement :
- découpe automatiquement train/validation,
- calcule les statistiques de normalisation globales sur le train,
- construit un graphe k-NN par scène,
- entraîne avec Adam + `ReduceLROnPlateau`, arrêt anticipé (patience),
- sauvegarde le meilleur modèle (objet complet — compatible directement
  avec « Inférence GNO sur raster / nuage de points ») ainsi qu'un fichier
  `<modèle>.norm.json`.

⚠️ **Point d'attention normalisation** : à l'inférence, le plugin utilise
en priorité les statistiques globales de `<model_path>.norm.json` si ce
fichier existe à côté du modèle (généré par `train_gno.py`) ; à défaut, il
recalcule un min/max *local* sur l'entrée courante (raster ou nuage de
points), avec un avertissement explicite, ce qui peut différer légèrement
du comportement à l'entraînement.

## Model Zoo

Le panneau dockable propose un bouton **« Model Zoo... »** à côté du champ
« Modèle », qui ouvre une boîte de dialogue listant des modèles GNO
pré-entraînés téléchargeables (mécanisme : `utils/model_zoo.py` +
`ui/model_zoo_dialog.py`).

Le registre est un fichier JSON simple :

```json
{
  "schema_version": 1,
  "models": [
    {
      "id": "dem_correction_v1",
      "name": "Correction de DEM -- v1",
      "description": "Corrige les artefacts d'un MNT LiDAR brut.",
      "task": "raster",
      "url": "https://.../dem_correction_v1.pt",
      "sha256": "…",
      "recommended_k": 8,
      "recommended_batch_size": 4096,
      "license": "CC-BY-4.0",
      "tags": ["dem", "lidar"]
    }
  ]
}
```

Champs obligatoires : `id`, `name`, `task` (`raster` / `pointcloud` /
`both`), `url`. Le `sha256` est optionnel mais fortement recommandé : s'il
est renseigné, le téléchargement est rejeté (et le fichier partiel
supprimé) si le hash ne correspond pas.

- Le registre **par défaut** est `resources/model_zoo.json`, **vide**
  (`"models": []`) : ce plugin ne distribue aucun modèle GNO pré-entraîné
  public. Le mécanisme (liste, téléchargement en arrière-plan avec
  progression et annulation, vérification sha256, cache local, sélection
  directe dans le panneau) est pleinement fonctionnel dès maintenant.
- Pour proposer des modèles : héberge les fichiers `.pt`/`.pth` quelque
  part en HTTPS (par exemple les assets d'une release GitHub du dépôt du
  plugin) et complète `resources/model_zoo.json` avec les entrées
  correspondantes.
- Pour un usage sans republier le plugin : renseigne l'URL (ou le chemin
  local) d'un registre alternatif dans le champ en haut de la boîte de
  dialogue Model Zoo, au même schéma JSON, puis « Rafraîchir ».

Les modèles téléchargés sont mis en cache dans
`<profil QGIS>/gno_inference/model_zoo/` et ne sont retéléchargés que si
le fichier est absent ou que son sha256 ne correspond plus à celui du
registre.

## Adapter le modèle

Par défaut, `utils/model_loader.py` sait charger :
- un modèle PyTorch complet sauvegardé avec `torch.save(model, path)` ;
- un `state_dict` (brut ou dans un dict `{"state_dict": ...}`), auquel cas
  il reconstruit l'architecture via `default_build_fn()`
  (un `GINO` de `neuraloperator` avec des hyperparamètres par défaut).

Si ton modèle a une architecture différente, adapte `default_build_fn()`
dans `utils/model_loader.py`, ou passe ton propre `build_fn` à
`load_model(...)`.

De même, si la signature `forward()` de ton modèle diffère de
`model(x=..., pos=..., edge_index=...)`, adapte l'appel dans
`core/gno_inference.py` (`GNOInferenceEngine._infer_on_tiles`).

## Structure du projet

```
GNO_Plugin/
├── __init__.py
├── metadata.txt
├── gno_plugin.py            # point d'entrée : dock + Processing + dépendances
├── core/
│   ├── gno_inference.py     # moteur d'inférence partagé (SEULE implémentation
│   │                         # du pipeline entrée -> tuiles -> GNO -> sortie,
│   │                         # pour un raster comme pour un nuage de points)
│   └── gno_training.py      # moteur d'entraînement partagé (SEULE implémentation
│                             # de la boucle d'entraînement, utilisée par le script
│                             # autonome et par l'algorithme Processing)
├── processing/
│   ├── gno_algorithm.py       # algorithme Processing d'inférence (fin wrapper)
│   ├── gno_train_algorithm.py # algorithme Processing d'entraînement (fin wrapper)
│   ├── gno_processor.py       # QThread (usage interactif, pattern Deepness)
│   └── gno_provider.py
├── ui/
│   ├── gno_dock_widget.py    # panneau dockable (pattern GeoAI), avec accès au Model Zoo
│   ├── model_zoo_dialog.py   # boîte de dialogue Model Zoo (liste, téléchargement, cache)
│   └── dependency_dialog.py  # installateur pip assisté
├── resources/
│   ├── icon.png   (à fournir, optionnel)
│   └── model_zoo.json        # registre par défaut des modèles pré-entraînés (vide par défaut)
├── scripts/
│   ├── train_gno.py                # fin wrapper CLI sur core/gno_training.py
│   └── requirements-train.txt
├── utils/
│   ├── data_preparation.py   # lecture raster / LAS-LAZ, tiling, k-NN, normalisation
│   ├── model_loader.py
│   ├── model_zoo.py           # registre, cache, téléchargement + vérification sha256
│   ├── export_results.py     # écriture GeoTIFF ou LAS/LAZ
│   └── dependencies.py       # détection + installation des dépendances
└── README.md
```

### Architecture (v0.3)

Un seul moteur (`core/gno_inference.py`, classe `GNOInferenceEngine`) porte
toute la logique métier, pour les deux sources d'entrée supportées
(sélection automatique sur l'extension du fichier d'entrée) :

- **Raster** : lecture raster → nuage de points → **tiling par bandes de
  lignes avec recouvrement** (`dp.iter_row_tiles`, adapté à une grille
  régulière) → inférence → écriture GeoTIFF.
- **Nuage de points (.las/.laz)** : lecture LAS/LAZ (`laspy`) → **tiling
  spatial 2D générique par grille de cellules avec recouvrement**
  (`dp.iter_spatial_tiles`, car un nuage de points n'a pas de structure de
  grille implicite) → inférence → écriture LAS/LAZ (nouvelle dimension de
  prédiction, en-tête et autres dimensions préservés).

Les deux chemins partagent la même boucle d'inférence par tuiles
(`GNOInferenceEngine._infer_on_tiles`), pour éviter exactement le type de
bug déjà rencontré : deux implémentations légèrement différentes du même
pipeline qui finissent par diverger. Le moteur communique par callbacks
simples (`progress_cb`, `message_cb`, `warning_cb`, `is_canceled_cb`),
sans dépendre de Qt ni de Processing.

Deux points d'entrée s'appuient dessus sans dupliquer le pipeline :
- `processing/gno_algorithm.py` : branche les callbacks sur
  `QgsProcessingFeedback` (pour la boîte à outils de Traitement), et
  expose des paramètres raster et nuage de points mutuellement exclusifs.
- `processing/gno_processor.py` (`GNOProcessor`, un `QThread`) : branche
  les callbacks sur des signaux Qt, pour le panneau dockable
  (`ui/gno_dock_widget.py`), sans geler l'interface pendant l'inférence
  et avec annulation coopérative.

De même côté entraînement, un seul moteur (`core/gno_training.py`, classe
`GNOTrainingEngine`) porte toute la boucle d'entraînement (dataset ->
graphe k-NN par scène -> GINO -> MSE -> Adam/`ReduceLROnPlateau` -> arrêt
anticipé), avec les mêmes callbacks simples que le moteur d'inférence
(`progress_cb`, `message_cb`, `warning_cb`, `is_canceled_cb`, vérifié
entre deux epochs). Deux points d'entrée s'appuient dessus :
- `scripts/train_gno.py` : wrapper CLI (`argparse`), branche les
  callbacks sur `print`, pour un entraînement intensif hors QGIS.
- `processing/gno_train_algorithm.py` : branche les callbacks sur
  `QgsProcessingFeedback`, pour lancer l'entraînement depuis la boîte à
  outils de Traitement, sans quitter QGIS.

`core/gno_training.py` importe `utils/` en relatif (`from ..utils import
...`) quand il est chargé comme sous-package de `GNO_Plugin` (cas QGIS),
et retombe automatiquement sur un import absolu (`from utils import
...`) sinon (cas script autonome, qui insère la racine du plugin dans
`sys.path`) -- voir `GNODataset._dp` / `GNOTrainingEngine._ml`.

Avant la première ouverture du panneau, `utils/dependencies.py` vérifie
la présence de torch/neuraloperator/rasterio/scipy/numpy (dépendances
requises) ; si des paquets manquent, `ui/dependency_dialog.py` propose une
installation `pip install --user` directement depuis QGIS (journal en
direct, dans un thread séparé). `laspy` est vérifié séparément, comme
dépendance optionnelle, uniquement quand l'utilisateur choisit une
entrée/sortie nuage de points.

## Limites connues de cette version

- Traite un seul raster mono-bande à la fois (bande 1 par défaut) ; côté
  nuage de points, une seule dimension LAS à la fois comme feature
  d'entrée.
- L'entraînement (script ou algorithme Processing) n'accepte que des
  paires de rasters GeoTIFF alignés en entrée ; pas encore de variante
  nuage de points pour l'entraînement (l'inférence, elle, gère les deux).
- L'annulation d'un entraînement en cours n'est vérifiée qu'entre deux
  epochs, pas à l'intérieur d'une epoch.
- Le Model Zoo est fonctionnel mais son registre par défaut est vide
  (aucun modèle GNO pré-entraîné public distribué avec ce plugin) --
  voir la section « Model Zoo » pour publier ou pointer vers des modèles.
