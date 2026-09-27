# -*- coding: utf-8 -*-
"""Fournisseur d'algorithmes Processing pour GNO Inference."""

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os


class GNOProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        from .gno_algorithm import GNOInferenceAlgorithm
        from .gno_train_algorithm import GNOTrainAlgorithm

        self.addAlgorithm(GNOInferenceAlgorithm())
        self.addAlgorithm(GNOTrainAlgorithm())

    def id(self):
        return "gno_inference"

    def name(self):
        return "GNO Inference"

    def longName(self):
        return "GNO Inference — Graph Neural Operator"

    def icon(self):
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "resources",
            "icon.png",
        )
        return QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
