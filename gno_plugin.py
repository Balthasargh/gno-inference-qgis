# -*- coding: utf-8 -*-
"""
Classe principale du plugin GNO Inference.

Deux points d'entrée pour l'utilisateur, partageant le même moteur
(core/gno_inference.py) :
  1. Un algorithme de la boîte à outils de Traitement (Processing) --
     pour l'automatisation / le traitement par lot.
  2. Un panneau dockable interactif -- pour explorer un modèle sur un
     raster ou un nuage de points LAS/LAZ sans configurer un run
     Processing complet. Avant sa première ouverture, les dépendances
     Python de base (torch, neuraloperator, rasterio, scipy, numpy) sont
     vérifiées et une installation assistée est proposée si besoin ;
     laspy (support LAS/LAZ) est vérifié séparément, seulement si
     l'utilisateur choisit une entrée/sortie nuage de points.
"""

import os

from qgis.core import QgsApplication
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox

from .processing.gno_provider import GNOProvider


class GNOPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.action_dock = None
        self.action_about = None
        self.dock_widget = None
        self.plugin_dir = os.path.dirname(__file__)

    # ------------------------------------------------------------------
    # Cycle de vie du plugin
    # ------------------------------------------------------------------
    def initGui(self):
        """Appelé par QGIS au chargement du plugin."""
        self.initProcessing()

        icon_path = os.path.join(self.plugin_dir, "resources", "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()

        self.action_dock = QAction(icon, "GNO Inference (panneau)", self.iface.mainWindow())
        self.action_dock.triggered.connect(self.toggle_dock)
        self.iface.addPluginToMenu("&GNO Inference", self.action_dock)
        self.iface.addToolBarIcon(self.action_dock)

        self.action_about = QAction("GNO Inference — À propos", self.iface.mainWindow())
        self.action_about.triggered.connect(self.show_about)
        self.iface.addPluginToMenu("&GNO Inference", self.action_about)

    def initProcessing(self):
        """Enregistre le fournisseur d'algorithmes GNO dans Processing."""
        self.provider = GNOProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        """Appelé par QGIS quand le plugin est désactivé/déchargé."""
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None

        if self.dock_widget is not None:
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None

        if self.action_dock is not None:
            self.iface.removePluginMenu("&GNO Inference", self.action_dock)
            self.iface.removeToolBarIcon(self.action_dock)
            self.action_dock = None

        if self.action_about is not None:
            self.iface.removePluginMenu("&GNO Inference", self.action_about)
            self.action_about = None

    # ------------------------------------------------------------------
    # Panneau dockable
    # ------------------------------------------------------------------
    def toggle_dock(self):
        from .utils.dependencies import missing_dependencies

        missing = missing_dependencies()
        if missing:
            from .ui.dependency_dialog import DependencyInstallDialog
            dlg = DependencyInstallDialog(missing, parent=self.iface.mainWindow())
            dlg.exec_()
            # On ne bloque pas l'ouverture du dock après coup : l'utilisateur
            # peut relancer l'action une fois l'installation terminée et
            # QGIS redémarré.
            return

        if self.dock_widget is None:
            from .ui.gno_dock_widget import GNODockWidget
            self.dock_widget = GNODockWidget(self.iface, parent=self.iface.mainWindow())
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
        else:
            self.dock_widget.setVisible(not self.dock_widget.isVisible())

    # ------------------------------------------------------------------
    # UI auxiliaire
    # ------------------------------------------------------------------
    def show_about(self):
        QMessageBox.information(
            self.iface.mainWindow(),
            "GNO Inference",
            "Deux façons d'utiliser GNO Inference :\n\n"
            "1. Traitement → Boîte à outils → GNO Inference → Inférence "
            "GNO sur raster (pour l'automatisation / le traitement par "
            "lot).\n\n"
            "2. Le panneau dockable GNO Inference (icône dans la barre "
            "d'outils / menu Extensions), pour une utilisation "
            "interactive : charge un raster (ou un nuage de points "
            "LAS/LAZ) et un modèle, ajuste k et la taille de tuile, "
            "lance et visualise directement le résultat.",
        )
