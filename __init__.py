# -*- coding: utf-8 -*-
"""
Point d'entrée du plugin GNO Inference pour QGIS.
QGIS appelle classFactory() au chargement du plugin.
"""


def classFactory(iface):
    from .gno_plugin import GNOPlugin
    return GNOPlugin(iface)
