# -*- coding: utf-8 -*-
"""Algorithme Processing d'inférence GNO (wrapper fin sur GNOInferenceEngine)."""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterString,
    QgsProcessingException,
)
from qgis.PyQt.QtCore import QCoreApplication

from ..utils.file_filters import (
    PROC_FILTER_MODEL,
    PROC_FILTER_POINTCLOUD,
    validate_input_path,
    validate_model_file,
    validate_output_path,
)


class GNOInferenceAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTER = "INPUT_RASTER"
    INPUT_POINTCLOUD = "INPUT_POINTCLOUD"
    LAS_DIMENSION = "LAS_DIMENSION"
    MODEL = "MODEL"
    K = "K"
    BATCH_SIZE = "BATCH_SIZE"
    DEVICE = "DEVICE"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    OUTPUT_POINTCLOUD = "OUTPUT_POINTCLOUD"

    def tr(self, string):
        return QCoreApplication.translate("GNOInferenceAlgorithm", string)

    def createInstance(self):
        return GNOInferenceAlgorithm()

    def name(self):
        return "gno_inference"

    def displayName(self):
        return self.tr("Inférence GNO sur raster / nuage de points")

    def group(self):
        return self.tr("GNO Inference")

    def groupId(self):
        return "gno_inference"

    def shortHelpString(self):
        return self.tr(
            "Applique un modèle Graph Neural Operator (GNO/GINO) pré-entraîné "
            "sur un raster (GeoTIFF) ou un nuage de points LAS/LAZ.\n\n"
            "Renseignez SOIT une entrée raster + sortie raster, "
            "SOIT une entrée nuage de points + sortie nuage de points.\n\n"
            "Extensions modèle : .pt, .pth, .ckpt\n"
            "Extensions nuage de points : .las, .laz"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER,
                self.tr("Raster d'entrée (optionnel)"),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_POINTCLOUD,
                self.tr("Nuage de points d'entrée LAS/LAZ (optionnel)"),
                optional=True,
                fileFilter=PROC_FILTER_POINTCLOUD,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.LAS_DIMENSION,
                self.tr("Dimension LAS à utiliser"),
                defaultValue="z",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.MODEL,
                self.tr("Fichier du modèle (.pt / .pth / .ckpt)"),
                fileFilter=PROC_FILTER_MODEL,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.K,
                self.tr("k (voisins du graphe)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=1,
                maxValue=64,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BATCH_SIZE,
                self.tr("Taille de lot (points cœur par tuile)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=4096,
                minValue=256,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEVICE,
                self.tr("Device"),
                options=["CPU", "GPU"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER,
                self.tr("Raster de sortie (si entrée raster)"),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_POINTCLOUD,
                self.tr("Nuage de points de sortie (si entrée LAS/LAZ)"),
                optional=True,
                fileFilter="LAS/LAZ (*.las *.laz)",
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        from ..core.gno_inference import GNOInferenceEngine

        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER, context)
        pc_path = self.parameterAsString(parameters, self.INPUT_POINTCLOUD, context)
        model_path = self.parameterAsString(parameters, self.MODEL, context)
        k = self.parameterAsInt(parameters, self.K, context)
        batch_size = self.parameterAsInt(parameters, self.BATCH_SIZE, context)
        device_idx = self.parameterAsEnum(parameters, self.DEVICE, context)
        device_str = "gpu" if device_idx == 1 else "cpu"
        las_dim = self.parameterAsString(parameters, self.LAS_DIMENSION, context) or "z"

        # --- Validation modèle ---
        err = validate_model_file(model_path)
        if err:
            raise QgsProcessingException(self.tr(err))

        # --- Choix entrée / sortie ---
        if raster_layer is not None and raster_layer.isValid():
            input_path = raster_layer.source()
            # Certaines sources QGIS sont des URI (vs chemin fichier)
            if input_path and not input_path.startswith("/"):
                # Windows drive ou URI GDAL — on laisse passer si le layer est valide
                pass
            output_path = self.parameterAsOutputLayer(
                parameters, self.OUTPUT_RASTER, context
            )
            if not output_path:
                raise QgsProcessingException(self.tr("Sortie raster non renseignée."))
            err, output_path = validate_output_path(output_path, "raster")
            if err:
                raise QgsProcessingException(self.tr(err))
            input_kind = "raster"
        elif pc_path:
            err, detected = validate_input_path(pc_path, expected="pointcloud")
            if err:
                raise QgsProcessingException(self.tr(err))
            input_path = pc_path
            output_path = self.parameterAsString(
                parameters, self.OUTPUT_POINTCLOUD, context
            )
            err, output_path = validate_output_path(output_path or "", "pointcloud")
            if err:
                raise QgsProcessingException(self.tr(err))
            input_kind = "pointcloud"
        else:
            raise QgsProcessingException(
                self.tr(
                    "Renseignez soit un raster d'entrée, "
                    "soit un nuage de points LAS/LAZ."
                )
            )

        engine = GNOInferenceEngine(
            model_path=model_path,
            k=k,
            batch_size=batch_size,
            device_str=device_str,
            progress_cb=lambda p: feedback.setProgress(p),
            message_cb=lambda m: feedback.pushInfo(m),
            warning_cb=lambda w: feedback.pushWarning(w),
            is_canceled_cb=lambda: feedback.isCanceled(),
        )

        try:
            result = engine.run(input_path, output_path, las_dimension=las_dim)
        except InterruptedError:
            raise QgsProcessingException(self.tr("Inférence annulée."))
        except Exception as exc:
            raise QgsProcessingException(str(exc))

        if input_kind == "raster":
            return {self.OUTPUT_RASTER: result}
        return {self.OUTPUT_POINTCLOUD: result}
