# -*- coding: utf-8 -*-
"""Algorithme Processing d'entraînement GNO (wrapper fin sur GNOTrainingEngine)."""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingException,
)
from qgis.PyQt.QtCore import QCoreApplication


class GNOTrainAlgorithm(QgsProcessingAlgorithm):
    DATA_DIR = "DATA_DIR"
    EPOCHS = "EPOCHS"
    LR = "LR"
    K = "K"
    BATCH_SIZE = "BATCH_SIZE"
    VAL_SPLIT = "VAL_SPLIT"
    PATIENCE = "PATIENCE"
    SEED = "SEED"
    DEVICE = "DEVICE"
    OUTPUT_MODEL = "OUTPUT_MODEL"

    def tr(self, string):
        return QCoreApplication.translate("GNOTrainAlgorithm", string)

    def createInstance(self):
        return GNOTrainAlgorithm()

    def name(self):
        return "gno_train"

    def displayName(self):
        return self.tr("Entraînement GNO")

    def group(self):
        return self.tr("GNO Inference")

    def groupId(self):
        return "gno_inference"

    def shortHelpString(self):
        return self.tr(
            "Entraîne un modèle GINO sur des paires de rasters GeoTIFF "
            "alignés (xxx_input.tif / xxx_target.tif) placés dans un dossier.\n\n"
            "Sauvegarde le meilleur modèle (.pt) et un fichier .norm.json "
            "de statistiques de normalisation."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.DATA_DIR,
                self.tr("Dossier de données (paires input/target)"),
                behavior=QgsProcessingParameterFile.Folder,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.EPOCHS,
                self.tr("Nombre d'epochs"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.LR,
                self.tr("Taux d'apprentissage"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.001,
                minValue=1e-6,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.K,
                self.tr("k (voisins du graphe)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BATCH_SIZE,
                self.tr("Taille de lot (scènes)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.VAL_SPLIT,
                self.tr("Proportion de scènes en validation"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.15,
                minValue=0.0,
                maxValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PATIENCE,
                self.tr("Patience (arrêt anticipé)"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=15,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SEED,
                self.tr("Graine aléatoire"),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=42,
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
            QgsProcessingParameterFileDestination(
                self.OUTPUT_MODEL,
                self.tr("Modèle de sortie (.pt)"),
                fileFilter="PyTorch (*.pt *.pth)",
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        from ..core.gno_training import GNOTrainingEngine

        data_dir = self.parameterAsString(parameters, self.DATA_DIR, context)
        epochs = self.parameterAsInt(parameters, self.EPOCHS, context)
        lr = self.parameterAsDouble(parameters, self.LR, context)
        k = self.parameterAsInt(parameters, self.K, context)
        batch_size = self.parameterAsInt(parameters, self.BATCH_SIZE, context)
        val_split = self.parameterAsDouble(parameters, self.VAL_SPLIT, context)
        patience = self.parameterAsInt(parameters, self.PATIENCE, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        device_idx = self.parameterAsEnum(parameters, self.DEVICE, context)
        device_str = "gpu" if device_idx == 1 else "cpu"
        output_model = self.parameterAsString(parameters, self.OUTPUT_MODEL, context)

        if not data_dir:
            raise QgsProcessingException(self.tr("Dossier de données manquant."))
        if not output_model:
            raise QgsProcessingException(self.tr("Chemin du modèle de sortie manquant."))

        engine = GNOTrainingEngine(
            data_dir=data_dir,
            output_path=output_model,
            epochs=epochs,
            lr=lr,
            k=k,
            batch_size=batch_size,
            val_split=val_split,
            patience=patience,
            seed=seed,
            device_str=device_str,
            progress_cb=lambda p: feedback.setProgress(p),
            message_cb=lambda m: feedback.pushInfo(m),
            warning_cb=lambda w: feedback.pushWarning(w),
            is_canceled_cb=lambda: feedback.isCanceled(),
        )

        try:
            result = engine.run()
        except Exception as exc:
            raise QgsProcessingException(str(exc))

        return {self.OUTPUT_MODEL: result}
