# -*- coding: utf-8 -*-
"""
QThread d'inférence pour le panneau dockable (pattern Deepness / non-bloquant).
"""

from __future__ import annotations

from qgis.PyQt.QtCore import QThread, pyqtSignal


class GNOProcessor(QThread):
    progress = pyqtSignal(float)
    message = pyqtSignal(str)
    warning = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        model_path: str,
        input_path: str,
        output_path: str,
        k: int = 8,
        batch_size: int = 4096,
        device_str: str = "cpu",
        las_dimension: str = "z",
        parent=None,
    ):
        super().__init__(parent)
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = output_path
        self.k = k
        self.batch_size = batch_size
        self.device_str = device_str
        self.las_dimension = las_dimension
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def run(self):
        from ..core.gno_inference import GNOInferenceEngine

        engine = GNOInferenceEngine(
            model_path=self.model_path,
            k=self.k,
            batch_size=self.batch_size,
            device_str=self.device_str,
            progress_cb=lambda p: self.progress.emit(p),
            message_cb=lambda m: self.message.emit(m),
            warning_cb=lambda w: self.warning.emit(w),
            is_canceled_cb=lambda: self._canceled,
        )
        try:
            result = engine.run(
                self.input_path,
                self.output_path,
                las_dimension=self.las_dimension,
            )
            if self._canceled:
                self.failed.emit("Inférence annulée.")
            else:
                self.finished_ok.emit(result)
        except InterruptedError:
            self.failed.emit("Inférence annulée.")
        except Exception as e:
            self.failed.emit(str(e))
