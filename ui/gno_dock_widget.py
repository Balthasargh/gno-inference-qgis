# -*- coding: utf-8 -*-
"""Panneau dockable interactif pour l'inférence GNO."""

from __future__ import annotations
import os
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QMessageBox, QGroupBox, QRadioButton, QButtonGroup,
)
from qgis.core import QgsProject, QgsRasterLayer


class GNODockWidget(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__("GNO Inference", parent)
        self.iface = iface
        self.plugin_dir = os.path.dirname(os.path.dirname(__file__))
        self._processor = None
        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)

        type_group = QGroupBox("Type d'entrée")
        type_layout = QHBoxLayout(type_group)
        self.radio_raster = QRadioButton("Raster")
        self.radio_pc = QRadioButton("Nuage de points (LAS/LAZ)")
        self.radio_raster.setChecked(True)
        self._type_group = QButtonGroup(self)
        self._type_group.addButton(self.radio_raster)
        self._type_group.addButton(self.radio_pc)
        type_layout.addWidget(self.radio_raster)
        type_layout.addWidget(self.radio_pc)
        layout.addWidget(type_group)
        self.radio_raster.toggled.connect(self._update_input_ui)

        form = QFormLayout()
        in_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.btn_input = QPushButton("\u2026")
        self.btn_input.setFixedWidth(30)
        self.btn_input.clicked.connect(self._browse_input)
        in_row.addWidget(self.input_edit)
        in_row.addWidget(self.btn_input)
        form.addRow("Entrée :", in_row)

        self.las_dim_edit = QLineEdit("z")
        form.addRow("Dimension LAS :", self.las_dim_edit)

        model_row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.btn_model = QPushButton("\u2026")
        self.btn_model.setFixedWidth(30)
        self.btn_model.clicked.connect(self._browse_model)
        self.btn_zoo = QPushButton("Model Zoo\u2026")
        self.btn_zoo.clicked.connect(self._open_zoo)
        model_row.addWidget(self.model_edit)
        model_row.addWidget(self.btn_model)
        model_row.addWidget(self.btn_zoo)
        form.addRow("Mod\u00e8le :", model_row)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.btn_output = QPushButton("\u2026")
        self.btn_output.setFixedWidth(30)
        self.btn_output.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(self.btn_output)
        form.addRow("Sortie :", out_row)

        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 64)
        self.k_spin.setValue(8)
        form.addRow("k (voisins) :", self.k_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(256, 100000)
        self.batch_spin.setSingleStep(512)
        self.batch_spin.setValue(4096)
        form.addRow("Taille de lot :", self.batch_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU"])
        form.addRow("Device :", self.device_combo)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Lancer l'inf\u00e9rence")
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        layout.addWidget(self.log)
        self._update_input_ui()

    def _update_input_ui(self):
        self.las_dim_edit.setEnabled(self.radio_pc.isChecked())

    def _browse_input(self):
        if self.radio_raster.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Raster d'entr\u00e9e", "", "Rasters (*.tif *.tiff);;Tous (*.*)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Nuage de points", "", "LAS/LAZ (*.las *.laz);;Tous (*.*)")
        if path:
            self.input_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mod\u00e8le GNO", "", "PyTorch (*.pt *.pth);;Tous (*.*)")
        if path:
            self.model_edit.setText(path)

    def _browse_output(self):
        if self.radio_raster.isChecked():
            path, _ = QFileDialog.getSaveFileName(self, "Raster de sortie", "", "GeoTIFF (*.tif)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Nuage de points de sortie", "", "LAS (*.las);;LAZ (*.laz)")
        if path:
            self.output_edit.setText(path)

    def _open_zoo(self):
        from .model_zoo_dialog import ModelZooDialog
        dlg = ModelZooDialog(self.plugin_dir, parent=self)
        if dlg.exec_() and dlg.selected_path():
            self.model_edit.setText(dlg.selected_path())

    def _log(self, msg):
        self.log.append(msg)

    def _run(self):
        input_path = self.input_edit.text().strip()
        model_path = self.model_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path or not os.path.isfile(input_path):
            QMessageBox.warning(self, "GNO Inference", "Fichier d'entr\u00e9e invalide.")
            return
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "GNO Inference", "Fichier mod\u00e8le invalide.")
            return
        if not output_path:
            QMessageBox.warning(self, "GNO Inference", "Chemin de sortie manquant.")
            return
        if self.radio_pc.isChecked():
            from ..utils.dependencies import missing_las_dependencies
            missing = missing_las_dependencies()
            if missing:
                from .dependency_dialog import DependencyInstallDialog
                dlg = DependencyInstallDialog(missing, parent=self)
                dlg.exec_()
                return
        from ..processing.gno_processor import GNOProcessor
        device_str = "gpu" if self.device_combo.currentIndex() == 1 else "cpu"
        self._processor = GNOProcessor(
            model_path=model_path, input_path=input_path, output_path=output_path,
            k=self.k_spin.value(), batch_size=self.batch_spin.value(),
            device_str=device_str, las_dimension=self.las_dim_edit.text().strip() or "z",
            parent=self,
        )
        self._processor.progress.connect(self.progress.setValue)
        self._processor.message.connect(self._log)
        self._processor.warning.connect(lambda w: self._log(f"\u26a0 {w}"))
        self._processor.finished_ok.connect(self._on_finished)
        self._processor.failed.connect(self._on_failed)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self._log("D\u00e9marrage de l'inf\u00e9rence\u2026")
        self._processor.start()

    def _cancel(self):
        if self._processor is not None:
            self._processor.cancel()
            self._log("Annulation demand\u00e9e\u2026")

    def _on_finished(self, path):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._log(f"\u2713 Termin\u00e9 : {path}")
        self._load_result(path)

    def _on_failed(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._log(f"\u2717 \u00c9chec : {msg}")
        QMessageBox.warning(self, "GNO Inference", msg)

    def _load_result(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tif", ".tiff"):
            layer = QgsRasterLayer(path, os.path.basename(path))
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self._log("Raster ajout\u00e9 au projet.")
        elif ext in (".las", ".laz"):
            self._log("Nuage de points \u00e9crit. Chargez-le manuellement.")
