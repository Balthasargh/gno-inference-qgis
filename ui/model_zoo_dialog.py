# -*- coding: utf-8 -*-
"""Bo\u00eete de dialogue Model Zoo."""

from __future__ import annotations
import os
from typing import Optional
from qgis.PyQt.QtCore import QThread, pyqtSignal, Qt
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QLineEdit, QProgressBar, QMessageBox, QAbstractItemView,
)


class _DownloadThread(QThread):
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def run(self):
        from ..utils.model_zoo import download_model
        try:
            path = download_model(
                self.model,
                progress_cb=lambda cur, tot: self.progress.emit(cur, tot),
                is_canceled_cb=lambda: self._canceled,
            )
            self.finished_ok.emit(path)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelZooDialog(QDialog):
    def __init__(self, plugin_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GNO Inference \u2014 Model Zoo")
        self.resize(560, 420)
        self._plugin_dir = plugin_dir
        self._selected_path = None
        self._thread = None
        self._models = []

        layout = QVBoxLayout(self)
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Registre :"))
        self.registry_edit = QLineEdit()
        from ..utils.model_zoo import default_registry_path
        self.registry_edit.setText(default_registry_path(plugin_dir))
        reg_row.addWidget(self.registry_edit)
        self.btn_refresh = QPushButton("Rafra\u00eechir")
        self.btn_refresh.clicked.connect(self._load_registry)
        reg_row.addWidget(self.btn_refresh)
        layout.addLayout(reg_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.list_widget)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.btn_download = QPushButton("T\u00e9l\u00e9charger / Utiliser")
        self.btn_download.clicked.connect(self._on_download)
        self.btn_cancel = QPushButton("Annuler le t\u00e9l\u00e9chargement")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_download)
        self.btn_close = QPushButton("Fermer")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_download)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.list_widget.currentItemChanged.connect(self._on_selection)
        self._load_registry()

    def selected_path(self):
        return self._selected_path

    def _load_registry(self):
        from ..utils.model_zoo import load_registry, is_cached, cached_path
        path = self.registry_edit.text().strip()
        self._models = load_registry(path)
        self.list_widget.clear()
        if not self._models:
            self.list_widget.addItem(QListWidgetItem("(Aucun mod\u00e8le dans ce registre)"))
            return
        for m in self._models:
            status = "\u2713 en cache" if is_cached(m) else "\u2193 \u00e0 t\u00e9l\u00e9charger"
            item = QListWidgetItem(f"{m.name}  [{m.task}]  \u2014 {status}")
            item.setData(Qt.UserRole, m.id)
            self.list_widget.addItem(item)

    def _current_model(self):
        item = self.list_widget.currentItem()
        if item is None:
            return None
        mid = item.data(Qt.UserRole)
        for m in self._models:
            if m.id == mid:
                return m
        return None

    def _on_selection(self, current, previous):
        m = self._current_model()
        if m is None:
            self.detail_label.setText("")
            return
        from ..utils.model_zoo import is_cached, cached_path
        lines = [
            f"<b>{m.name}</b> (id={m.id})",
            m.description or "",
            f"T\u00e2che : {m.task}  |  k : {m.recommended_k}  |  batch : {m.recommended_batch_size}",
        ]
        if m.license:
            lines.append(f"Licence : {m.license}")
        if is_cached(m):
            lines.append(f"Cache : {cached_path(m)}")
        self.detail_label.setText("<br>".join(lines))

    def _on_download(self):
        from ..utils.model_zoo import is_cached, cached_path
        m = self._current_model()
        if m is None:
            return
        if is_cached(m):
            self._selected_path = cached_path(m)
            self.accept()
            return
        self.btn_download.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._thread = _DownloadThread(m, parent=self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(self._on_downloaded)
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _on_progress(self, cur, tot):
        if tot > 0:
            self.progress.setMaximum(tot)
            self.progress.setValue(cur)

    def _on_downloaded(self, path):
        self.progress.setVisible(False)
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._selected_path = path
        self._load_registry()
        self.accept()

    def _on_failed(self, msg):
        self.progress.setVisible(False)
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        QMessageBox.warning(self, "T\u00e9l\u00e9chargement \u00e9chou\u00e9", msg)

    def _cancel_download(self):
        if self._thread is not None:
            self._thread.cancel()
