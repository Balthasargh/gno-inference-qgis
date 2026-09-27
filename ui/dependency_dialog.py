# -*- coding: utf-8 -*-
"""Dialogue d'installation assistée des dépendances Python."""

from __future__ import annotations

from typing import List

from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout, QProgressBar,
)


class _InstallThread(QThread):
    line = pyqtSignal(str)
    finished_code = pyqtSignal(int)

    def __init__(self, packages: List[str], parent=None):
        super().__init__(parent)
        self.packages = packages

    def run(self):
        from ..utils.dependencies import install_packages
        code = install_packages(
            self.packages, user=True, log_cb=lambda msg: self.line.emit(msg),
        )
        self.finished_code.emit(code)


class DependencyInstallDialog(QDialog):
    def __init__(self, missing: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("GNO Inference — Dépendances manquantes")
        self.resize(520, 360)
        self._missing = missing
        self._thread = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Les paquets Python suivants sont requis mais absents de "
            "l'environnement QGIS :"
        ))
        layout.addWidget(QLabel("<b>" + ", ".join(missing) + "</b>"))
        layout.addWidget(QLabel(
            "Vous pouvez les installer maintenant (pip install --user). "
            "Redémarrez QGIS après l'installation."
        ))

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.btn_install = QPushButton("Installer")
        self.btn_install.clicked.connect(self._start_install)
        self.btn_close = QPushButton("Fermer")
        self.btn_close.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_install)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def _start_install(self):
        self.btn_install.setEnabled(False)
        self.progress.setVisible(True)
        self.log.append("Installation en cours…\n")
        self._thread = _InstallThread(self._missing, parent=self)
        self._thread.line.connect(self.log.append)
        self._thread.finished_code.connect(self._on_finished)
        self._thread.start()

    def _on_finished(self, code: int):
        self.progress.setVisible(False)
        if code == 0:
            self.log.append(
                "\n\u2713 Installation terminée. Redémarrez QGIS pour prendre "
                "en compte les nouveaux paquets."
            )
        else:
            self.log.append(
                f"\n\u2717 Installation terminée avec le code {code}. "
                "Vérifiez les messages ci-dessus."
            )
            self.btn_install.setEnabled(True)
