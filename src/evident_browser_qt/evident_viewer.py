#!/usr/bin/env python
"""Olympus VSI viewer.

This module is adapted from the public omero-browser-qt viewer design:
zoomable microscopy canvas, channel toggles, contrast controls, Z/T controls,
scale bar, status readout, and an embeddable QMainWindow. OMERO-specific
login/ICE/pyramid code has been replaced by OlympusImageContext loading through
OlympusBrowserDialog and OlympusPreviewPlaneProvider.

Portions adapted from omero-browser-qt 0.2.5, Copyright (c) 2026 Ron Hoebe,
MIT License.
"""

from __future__ import annotations

import sys
import argparse
from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QSlider,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from .olympus_browser_dialog import OlympusBrowserDialog
from .icons import make_app_icon
from .olympus_image_loader import OlympusPreviewPlaneProvider
from .metadata import metadata_rows
from .models import OlympusImageContext
from .scale_bar import compute_scale_bar

_FALLBACK_PALETTE = [
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]
_PROJECTION_MODES = ["Slice", "MIP", "SUM", "Mean", "Median"]


def _make_app_icon() -> QIcon:
    """Create a microscopy-inspired app icon programmatically."""

    size = 128
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    grad = QRadialGradient(size / 2, size / 2, size / 2)
    grad.setColorAt(0.0, QColor(50, 60, 80))
    grad.setColorAt(1.0, QColor(20, 25, 35))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, size - 4, size - 4)

    cx, cy = size / 2, size / 2
    r = size * 0.22
    off = size * 0.12
    channels = [
        (cx - off, cy + off * 0.6, QColor(0, 200, 80, 140)),
        (cx + off, cy + off * 0.6, QColor(220, 0, 80, 140)),
        (cx, cy - off * 0.8, QColor(0, 120, 255, 140)),
    ]
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    for x, y, col in channels:
        channel_grad = QRadialGradient(x, y, r)
        channel_grad.setColorAt(0.0, col)
        channel_grad.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
        p.setBrush(QBrush(channel_grad))
        p.drawEllipse(int(x - r), int(y - r), int(2 * r), int(2 * r))

    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    p.setPen(QPen(QColor(180, 200, 220, 160), 2.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    ring = size * 0.32
    p.drawEllipse(int(cx - ring), int(cy - ring), int(2 * ring), int(2 * ring))
    p.end()
    return QIcon(pix)


def _resolve_channel_colors(channels: list[dict]) -> list[tuple[int, int, int]]:
    colors = []
    for idx, channel in enumerate(channels):
        color = channel.get("color")
        if isinstance(color, tuple) and len(color) == 3:
            colors.append(tuple(int(v) for v in color))
        else:
            colors.append(_FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)])
    return colors


def _composite_to_pixmap(
    slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]],
) -> QPixmap:
    """Composite multiple grayscale channel planes into an RGB pixmap."""

    if not slices:
        return QPixmap()

    h, w = slices[0][0].shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    for arr, (cr, cg, cb), (lo, hi) in slices:
        if hi <= lo:
            hi = lo + 1.0
        norm = (arr.astype(np.float64) - lo) / (hi - lo)
        np.clip(norm, 0.0, 1.0, out=norm)
        canvas[..., 0] += norm * (cr / 255.0)
        canvas[..., 1] += norm * (cg / 255.0)
        canvas[..., 2] += norm * (cb / 255.0)

    np.clip(canvas, 0.0, 1.0, out=canvas)
    rgb = np.ascontiguousarray((canvas * 255).astype(np.uint8))
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _project_stack(stack: np.ndarray, mode: str, z_index: int) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3-D stack, got shape {stack.shape}")
    z = max(0, min(z_index, stack.shape[0] - 1))
    if mode == "Slice":
        return stack[z]
    if mode == "MIP":
        return stack.max(axis=0)
    if mode == "SUM":
        return stack.sum(axis=0).astype(np.float64)
    if mode == "Mean":
        return stack.mean(axis=0)
    if mode == "Median":
        return np.median(stack.astype(np.float64, copy=False), axis=0)
    raise ValueError(f"Unknown projection mode: {mode}")


class ZoomableImageView(QGraphicsView):
    """Pannable, zoomable image viewer widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._scale_bar_um_per_pixel: float | None = None
        self.setStyleSheet(
            "QGraphicsView { background: #111315; border: 1px solid #43484d; border-radius: 8px; }"
        )

    def set_pixmap(self, pix: QPixmap) -> None:
        self._pix_item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.viewport().update()

    def set_scale_bar_um_per_pixel(self, value: float | None) -> None:
        self._scale_bar_um_per_pixel = value
        self.viewport().update()

    def fit_in_view(self) -> None:
        self.resetTransform()
        target = self._pix_item.sceneBoundingRect()
        if target.width() > 0 and target.height() > 0:
            self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(target.center())
        self.viewport().update()

    def actual_size(self) -> None:
        self.resetTransform()
        if not self._pix_item.pixmap().isNull():
            self.centerOn(self._pix_item.sceneBoundingRect().center())
        self.viewport().update()

    def wheelEvent(self, event):  # noqa: N802
        factor = 1.15
        self.scale(factor, factor) if event.angleDelta().y() > 0 else self.scale(1 / factor, 1 / factor)
        self.viewport().update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        self._draw_scale_bar()

    def _draw_scale_bar(self) -> None:
        spec = compute_scale_bar(self._scale_bar_um_per_pixel, self.transform().m11())
        if spec is None:
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin = 18
        bar_y = self.viewport().height() - margin
        bar_x = margin
        label = spec.label
        text_rect = painter.fontMetrics().boundingRect(label)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(2, 6, 23, 190))
        painter.drawRoundedRect(
            bar_x - 8,
            bar_y - 34,
            int(max(spec.screen_pixels + 16, text_rect.width() + 20)),
            38,
            8,
            8,
        )
        painter.setPen(QPen(QColor("#f8fafc"), 2))
        painter.drawLine(bar_x, bar_y, int(bar_x + spec.screen_pixels), bar_y)
        painter.drawText(bar_x, bar_y - 9, label)
        painter.end()


class OlympusViewerWindow(QMainWindow):
    """OMERO-viewer-style window adapted for Olympus image contexts."""

    def __init__(
        self,
        context: OlympusImageContext | None = None,
        roots: list[str] | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Olympus Viewer")
        self.setWindowIcon(make_app_icon())
        self.resize(1120, 760)

        self._provider: OlympusPreviewPlaneProvider | None = None
        self._context: OlympusImageContext | None = None
        self._roots = roots
        self._metadata: dict[str, Any] = {}
        self._channel_colors: list[tuple[int, int, int]] = []
        self._channel_buttons: list[QPushButton] = []

        self._build_ui()
        if context is not None:
            self.open_context(context)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(18, 18, 18, 12)
        main_lay.setSpacing(14)
        self.setStyleSheet(
            "QMainWindow { background: #111315; color: #eceff1; }"
            "QLabel { color: #d5d9dd; }"
            "QLabel#title { color: #f3f4f6; font-size: 22px; font-weight: 700; }"
            "QLabel#hint { color: #8f969d; font-size: 12px; }"
            "QPushButton { background: #1e293b; color: #e2e8f0; border: 1px solid #334155;"
            "border-radius: 6px; padding: 6px 10px; font-weight: 600; }"
            "QPushButton:hover { background: #273449; border-color: #475569; }"
            "QPushButton:checked { background: #0f766e; color: #ecfeff; }"
            "QPushButton#primary { background: #0ea5e9; color: #082f49; border-color: #38bdf8; }"
            "QComboBox, QDoubleSpinBox { background: #1b1e21; color: #eceff1; border: 1px solid #43484d;"
            "border-radius: 6px; padding: 4px 8px; min-height: 18px; }"
            "QSlider::groove:horizontal { background: #262a2e; height: 6px; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #f3f4f6; width: 16px; margin: -6px 0; border-radius: 8px; }"
            "QSlider::groove:vertical { background: #262a2e; width: 6px; border-radius: 3px; }"
            "QSlider::handle:vertical { background: #f3f4f6; height: 16px; margin: 0 -6px; border-radius: 8px; }"
            "QTableWidget { background: #15181b; color: #d5d9dd; border: 1px solid #2d3338; }"
            "QHeaderView::section { background: #1e293b; color: #e2e8f0; border: 0; padding: 4px; }"
            "QStatusBar { background: #0d0f11; color: #8f969d; }"
        )

        top_row = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Olympus Viewer")
        title.setObjectName("title")
        self._path_label = QLabel("Open a Olympus .vsi image")
        self._path_label.setObjectName("hint")
        self._path_label.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(self._path_label)
        top_row.addLayout(title_box, 1)

        top_row.addWidget(QLabel("Lo"))
        self._lo_spin = QDoubleSpinBox()
        self._lo_spin.setRange(0.0, 50.0)
        self._lo_spin.setValue(0.1)
        self._lo_spin.setSingleStep(0.1)
        self._lo_spin.setDecimals(2)
        self._lo_spin.setFixedWidth(72)
        self._lo_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._lo_spin.valueChanged.connect(self._update_viewer)
        top_row.addWidget(self._lo_spin)

        top_row.addWidget(QLabel("Hi"))
        self._hi_spin = QDoubleSpinBox()
        self._hi_spin.setRange(50.0, 100.0)
        self._hi_spin.setValue(99.9)
        self._hi_spin.setSingleStep(0.1)
        self._hi_spin.setDecimals(2)
        self._hi_spin.setFixedWidth(72)
        self._hi_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._hi_spin.valueChanged.connect(self._update_viewer)
        top_row.addWidget(self._hi_spin)

        actual_btn = QPushButton("100%")
        actual_btn.clicked.connect(self._viewer_actual_size_later)
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._viewer_fit_later)
        open_btn = QPushButton("Open VSI")
        open_btn.setObjectName("primary")
        open_btn.clicked.connect(self.open_from_browser)
        top_row.addWidget(actual_btn)
        top_row.addWidget(fit_btn)
        top_row.addWidget(open_btn)
        main_lay.addLayout(top_row)

        self._ch_row = QHBoxLayout()
        self._ch_row.addStretch()
        main_lay.addLayout(self._ch_row)

        body = QHBoxLayout()
        self._viewer = ZoomableImageView()
        body.addWidget(self._viewer, 1)

        z_col = QVBoxLayout()
        z_lbl = QLabel("Z")
        z_lbl.setObjectName("hint")
        z_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_col.addWidget(z_lbl)
        self._z_slider = QSlider(Qt.Orientation.Vertical)
        self._z_slider.setRange(0, 0)
        self._z_slider.valueChanged.connect(self._update_viewer)
        z_col.addWidget(self._z_slider, 1, Qt.AlignmentFlag.AlignHCenter)
        self._z_label = QLabel("0/0")
        self._z_label.setObjectName("hint")
        self._z_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_col.addWidget(self._z_label)
        body.addLayout(z_col)

        side = QVBoxLayout()
        self._metadata_table = QTableWidget(0, 2)
        self._metadata_table.setHorizontalHeaderLabels(["Key", "Value"])
        self._metadata_table.horizontalHeader().setStretchLastSection(True)
        self._metadata_table.verticalHeader().setVisible(False)
        side.addWidget(self._metadata_table)
        body.addLayout(side, 0)
        main_lay.addLayout(body, 1)

        self._s_controls = QWidget()
        s_row = QHBoxLayout(self._s_controls)
        s_row.setContentsMargins(0, 0, 0, 0)
        s_row.addWidget(QLabel("S"))
        self._s_slider = QSlider(Qt.Orientation.Horizontal)
        self._s_slider.setRange(0, 0)
        self._s_slider.valueChanged.connect(self._update_viewer)
        s_row.addWidget(self._s_slider, 1)
        self._s_label = QLabel("0/0")
        self._s_label.setObjectName("hint")
        s_row.addWidget(self._s_label)
        self._s_controls.setVisible(False)
        main_lay.addWidget(self._s_controls)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("T"))
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setRange(0, 0)
        self._t_slider.valueChanged.connect(self._update_viewer)
        bottom.addWidget(self._t_slider, 1)
        self._t_label = QLabel("0/0")
        self._t_label.setObjectName("hint")
        bottom.addWidget(self._t_label)
        self._proj_combo = QComboBox()
        self._proj_combo.addItems(_PROJECTION_MODES)
        self._proj_combo.currentIndexChanged.connect(self._update_viewer)
        bottom.addWidget(self._proj_combo)
        main_lay.addLayout(bottom)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

    def open_from_browser(self) -> None:
        context = OlympusBrowserDialog.select_image_context(roots=self._roots, parent=self)
        if context is not None:
            self.open_context(context)

    def open_context(self, context: OlympusImageContext) -> None:
        self._context = context
        self._provider = OlympusPreviewPlaneProvider(context)
        self._metadata = self._provider.metadata
        self._channel_colors = _resolve_channel_colors(self._metadata.get("channels", []))
        self._path_label.setText(f"{context.container_path} / {context.internal_path}")
        self._viewer.set_scale_bar_um_per_pixel(self._metadata.get("pixel_size_x"))
        self._rebuild_channel_buttons()
        self._configure_dimension_controls()
        self._populate_metadata_table()
        self._update_viewer()
        self._viewer_fit_later()
        self._status.showMessage(f"Loaded {context.name}")

    def _rebuild_channel_buttons(self) -> None:
        while self._ch_row.count() > 0:
            item = self._ch_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._channel_buttons.clear()

        channels = self._metadata.get("channels", []) or []
        for idx, channel in enumerate(channels):
            color = self._channel_colors[idx] if idx < len(self._channel_colors) else (255, 255, 255)
            button = QPushButton(channel.get("name") or f"Ch{idx + 1}")
            button.setCheckable(True)
            button.setChecked(bool(channel.get("active", True)))
            button.setStyleSheet(
                f"QPushButton:checked {{ background: rgb({color[0]}, {color[1]}, {color[2]}); color: #020617; }}"
            )
            button.toggled.connect(self._update_viewer)
            self._channel_buttons.append(button)
            self._ch_row.addWidget(button)
        self._ch_row.addStretch()

    def _configure_dimension_controls(self) -> None:
        z_count = max(int(self._metadata.get("size_z") or 1), 1)
        t_count = max(int(self._metadata.get("size_t") or 1), 1)
        s_count = max(int(self._metadata.get("size_s") or 1), 1)
        selected_s = self._context.selected_s if self._context is not None else None
        self._z_slider.blockSignals(True)
        self._z_slider.setRange(0, z_count - 1)
        self._z_slider.setValue(min(z_count // 2, z_count - 1))
        self._z_slider.blockSignals(False)
        self._t_slider.blockSignals(True)
        self._t_slider.setRange(0, t_count - 1)
        self._t_slider.setValue(0)
        self._t_slider.blockSignals(False)
        self._s_slider.blockSignals(True)
        self._s_slider.setRange(0, s_count - 1)
        self._s_slider.setValue(selected_s if selected_s is not None else min(s_count // 2, s_count - 1))
        self._s_slider.blockSignals(False)
        self._s_controls.setVisible(s_count > 1 and selected_s is None)
        self._refresh_dimension_labels()

    def _populate_metadata_table(self) -> None:
        rows = metadata_rows((self._context.metadata if self._context else {}) or {})
        self._metadata_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self._metadata_table.setItem(row, 0, QTableWidgetItem(key))
            self._metadata_table.setItem(row, 1, QTableWidgetItem(value))
        self._metadata_table.resizeColumnsToContents()

    def _update_viewer(self, *_args) -> None:
        if self._provider is None:
            return
        self._refresh_dimension_labels()
        active = [idx for idx, button in enumerate(self._channel_buttons) if button.isChecked()]
        if not active:
            self._viewer.set_pixmap(QPixmap())
            self._status.showMessage("No active channels")
            return

        z = self._z_slider.value()
        t = self._t_slider.value()
        s = self._current_s_value()
        mode = self._proj_combo.currentText()
        slices = []
        for channel_index in active:
            stack = self._provider.get_stack(channel_index, t, s=s)
            plane = _project_stack(stack, mode, z)
            lo, hi = self._contrast_limits(plane)
            color = self._channel_colors[channel_index % len(self._channel_colors)]
            slices.append((plane, color, (lo, hi)))

        pixmap = _composite_to_pixmap(slices)
        self._viewer.set_pixmap(pixmap)
        self._status.showMessage(
            f"{self._metadata.get('name', '')}  X={self._metadata.get('size_x')} "
            f"Y={self._metadata.get('size_y')} Z={z + 1} T={t + 1}{self._format_s_status(s)}"
        )

    def _contrast_limits(self, arr: np.ndarray) -> tuple[float, float]:
        lo_pct = min(self._lo_spin.value(), self._hi_spin.value() - 0.001)
        hi_pct = max(self._hi_spin.value(), lo_pct + 0.001)
        lo, hi = np.percentile(arr.astype(np.float64, copy=False), [lo_pct, hi_pct])
        if hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

    def _refresh_dimension_labels(self) -> None:
        self._z_label.setText(f"{self._z_slider.value() + 1}/{self._z_slider.maximum() + 1}")
        self._t_label.setText(f"{self._t_slider.value() + 1}/{self._t_slider.maximum() + 1}")
        self._s_label.setText(f"{self._current_s_value() + 1}/{self._s_slider.maximum() + 1}")

    def _current_s_value(self) -> int:
        if self._context is not None and self._context.selected_s is not None:
            return int(self._context.selected_s)
        return self._s_slider.value()

    def _format_s_status(self, s: int) -> str:
        size_s = int(self._metadata.get("size_s") or 1)
        if size_s <= 1 and (self._context is None or self._context.selected_s is None):
            return ""
        suffix = f" S={s + 1}"
        tile_positions = self._metadata.get("source_metadata", {}).get("tile_positions")
        if isinstance(tile_positions, list) and 0 <= s < len(tile_positions):
            tile = tile_positions[s]
            pos_x = tile.get("PosX")
            pos_y = tile.get("PosY")
            if pos_x is not None and pos_y is not None:
                suffix += f" ({pos_x:.3f}, {pos_y:.3f})"
        return suffix

    def _viewer_fit_later(self) -> None:
        QTimer.singleShot(0, self._viewer.fit_in_view)

    def _viewer_actual_size_later(self) -> None:
        QTimer.singleShot(0, self._viewer.actual_size)


_APP_REF: QApplication | None = None


def ensure_app() -> QApplication:
    global _APP_REF
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    _APP_REF = app
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the Olympus VSI viewer.")
    parser.add_argument("paths", nargs="*", help="Optional files or folders to browse first.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = ensure_app()
    win = OlympusViewerWindow(roots=args.paths or None)
    win.show()
    return app.exec()



if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
