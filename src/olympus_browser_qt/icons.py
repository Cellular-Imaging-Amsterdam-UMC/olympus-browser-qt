from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QRadialGradient


def make_app_icon() -> QIcon:
    """Create a microscopy-inspired Olympus application icon."""

    size = 128
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    grad = QRadialGradient(size / 2, size / 2, size / 2)
    grad.setColorAt(0.0, QColor(27, 64, 96))
    grad.setColorAt(1.0, QColor(6, 18, 34))
    painter.setBrush(QBrush(grad))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, size - 4, size - 4)

    cx, cy = size / 2, size / 2
    r = size * 0.22
    off = size * 0.12
    channels = [
        (cx - off, cy + off * 0.6, QColor(26, 196, 255, 150)),
        (cx + off, cy + off * 0.6, QColor(0, 214, 143, 145)),
        (cx, cy - off * 0.9, QColor(255, 193, 59, 145)),
    ]
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
    for x, y, color in channels:
        halo = QRadialGradient(x, y, r)
        halo.setColorAt(0.0, color)
        halo.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        painter.setBrush(QBrush(halo))
        painter.drawEllipse(int(x - r), int(y - r), int(2 * r), int(2 * r))

    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setPen(QPen(QColor(220, 240, 255, 185), 2.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    ring = size * 0.32
    painter.drawEllipse(int(cx - ring), int(cy - ring), int(2 * ring), int(2 * ring))
    painter.setPen(QPen(QColor(255, 255, 255, 230), 3))
    painter.drawLine(int(cx - 14), int(cy + 28), int(cx + 14), int(cy - 28))
    painter.drawLine(int(cx + 6), int(cy - 10), int(cx + 20), int(cy - 34))
    painter.end()
    return QIcon(pix)