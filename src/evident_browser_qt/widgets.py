"""Small Qt widgets adapted from omero-browser-qt.

Portions adapted from omero-browser-qt 0.2.5, Copyright (c) 2026 Ron Hoebe,
MIT License.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPolygonF
from PyQt6.QtWidgets import QComboBox


class ArrowComboBox(QComboBox):
    """Combo box that paints a simple down-arrow reliably across styles."""

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#d5d9dd") if self.isEnabled() else QColor("#727980"))

        center_x = self.width() - 14
        center_y = self.height() / 2.0 + 0.5
        arrow = QPolygonF(
            [
                QPointF(center_x - 4.5, center_y - 2.5),
                QPointF(center_x + 4.5, center_y - 2.5),
                QPointF(center_x, center_y + 3.0),
            ]
        )
        painter.drawPolygon(arrow)

