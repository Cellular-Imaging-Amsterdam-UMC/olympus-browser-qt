"""Olympus image-provider helpers for the Qt viewer.

The provider mirrors the lightweight plane-provider interface used by
omero-browser-qt's viewer, but reads from OlympusImageContext objects. It prefers
real native VSI planes when available, then native scene previews, and only
falls back to a generated placeholder as a last resort.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from PyQt6.QtGui import QImage

from .models import OlympusImageContext
from .preview import preview_png_from_metadata

_DEFAULT_COLORS = [
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]
_COLOR_NAMES = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
    "gray": (180, 180, 180),
    "grey": (180, 180, 180),
}


def get_context_metadata(context: OlympusImageContext) -> dict[str, Any]:
    size_c = context.size_c or len(context.channel_names) or 1
    lut_names = context.metadata.get("lutname")
    channels = []
    for idx in range(max(size_c, 1)):
        name = context.channel_names[idx] if idx < len(context.channel_names) else f"Ch{idx + 1}"
        color = None
        if isinstance(lut_names, list) and idx < len(lut_names):
            color = _COLOR_NAMES.get(str(lut_names[idx]).strip().lower())
        if color is None:
            color = _color_from_channel_name(name)
        channels.append(
            {
                "name": name,
                "index": idx,
                "color": color or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)],
                "active": True,
                "window_start": None,
                "window_end": None,
            }
        )
    return {
        "name": context.name,
        "id": context.image_id,
        "container_path": str(context.container_path),
        "internal_path": context.internal_path,
        "size_x": context.size_x,
        "size_y": context.size_y,
        "size_z": context.size_z or 1,
        "size_c": max(size_c, 1),
        "size_t": context.size_t or 1,
        "size_s": context.size_s or 1,
        "selected_s": context.selected_s,
        "pixel_type": context.metadata.get("pixel_type", "u1"),
        "pixel_size_x": context.pixel_size_x_um,
        "pixel_size_y": context.pixel_size_y_um,
        "pixel_size_z": context.pixel_size_z_um,
        "channels": channels,
        "source_metadata": context.metadata,
    }


def _color_from_channel_name(name: str) -> tuple[int, int, int] | None:
    low = str(name).strip().lower()
    if any(token in low for token in ("dapi", "hoechst", "405")):
        return 0, 128, 255
    if any(token in low for token in ("488", "fitc", "gfp", "alexa fluor 488")):
        return 0, 255, 0
    if any(token in low for token in ("568", "594", "tritc", "cy3")):
        return 255, 128, 0
    if any(token in low for token in ("647", "660", "cy5", "alexa fluor 647")):
        return 255, 0, 128
    return None


class OlympusPreviewPlaneProvider:
    """Plane provider for the Olympus viewer.

    It first asks the context handle for real planes. If that backend is not
    implemented, it loads the existing preview PNG and exposes its
    RGB channels as display planes.
    """

    def __init__(self, context: OlympusImageContext, max_cache_items: int = 64) -> None:
        self.context = context
        self._meta = get_context_metadata(context)
        self._cache: OrderedDict[tuple[int, int, int, int], np.ndarray] = OrderedDict()
        self._max_cache_items = max_cache_items
        self._preview_rgb: np.ndarray | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_preview_loaded()
        if self._preview_rgb is not None:
            height, width = self._preview_rgb.shape[:2]
            self._meta["size_x"] = self._meta.get("size_x") or width
            self._meta["size_y"] = self._meta.get("size_y") or height
        return self._meta

    def get_plane(self, c: int, z: int, t: int = 0, s: int | None = None) -> np.ndarray:
        resolved_s = self.context.selected_s if s is None else s
        key = (c, z, t, int(resolved_s or 0))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if not self._has_real_pixels():
            return self._remember(key, self._preview_channel(c))

        try:
            arr = self.context.open().read_plane(z=z, c=c, t=t, s=resolved_s)
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[..., min(c, arr.shape[-1] - 1)]
        except Exception:
            arr = self._preview_channel(c)
        return self._remember(key, arr)

    def get_stack(self, c: int, t: int = 0, s: int | None = None, progress=None) -> np.ndarray:
        z_count = max(int(self.metadata.get("size_z") or 1), 1)
        planes = []
        for z in range(z_count):
            planes.append(self.get_plane(c, z, t, s=s))
            if progress is not None:
                progress(z + 1, z_count)
        return np.stack(planes, axis=0)

    def _preview_channel(self, c: int) -> np.ndarray:
        self._ensure_preview_loaded()
        assert self._preview_rgb is not None
        channel = c % self._preview_rgb.shape[2]
        return self._preview_rgb[..., channel]

    def _ensure_preview_loaded(self) -> None:
        if self._preview_rgb is not None:
            return
        try:
            png = preview_png_from_metadata(
                self.context.metadata,
                selected_s=self.context.selected_s,
                preview_height=768,
            )
            image = QImage(str(png)).convertToFormat(QImage.Format.Format_RGB888)
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * image.bytesPerLine())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, image.bytesPerLine()))
            arr = arr[:, : width * 3].reshape((height, width, 3)).copy()
            self._preview_rgb = arr
        except Exception:
            self._preview_rgb = self._placeholder_preview()

    def _has_real_pixels(self) -> bool:
        pixel_backend = self.context.metadata.get("pixel_backend")
        if isinstance(pixel_backend, str):
            return pixel_backend.strip().lower() in {"native", "real"}
        return False

    def _placeholder_preview(self) -> np.ndarray:
        width = int(self._meta.get("size_x") or 512)
        height = int(self._meta.get("size_y") or 384)
        width = max(64, min(width, 1024))
        height = max(64, min(height, 768))
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        red = np.broadcast_to(x * 120 + y * 60, (height, width)).astype(np.uint8)
        green = np.broadcast_to(y * 160 + 40, (height, width)).astype(np.uint8)
        blue = np.broadcast_to((1 - x) * 120 + 40, (height, width)).astype(np.uint8)
        return np.dstack([red, green, blue])

    def _remember(self, key: tuple[int, int, int, int], arr: np.ndarray) -> np.ndarray:
        self._cache[key] = arr
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache_items:
            self._cache.popitem(last=False)
        return arr
