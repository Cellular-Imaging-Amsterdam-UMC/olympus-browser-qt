from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import os
from typing import Any

import cv2
import numpy as np

from .vsi_attachments import extract_embedded_preview
from .vsi_native import create_vsi_preview


def cache_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "olympus_browser_qt_preview_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_digest(*parts: object) -> str:
    payload = json.dumps([str(part) for part in parts], sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _autocontrast_image(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image
    if image.ndim == 2:
        luminance = image.astype(np.float32, copy=False)
        alpha = None
        channels = image[..., None]
    else:
        alpha = image[..., 3:4].copy() if image.shape[2] == 4 else None
        color = image[..., :3]
        luminance = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32, copy=False)
        channels = color
    lo, hi = np.percentile(luminance, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return image
    scaled = np.clip((channels.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo)), 0.0, 255.0).astype(np.uint8)
    if image.ndim == 2:
        return scaled[..., 0]
    if alpha is not None:
        return np.dstack([scaled, alpha])
    return scaled


def _load_autocontrasted_preview(source: Path, *, cache_key: str) -> Path | None:
    if not source.exists():
        return None
    cached = cache_dir() / f"{cache_key}.png"
    if cached.exists():
        return cached
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    adjusted = _autocontrast_image(image)
    if not cv2.imwrite(str(cached), adjusted):
        return None
    return cached


def _placeholder_preview_path(metadata: dict[str, Any], preview_height: int) -> Path:
    digest = _cache_digest(metadata, preview_height, "placeholder")
    path = cache_dir() / f"{digest}.png"
    if path.exists():
        return path

    width = max(128, int(round(preview_height * 4 / 3)))
    height = max(96, int(preview_height))
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.broadcast_to((25 + 80 * x + 15 * y), (height, width)).astype(np.uint8)
    image[..., 1] = np.broadcast_to((40 + 120 * y), (height, width)).astype(np.uint8)
    image[..., 2] = np.broadcast_to((90 + 100 * (1 - x)), (height, width)).astype(np.uint8)

    name = str(metadata.get("save_child_name") or metadata.get("name") or metadata.get("source_file") or "VSI")
    status = str(metadata.get("backend_status") or "placeholder preview")
    scene_text = None
    selected_s = metadata.get("selected_s")
    scene_names = metadata.get("scene_names")
    if selected_s is not None:
        try:
            scene_index = int(selected_s)
            if isinstance(scene_names, list) and 0 <= scene_index < len(scene_names):
                scene_text = f"S={scene_index + 1} {scene_names[scene_index]}"
            else:
                scene_text = f"S={scene_index + 1}"
        except (TypeError, ValueError):
            scene_text = None
    cv2.putText(image, "VSI", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 245, 245), 2, cv2.LINE_AA)
    if scene_text:
        cv2.putText(image, scene_text[:32], (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (238, 244, 250), 1, cv2.LINE_AA)
    cv2.putText(image, name[:42], (24, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(image, status[:46], (24, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)
    return path


def preview_png_from_metadata(
    metadata: dict[str, Any],
    *,
    selected_s: int | None = None,
    preview_height: int = 512,
    use_memmap: bool = True,
    max_cache_size: int = 500,
) -> Path:
    metadata = dict(metadata)
    if selected_s is None:
        metadata.pop("selected_s", None)
    else:
        metadata["selected_s"] = int(selected_s)
    explicit_path = metadata.get("preview_png")
    if explicit_path:
        explicit = Path(str(explicit_path))
        explicit_stat = explicit.stat() if explicit.exists() else None
        if explicit_stat is not None:
            cached = _load_autocontrasted_preview(
                explicit,
                cache_key=_cache_digest(explicit.resolve(), explicit_stat.st_mtime_ns, preview_height, "explicit"),
            )
            if cached is not None:
                return cached
    source_file = metadata.get("source_file")
    if source_file:
        source = Path(str(source_file))
        native_preview = create_vsi_preview(metadata, selected_s=selected_s, preview_height=int(preview_height))
        if native_preview is not None and native_preview.exists():
            return native_preview
        if not _skip_embedded_preview(metadata, selected_s):
            preview = extract_embedded_preview(source, cache_dir())
            if preview is not None and preview.exists():
                preview_stat = preview.stat()
                cached = _load_autocontrasted_preview(
                    preview,
                    cache_key=_cache_digest(source.resolve(), preview_stat.st_mtime_ns, preview_height, "embedded"),
                )
                if cached is not None:
                    return cached
    return _placeholder_preview_path(metadata, int(preview_height))


def _skip_embedded_preview(metadata: dict[str, Any], selected_s: int | None) -> bool:
    try:
        size_s = int(metadata.get("size_s", metadata.get("tiles", 1)) or 1)
    except (TypeError, ValueError):
        size_s = 1
    return selected_s is not None and size_s > 1
