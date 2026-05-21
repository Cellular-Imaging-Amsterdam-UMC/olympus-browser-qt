from __future__ import annotations

from pathlib import Path
from typing import Any

import tifffile

from .vsi_native import companion_ets_files, dataset_info


def extract_vsi_xml_metadata(path: Path, ets_path: str | Path | None = None) -> dict[str, Any]:
    """Return normalized Olympus VSI metadata.

    The function name is kept for API continuity with the scaffold, but Olympus
    VSI files commonly expose TIFF/SIS metadata plus external ETS pixel files
    rather than embedded XML.
    """

    path = Path(path)
    tiff_metadata = _read_tiff_metadata(path)
    ets_info = dataset_info(path, ets_path=ets_path)
    if ets_info is not None:
        size_x = ets_info.size_x
        size_y = ets_info.size_y
        size_z = ets_info.size_z
        size_c = ets_info.size_c
        pixel_type = _pixel_type_name(ets_info.pixel_type)
        channel_resolution = [_bit_depth(ets_info.dtype.itemsize)] * size_c
        backend_status = "native-ets-raw" if ets_info.is_raw else f"native-ets-compression-{ets_info.compression}"
    else:
        size_x = int(tiff_metadata.get("preview_size_x") or 512)
        size_y = int(tiff_metadata.get("preview_size_y") or 384)
        size_z = 1
        size_c = int(tiff_metadata.get("preview_size_c") or 1)
        pixel_type = str(tiff_metadata.get("preview_pixel_type") or "uint8")
        channel_resolution = [8] * max(size_c, 1)
        backend_status = "tiff-preview"

    channel_names = _channel_names(path, size_c)
    placeholder_size_x, placeholder_size_y = _placeholder_shape(size_x, size_y)
    ets_files = companion_ets_files(path)
    selected_ets = Path(ets_path) if ets_path is not None else (ets_files[0] if ets_files else None)
    metadata: dict[str, Any] = {
        "filetype": ".vsi",
        "source_file": str(path),
        "save_child_name": path.stem or path.name,
        "name": path.stem or path.name,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "size_c": size_c,
        "size_t": 1,
        "size_s": 1,
        "xs": size_x,
        "ys": size_y,
        "zs": size_z,
        "channels": size_c,
        "ts": 1,
        "tiles": 1,
        "dimensions": {"x": size_x, "y": size_y, "z": size_z, "c": size_c, "t": 1, "s": 1},
        "channel_names": channel_names,
        "pixel_type": pixel_type,
        "channelResolution": channel_resolution,
        "isrgb": False,
        "placeholder_size_x": placeholder_size_x,
        "placeholder_size_y": placeholder_size_y,
        "backend_status": backend_status,
        "pixel_backend": "native",
        "ets_files": [str(p) for p in ets_files],
        "ets_file_count": len(ets_files),
    }
    if selected_ets is not None:
        metadata["ets_file"] = str(selected_ets)
        metadata["ets_stack_name"] = selected_ets.parent.name
    metadata.update(tiff_metadata)
    if ets_info is not None:
        metadata.update(
            {
                "ets_tile_width": ets_info.tile_width,
                "ets_tile_height": ets_info.tile_height,
                "ets_chunk_count": len(ets_info.chunks),
                "ets_compression": ets_info.compression,
                "ets_pixel_type": ets_info.pixel_type,
                "ets_use_pyramid": ets_info.use_pyramid,
            }
        )
    return metadata


def _read_tiff_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                return metadata
            page = tif.pages[0]
            shape = tuple(int(v) for v in page.shape)
            metadata["preview_size_y"] = shape[0] if len(shape) >= 1 else None
            metadata["preview_size_x"] = shape[1] if len(shape) >= 2 else None
            metadata["preview_size_c"] = shape[2] if len(shape) >= 3 else 1
            metadata["preview_pixel_type"] = page.dtype.name
            metadata["preview_png"] = ""
            metadata["experiment_datetime"] = _tag_value(page, "DateTime")
            metadata["camera_make"] = _tag_value(page, "Make")
            metadata["camera_model"] = _tag_value(page, "Model")
            sis = page.tags.get(33560)
            if sis is not None and isinstance(sis.value, dict):
                metadata["olympus_sis"] = {str(k): str(v) for k, v in sis.value.items()}
                px = _as_float(sis.value.get("pixelsizex"))
                py = _as_float(sis.value.get("pixelsizey"))
                if px:
                    metadata["xres2"] = px
                if py:
                    metadata["yres2"] = py
                if px or py:
                    metadata["resunit2"] = "micrometer"
    except Exception as exc:
        metadata["warning"] = f"VSI TIFF preview metadata could not be parsed: {exc}"
    return metadata


def _tag_value(page: Any, name: str) -> str | None:
    tag = page.tags.get(name)
    if tag is None or tag.value is None:
        return None
    return str(tag.value)


def _channel_names(path: Path, size_c: int) -> list[str]:
    stem = path.stem
    prefix = stem.split("_", 1)[1] if "_" in stem else stem
    prefix = prefix.rsplit("_", 1)[0] if "," not in prefix and "_" in prefix else prefix
    names = [part.strip().split("_", 1)[0].strip() for part in prefix.split(",") if part.strip()]
    if len(names) >= size_c:
        return names[:size_c]
    names.extend(f"Channel {idx + 1}" for idx in range(len(names), size_c))
    return names


def _pixel_type_name(pixel_type: int) -> str:
    return {
        1: "int8",
        2: "uint8",
        3: "int16",
        4: "uint16",
        5: "int32",
        6: "uint32",
        9: "float32",
        10: "float64",
    }.get(pixel_type, f"olympus-{pixel_type}")


def _bit_depth(itemsize: int) -> int:
    return max(int(itemsize) * 8, 8)


def _placeholder_shape(size_x: int, size_y: int, max_long_edge: int = 1536) -> tuple[int, int]:
    if size_x <= 0 or size_y <= 0:
        return 512, 384
    long_edge = max(size_x, size_y)
    if long_edge <= max_long_edge:
        return size_x, size_y
    scale = max_long_edge / float(long_edge)
    return max(64, int(round(size_x * scale))), max(64, int(round(size_y * scale)))


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
