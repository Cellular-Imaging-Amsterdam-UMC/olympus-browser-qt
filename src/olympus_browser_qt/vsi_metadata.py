from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import tifffile

from .vsi_native import companion_ets_files, dataset_info

# ──────────────────────── VSI binary-container constants ──────────────────────
# Tag IDs (from Bio-Formats CellSensReader)
_RWC_FRAME_SCALE = 2019    # physical pixel size X/Y (pair of float64)
_Z_INCREMENT = 2013        # Z-step (inner VALUE tag holds float64)
_CHANNEL_NAME = 2419       # channel name (UTF-16 LE string)
_VALUE = 268435458         # 0x10000002 — generic value leaf tag

# Objective / optics tags
_OBJECTIVE_MAG = 120060    # nominal magnification (float64)
_NUMERICAL_APERTURE = 120061  # NA (float64)
_WORKING_DISTANCE = 120062  # working distance (float64 or inside sub-volume)
_OBJECTIVE_NAME = 120063   # objective model name (UTF-16 LE)
_REFRACTIVE_INDEX = 120079 # immersion refractive index (float64)

# realType codes
_FLOAT = 9
_DOUBLE = 10
_DOUBLE_2 = 260
_TCHAR = 13
_UNICODE_TCHAR = 8192

# Extended-field sub-types
_NEW_VOLUME_HEADER = 0
_PROPERTY_SET_VOLUME = 1
_NEW_MDIM_VOLUME_HEADER = 2

_SENTINEL_TAG = 0xE2A7A001  # -494804095 as uint32 — marks end-of-container



def _read_vsi_binary_metadata(path: Path) -> dict[str, Any]:
    """Parse the VSI binary container starting at file offset 8.

    Extracts physical pixel sizes (XY), Z-step, and channel names from the
    CellSens hierarchical tag container.  The TIFF/SIS tag (33560) inside the
    VSI preview page returns dummy values (1.0, 1.0) for the pixel size; the
    correct calibration is in this binary container.
    """
    result: dict[str, Any] = {}
    try:
        raw = path.read_bytes()
        _parse_vsi_volume(raw, 8, result, "")
        sizes = result.pop("_phys_sizes", [])
        if sizes:
            result["xres2"] = sizes[0][0]
            result["yres2"] = sizes[0][1]
            result["resunit2"] = "micrometer"
        zincs = result.pop("_z_increments", [])
        if zincs:
            result["zres2"] = zincs[0]
        wds = result.pop("_working_distances", [])
        if wds:
            result["working_distance_um"] = wds[0]
        names = result.pop("_channel_names_vsi", [])
        if names:
            # De-duplicate while preserving order (the container may repeat names
            # across multiple pyramid metadata blocks).
            seen: set[str] = set()
            unique: list[str] = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    unique.append(n)
            result["channel_names_vsi"] = unique
    except Exception as exc:
        result["vsi_binary_warning"] = str(exc)
    return result


def _parse_vsi_volume(raw: bytes, fp: int, result: dict, tag_prefix: str) -> int:
    """Recursively parse one VSI volume header + tag list.

    Parameters
    ----------
    raw        : complete file bytes
    fp         : start of this volume's 24-byte header
    result     : accumulator dict
    tag_prefix : inherited context string (e.g. ``"Z increment"``)

    Returns the byte offset after the last processed tag (best-effort),
    used by the NEW_VOLUME_HEADER sibling-loop to advance.
    """
    n = len(raw)
    if fp + 24 > n:
        return fp

    # Volume header (24 bytes, little-endian):
    #   headerSize(2)  version(2)  volumeVersion(4)
    #   dataFieldOffset(8, unsigned)  flags(4)  padding(4)
    data_field_offset = struct.unpack_from("<Q", raw, fp + 8)[0]
    flags = struct.unpack_from("<I", raw, fp + 16)[0]
    tag_count = flags & 0x0FFFFFFF

    if tag_count > 200_000:
        return fp + 24

    pos = fp + int(data_field_offset)
    if pos >= n:
        return fp + 24

    last_data_size = 0

    for _ in range(tag_count):
        if pos + 16 > n:
            break

        field_type = struct.unpack_from("<I", raw, pos)[0]
        tag        = struct.unpack_from("<I", raw, pos + 4)[0]
        next_field = struct.unpack_from("<I", raw, pos + 8)[0]   # relative to fp
        data_size  = struct.unpack_from("<I", raw, pos + 12)[0]
        last_data_size = data_size

        extra_tag_flag = (field_type >> 27) & 1
        extended_field = (field_type >> 28) & 1
        inline_data    = (field_type >> 30) & 1
        real_type      = field_type & 0x00FFFFFF

        cursor = pos + 16
        if extra_tag_flag:
            cursor += 4   # skip secondTag int

        if tag == _SENTINEL_TAG:
            return fp + last_data_size + 32

        if extended_field:
            if real_type == _NEW_VOLUME_HEADER:
                end_ptr = cursor + data_size
                sub_pos = cursor
                while sub_pos < end_ptr and sub_pos < n:
                    prev = sub_pos
                    sub_pos = _parse_vsi_volume(raw, sub_pos, result, _vsi_volume_prefix(tag))
                    if sub_pos <= prev:
                        break
            elif real_type in (_PROPERTY_SET_VOLUME, _NEW_MDIM_VOLUME_HEADER):
                if real_type == _NEW_MDIM_VOLUME_HEADER:
                    sub_prefix = _vsi_volume_prefix(tag)
                    if not sub_prefix and tag == _Z_INCREMENT:
                        sub_prefix = "Z increment"
                else:
                    sub_prefix = tag_prefix
                _parse_vsi_volume(raw, cursor, result, sub_prefix)
        else:
            if not inline_data and data_size > 0:
                _extract_vsi_leaf(raw, cursor, tag, real_type, data_size, tag_prefix, result)

        if next_field == 0:
            return fp + last_data_size + 32

        next_pos = fp + next_field
        if next_pos <= pos or next_pos >= n:
            break
        pos = next_pos

    return pos


def _vsi_volume_prefix(tag: int) -> str:
    """Return the tag-prefix string used when entering a nested VSI volume."""
    if tag == 2417:
        return "Channel Wavelength "
    if tag == _WORKING_DISTANCE:   # 120062
        return "Objective Working Distance "
    if tag == 2100:          # TIME_VALUE
        return "Timestamp "
    return ""


def _extract_vsi_leaf(
    raw: bytes,
    data_start: int,
    tag: int,
    real_type: int,
    data_size: int,
    tag_prefix: str,
    result: dict,
) -> None:
    n = len(raw)
    if data_start + data_size > n:
        return

    # Physical pixel size XY
    if tag == _RWC_FRAME_SCALE and data_size >= 16:
        x, y = struct.unpack_from("<dd", raw, data_start)
        if x > 0 and y > 0:
            result.setdefault("_phys_sizes", []).append((x, y))
        return

    # Z-step (VALUE tag inside a "Z increment" sub-volume)
    if tag == _VALUE and real_type == _DOUBLE and data_size == 8 and tag_prefix == "Z increment":
        z = struct.unpack_from("<d", raw, data_start)[0]
        if z > 0:
            result.setdefault("_z_increments", []).append(z)
        return

    # Working distance — VALUE inside "Objective Working Distance" sub-volume
    if tag == _VALUE and real_type == _DOUBLE and data_size == 8 and tag_prefix.startswith("Objective Working Distance"):
        wd = struct.unpack_from("<d", raw, data_start)[0]
        if wd >= 0:
            result.setdefault("_working_distances", []).append(wd)
        return

    # Working distance — direct leaf variant (float or double)
    if tag == _WORKING_DISTANCE and real_type in (_FLOAT, _DOUBLE) and data_size in (4, 8):
        fmt = "<f" if data_size == 4 else "<d"
        wd = struct.unpack_from(fmt, raw, data_start)[0]
        if wd >= 0:
            result.setdefault("_working_distances", []).append(wd)
        return

    # Objective magnification
    if tag == _OBJECTIVE_MAG and real_type in (_FLOAT, _DOUBLE) and data_size in (4, 8):
        fmt = "<f" if data_size == 4 else "<d"
        mag = struct.unpack_from(fmt, raw, data_start)[0]
        if mag > 0 and "objective_mag" not in result:
            result["objective_mag"] = mag
        return

    # Numerical aperture
    if tag == _NUMERICAL_APERTURE and real_type in (_FLOAT, _DOUBLE) and data_size in (4, 8):
        fmt = "<f" if data_size == 4 else "<d"
        na = struct.unpack_from(fmt, raw, data_start)[0]
        if na > 0 and "numerical_aperture" not in result:
            result["numerical_aperture"] = na
        return

    # Refractive index
    if tag == _REFRACTIVE_INDEX and real_type in (_FLOAT, _DOUBLE) and data_size in (4, 8):
        fmt = "<f" if data_size == 4 else "<d"
        ri = struct.unpack_from(fmt, raw, data_start)[0]
        if ri > 0 and "refractive_index" not in result:
            result["refractive_index"] = ri
        return

    # Objective name (first occurrence = objective, subsequent = filter cubes)
    if tag == _OBJECTIVE_NAME and real_type in (_UNICODE_TCHAR, _TCHAR) and data_size > 0:
        raw_bytes = raw[data_start : data_start + data_size]
        if real_type == _UNICODE_TCHAR:
            name = raw_bytes.decode("utf-16-le", errors="replace").rstrip("\x00").strip()
        else:
            name = raw_bytes.decode("latin-1", errors="replace").rstrip("\x00").strip()
        if name and "objective_name" not in result:
            result["objective_name"] = name
        return

    # Channel name
    if tag == _CHANNEL_NAME and real_type == _UNICODE_TCHAR:
        raw_bytes = raw[data_start : data_start + data_size]
        name = raw_bytes.decode("utf-16-le", errors="replace").rstrip("\x00").strip()
        if name:
            result.setdefault("_channel_names_vsi", []).append(name)


def extract_vsi_xml_metadata(path: Path, ets_path: str | Path | None = None) -> dict[str, Any]:
    """Return normalized Olympus VSI metadata.

    The function name is kept for API continuity with the scaffold, but Olympus
    VSI files commonly expose TIFF/SIS metadata plus external ETS pixel files
    rather than embedded XML.
    """

    path = Path(path)
    tiff_metadata = _read_tiff_metadata(path)

    # Override dummy pixel-size values (the SIS TIFF tag on the preview page
    # returns 1.0 for both X and Y) with calibration from the binary container.
    binary_meta = _read_vsi_binary_metadata(path)
    if "xres2" in binary_meta:
        tiff_metadata["xres2"] = binary_meta["xres2"]
        tiff_metadata["yres2"] = binary_meta.get("yres2", binary_meta["xres2"])
        tiff_metadata["resunit2"] = "micrometer"
    if "zres2" in binary_meta:
        tiff_metadata["zres2"] = binary_meta["zres2"]
    for _field in ("objective_name", "objective_mag", "numerical_aperture",
                   "refractive_index", "working_distance_um"):
        if _field in binary_meta:
            tiff_metadata[_field] = binary_meta[_field]

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

    # Use channel names from the binary container when available and the count
    # matches the number of channels.  Fall back to the filename-based heuristic.
    vsi_names = binary_meta.get("channel_names_vsi", [])
    if len(vsi_names) >= size_c:
        channel_names = vsi_names[:size_c]
    else:
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
