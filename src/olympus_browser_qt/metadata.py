from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .models import OlympusImageContext

SPATIAL_RESOLUTION_AXES = ("x", "y", "z")


def pick(metadata: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return default


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def unit_to_micrometer_factor(unit: Any) -> float:
    text = str(unit or "").strip().lower()
    if text in {"meter", "metre", "meters", "metres", "m"}:
        return 1_000_000.0
    if text in {"centimeter", "centimetre", "centimeters", "centimetres", "cm"}:
        return 10_000.0
    if text in {"millimeter", "millimetre", "millimeters", "millimetres", "mm"}:
        return 1_000.0
    if text in {"micrometer", "micrometre", "micrometers", "micrometres", "um", "µm"}:
        return 1.0
    if text in {"nanometer", "nanometre", "nanometers", "nanometres", "nm"}:
        return 0.001
    if text in {"inch", "in"}:
        return 25_400.0
    return 1.0


def normalize_resolution_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Repair micrometer-resolution fields when microscopy metadata is incomplete."""

    factor = unit_to_micrometer_factor(pick(metadata, "resunit", "xresunit", "unit"))
    for axis in SPATIAL_RESOLUTION_AXES:
        native = as_float(metadata.get(f"{axis}res"))
        converted = as_float(metadata.get(f"{axis}res2"))
        if native is None:
            continue
        expected = native * factor
        if _needs_resolution_fallback(native, converted, factor):
            metadata[f"{axis}res2"] = expected
    if any(f"{axis}res2" in metadata for axis in SPATIAL_RESOLUTION_AXES):
        metadata["resunit2"] = "micrometer"
    return metadata


def _needs_resolution_fallback(native: float, converted: float | None, factor: float) -> bool:
    if converted is None or converted == 0:
        return native != 0
    if factor == 1.0:
        return False
    return abs(converted - native) <= max(abs(native), 1.0) * 1e-12


def channel_names_from_metadata(metadata: dict[str, Any]) -> list[str]:
    for key in ("channel_names", "channelNames", "channels_names"):
        value = metadata.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]

    lut_names = metadata.get("lutname")
    if isinstance(lut_names, list):
        return [str(v) for v in lut_names]

    count = as_int(metadata.get("channels") or (metadata.get("dimensions") or {}).get("c"))
    if count:
        return [f"Channel {idx + 1}" for idx in range(count)]
    return []


def context_from_metadata(
    *,
    name: str,
    container_path: Path,
    internal_path: str,
    image_id: str | None,
    kind: str,
    metadata: dict[str, Any],
) -> OlympusImageContext:
    metadata = normalize_resolution_metadata(metadata)
    dims = metadata.get("dimensions") if isinstance(metadata.get("dimensions"), dict) else {}
    return OlympusImageContext(
        name=name,
        container_path=container_path,
        internal_path=internal_path,
        image_id=image_id,
        kind=kind,
        size_x=as_int(pick(metadata, "xs", "size_x", default=dims.get("x"))),
        size_y=as_int(pick(metadata, "ys", "size_y", default=dims.get("y"))),
        size_z=as_int(pick(metadata, "zs", "size_z", default=dims.get("z"))),
        size_c=as_int(pick(metadata, "channels", "size_c", default=dims.get("c"))),
        size_t=as_int(pick(metadata, "ts", "size_t", default=dims.get("t"))),
        size_s=as_int(pick(metadata, "tiles", "size_s", default=dims.get("s"))),
        pixel_size_x_um=as_float(pick(metadata, "xres2", "pixel_size_x_um", "PhysicalSizeX")),
        pixel_size_y_um=as_float(pick(metadata, "yres2", "pixel_size_y_um", "PhysicalSizeY")),
        pixel_size_z_um=as_float(pick(metadata, "zres2", "pixel_size_z_um", "PhysicalSizeZ")),
        channel_names=channel_names_from_metadata(metadata),
        metadata=metadata,
    )


def metadata_rows(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = normalize_resolution_metadata(metadata)
    rows: list[tuple[str, str]] = []
    for key in sorted(metadata):
        value = metadata[key]
        if isinstance(value, (dict, list)):
            text = repr(value)
        else:
            text = "" if value is None else str(value)
        rows.append((str(key), text))
    return rows


def format_metadata_summary(metadata: dict[str, Any]) -> str:
    """Return the compact metadata summary used by the Olympus browser."""

    metadata = normalize_resolution_metadata(metadata)
    name = pick(metadata, "save_child_name", "name", "ElementName", default="(unnamed)")

    dims = metadata.get("dimensions") if isinstance(metadata.get("dimensions"), dict) else {}
    xs = pick(metadata, "xs", "size_x", default=dims.get("x"))
    ys = pick(metadata, "ys", "size_y", default=dims.get("y"))
    zs = pick(metadata, "zs", "size_z", default=dims.get("z"))
    ts = pick(metadata, "ts", "size_t", default=dims.get("t"))
    cs = pick(metadata, "channels", "size_c", default=dims.get("c"))
    ss = pick(metadata, "tiles", "size_s", default=dims.get("s"))

    dims_parts = []
    if xs and ys:
        dims_parts.append(f"{xs} x {ys}")
    if zs:
        dims_parts.append(f"Z={zs}")
    if ts:
        dims_parts.append(f"T={ts}")
    if cs:
        dims_parts.append(f"C={cs}")
    if as_int(ss) and as_int(ss) > 1:
        dims_parts.append(f"S={ss}")

    vx = pick(metadata, "xres2", "pixel_size_x_um", "PhysicalSizeX")
    vy = pick(metadata, "yres2", "pixel_size_y_um", "PhysicalSizeY")
    vz = pick(metadata, "zres2", "pixel_size_z_um", "PhysicalSizeZ")
    vunit = pick(metadata, "resunit2", default="um")

    def fmt2(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    scale_parts = []
    if vx:
        scale_parts.append(f"X={fmt2(vx)} {vunit}")
    if vy:
        scale_parts.append(f"Y={fmt2(vy)} {vunit}")
    if vz:
        scale_parts.append(f"Z={fmt2(vz)} {vunit}")

    is_rgb = bool(pick(metadata, "isrgb", default=False))
    channel_resolution = metadata.get("channelResolution") or []
    pixel_type = None
    if isinstance(channel_resolution, list) and channel_resolution:
        try:
            first = channel_resolution[0]
            pixel_type = (
                f"{first}-bit"
                if all(value == first for value in channel_resolution if value is not None)
                else "mixed-bit"
            )
        except Exception:
            pixel_type = None
    if pixel_type and is_rgb:
        pixel_type = f"{pixel_type} RGB"
    elif is_rgb:
        pixel_type = "RGB"

    lines = [
        f"Name: {name}",
    ]

    date_text = _format_datetime(pick(metadata, "experiment_datetime", "experiment_datetime_str"))
    if date_text:
        lines.append(f"Date: {date_text}")

    fov_size = _format_fov_size_um2(xs, ys, vx, vy)
    image_size = _format_image_size(xs, ys, zs, ts, cs, channel_resolution)

    lines.extend(
        [
            f"Dimensions: {'  '.join(dims_parts)}" if dims_parts else "Dimensions: (n/a)",
            f"Pixel size: {', '.join(scale_parts)}" if scale_parts else "Pixel size: (n/a)",
            f"FOV size: {fov_size}" if fov_size else "FOV size: (n/a)",
            f"Image size: {image_size}" if image_size else "Image size: (n/a)",
            f"Pixel type: {pixel_type}" if pixel_type else "Pixel type: (n/a)",
        ]
    )

    obj_parts = []
    obj_name = pick(metadata, "objective_name")
    obj_mag  = as_float(pick(metadata, "objective_mag"))
    na       = as_float(pick(metadata, "numerical_aperture"))
    ri       = as_float(pick(metadata, "refractive_index"))
    wd       = as_float(pick(metadata, "working_distance_um"))
    if obj_name:
        obj_parts.append(str(obj_name))
    if obj_mag:
        mag_str = f"{int(obj_mag)}x" if obj_mag == int(obj_mag) else f"{obj_mag}x"
        obj_parts.append(mag_str)
    if na:
        obj_parts.append(f"NA {na:.2f}")
    if ri:
        obj_parts.append(f"RI {ri:.3f}")
    if wd is not None:
        obj_parts.append(f"WD {wd:.0f} um")
    if obj_parts:
        lines.append(f"Objective: {'  '.join(obj_parts)}")

    experiment = pick(metadata, "experiment_name")
    if experiment:
        lines.append(f"Experiment: {experiment}")
    return "\n".join(lines)


def _format_fov_size_um2(xs: Any, ys: Any, vx: Any, vy: Any) -> str | None:
    size_x = as_int(xs)
    size_y = as_int(ys)
    pixel_x = as_float(vx)
    pixel_y = as_float(vy)
    if not size_x or not size_y or not pixel_x or not pixel_y:
        return None
    return f"{_format_decimal(size_x * pixel_x * size_y * pixel_y)} um^2"


def _format_image_size(
    xs: Any,
    ys: Any,
    zs: Any,
    ts: Any,
    cs: Any,
    channel_resolution: Any,
) -> str | None:
    size_x = as_int(xs)
    size_y = as_int(ys)
    size_z = as_int(zs) or 1
    size_t = as_int(ts) or 1
    channels = as_int(cs)
    if not size_x or not size_y:
        return None

    bit_depths = _channel_bit_depths(channel_resolution, channels)
    if not bit_depths:
        return None

    bytes_per_xyzt = sum((bits + 7) // 8 for bits in bit_depths)
    byte_count = size_x * size_y * size_z * size_t * bytes_per_xyzt
    return _format_byte_size(byte_count)


def _channel_bit_depths(channel_resolution: Any, channels: int | None) -> list[int]:
    if isinstance(channel_resolution, list):
        bit_depths = [as_int(value) for value in channel_resolution]
        bit_depths = [value for value in bit_depths if value]
        if not bit_depths:
            return []
        if channels and len(bit_depths) == 1:
            return bit_depths * channels
        if channels and len(bit_depths) < channels:
            return bit_depths + [bit_depths[-1]] * (channels - len(bit_depths))
        return bit_depths[:channels] if channels else bit_depths

    bit_depth = as_int(channel_resolution)
    if bit_depth:
        return [bit_depth] * (channels or 1)
    return []


def _format_byte_size(byte_count: int) -> str:
    if byte_count >= 1_000_000_000:
        return f"{byte_count / 1_000_000_000:.2f} GB"
    return f"{byte_count / 1_000_000:.1f} MB"


def _format_decimal(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_datetime(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pretty = raw.replace("T", " ")
        if "." in pretty:
            pretty = pretty.split(".", 1)[0]
        for sep in ("+", "-"):
            pos = pretty.find(sep, 10)
            if pos != -1:
                pretty = pretty[:pos]
                break
        return pretty
