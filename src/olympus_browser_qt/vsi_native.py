from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
import tifffile

try:
    import imagecodecs
except ImportError:  # pragma: no cover - dependency is declared by the environment/package
    imagecodecs = None


RAW = 0
JPEG = 2
JPEG_2000 = 3
JPEG_LOSSLESS = 5
PNG = 8
BMP = 9
PIXEL_DTYPES = {
    1: np.dtype("int8"),
    2: np.dtype("uint8"),
    3: np.dtype("<i2"),
    4: np.dtype("<u2"),
    5: np.dtype("<i4"),
    6: np.dtype("<u4"),
    9: np.dtype("<f4"),
    10: np.dtype("<f8"),
}


@dataclass(frozen=True)
class EtsChunk:
    coordinates: tuple[int, ...]
    offset: int
    byte_count: int

    @property
    def x_index(self) -> int:
        return self.coordinates[0] if len(self.coordinates) > 0 else 0

    @property
    def y_index(self) -> int:
        return self.coordinates[1] if len(self.coordinates) > 1 else 0

    @property
    def z_index(self) -> int:
        return self.coordinates[3] if len(self.coordinates) > 3 else 0

    @property
    def c_index(self) -> int:
        return self.coordinates[4] if len(self.coordinates) > 4 else 0

    @property
    def level(self) -> int:
        return self.coordinates[-1] if self.coordinates else 0


@dataclass(frozen=True)
class EtsInfo:
    path: Path
    n_dimensions: int
    pixel_type: int
    size_c_hint: int
    colorspace: int
    compression: int
    compression_quality: int
    tile_width: int
    tile_height: int
    tile_depth: int
    use_pyramid: bool
    chunks: tuple[EtsChunk, ...]

    @property
    def dtype(self) -> np.dtype:
        return PIXEL_DTYPES.get(self.pixel_type, np.dtype("<u2"))

    @property
    def size_c(self) -> int:
        return max(max((self.chunk_c_index(chunk) for chunk in self.chunks), default=0) + 1, self.size_c_hint, 1)

    @property
    def size_z(self) -> int:
        return max((self.chunk_z_index(chunk) for chunk in self.chunks), default=0) + 1

    @property
    def size_t(self) -> int:
        return 1

    @property
    def size_x(self) -> int:
        max_x = max((chunk.x_index for chunk in self.level0_chunks), default=0) + 1
        return max(max_x * self.tile_width, 1)

    @property
    def size_y(self) -> int:
        max_y = max((chunk.y_index for chunk in self.level0_chunks), default=0) + 1
        return max(max_y * self.tile_height, 1)

    @property
    def level0_chunks(self) -> tuple[EtsChunk, ...]:
        return tuple(chunk for chunk in self.chunks if self.chunk_level(chunk) == 0)

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(sorted({self.chunk_level(chunk) for chunk in self.chunks}))

    @property
    def is_raw(self) -> bool:
        return self.compression == RAW

    @property
    def is_rgb_encoded(self) -> bool:
        return self.size_c_hint > 1 and self.colorspace != 1

    @property
    def is_supported_compression(self) -> bool:
        return self.compression in {RAW, JPEG, JPEG_2000, JPEG_LOSSLESS, PNG, BMP}

    def chunk_level(self, chunk: EtsChunk) -> int:
        if not self.use_pyramid:
            return 0
        return chunk.coordinates[-1] if chunk.coordinates else 0

    def chunk_z_index(self, chunk: EtsChunk) -> int:
        if self.use_pyramid:
            return chunk.coordinates[2] if len(chunk.coordinates) > 3 and not self.is_rgb_encoded else 0
        return chunk.coordinates[3] if len(chunk.coordinates) > 3 else 0

    def chunk_c_index(self, chunk: EtsChunk) -> int:
        if self.is_rgb_encoded:
            return 0
        if self.use_pyramid:
            return chunk.coordinates[3] if len(chunk.coordinates) > 4 else 0
        return chunk.coordinates[4] if len(chunk.coordinates) > 4 else 0

    def level_size(self, level: int) -> tuple[int, int]:
        chunks = [chunk for chunk in self.chunks if self.chunk_level(chunk) == level]
        if not chunks:
            return 0, 0
        width = (max(chunk.x_index for chunk in chunks) + 1) * self.tile_width
        height = (max(chunk.y_index for chunk in chunks) + 1) * self.tile_height
        return max(width, 1), max(height, 1)

    def best_level_for_max_size(self, max_size: int) -> int:
        best = self.levels[0] if self.levels else 0
        for level in self.levels:
            width, height = self.level_size(level)
            if max(width, height) >= max_size:
                best = level
            else:
                break
        return best

    def chunk_for(self, *, x_tile: int, y_tile: int, z: int, c: int, level: int = 0) -> EtsChunk | None:
        for chunk in self.chunks:
            if (
                chunk.x_index == x_tile
                and chunk.y_index == y_tile
                and self.chunk_z_index(chunk) == z
                and (self.is_rgb_encoded or self.chunk_c_index(chunk) == c)
                and self.chunk_level(chunk) == level
            ):
                return chunk
        return None


def native_vsi_available() -> bool:
    """Return True because this package includes a pure Python VSI/ETS reader."""

    return True


def companion_ets_files(vsi_path: Path) -> list[Path]:
    pixels_dir = vsi_path.with_name(f"_{vsi_path.stem}_")
    if not pixels_dir.is_dir():
        return []
    files: list[Path] = []
    for stack_dir in sorted((p for p in pixels_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        files.extend(sorted(stack_dir.glob("frame_*.ets"), key=lambda p: p.name.lower()))
    return files


@lru_cache(maxsize=32)
def parse_ets(path_text: str) -> EtsInfo:
    path = Path(path_text)
    with path.open("rb") as fh:
        if fh.read(4).rstrip(b"\x00") != b"SIS":
            raise ValueError(f"Not an Olympus ETS/SIS file: {path}")
        header_size = _read_u32(fh)
        _version = _read_u32(fh)
        n_dimensions = _read_u32(fh)
        additional_header_offset = _read_u64(fh)
        _additional_header_size = _read_u32(fh)
        fh.seek(4, 1)
        used_chunk_offset = _read_u64(fh)
        n_used_chunks = _read_u32(fh)

        fh.seek(additional_header_offset)
        if fh.read(4).rstrip(b"\x00") != b"ETS":
            raise ValueError(f"ETS additional header not found in {path}")
        fh.seek(4, 1)
        pixel_type = _read_u32(fh)
        size_c_hint = _read_u32(fh)
        colorspace = _read_u32(fh)
        compression = _read_u32(fh)
        compression_quality = _read_u32(fh)
        tile_width = _read_u32(fh)
        tile_height = _read_u32(fh)
        tile_depth = _read_u32(fh)
        fh.seek(4 * 17, 1)
        dtype = PIXEL_DTYPES.get(pixel_type, np.dtype("<u2"))
        background_bytes = max(size_c_hint, 1) * dtype.itemsize
        fh.seek(background_bytes, 1)
        fh.seek(max(4 * 10 - background_bytes, 0), 1)
        _component_order = _read_u32(fh)
        use_pyramid = _read_u32(fh) != 0

        fh.seek(used_chunk_offset)
        chunks: list[EtsChunk] = []
        for _ in range(n_used_chunks):
            fh.seek(4, 1)
            coords = tuple(_read_i32(fh) for _ in range(n_dimensions))
            offset = _read_u64(fh)
            byte_count = _read_u32(fh)
            fh.seek(4, 1)
            chunks.append(EtsChunk(coords, offset, byte_count))

    if header_size <= 0 or not chunks:
        raise ValueError(f"No readable ETS chunk table found in {path}")
    return EtsInfo(
        path=path,
        n_dimensions=n_dimensions,
        pixel_type=pixel_type,
        size_c_hint=size_c_hint,
        colorspace=colorspace,
        compression=compression,
        compression_quality=compression_quality,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_depth=tile_depth,
        use_pyramid=use_pyramid,
        chunks=tuple(chunks),
    )


def dataset_info(vsi_path: Path, ets_path: str | Path | None = None) -> EtsInfo | None:
    if ets_path is not None:
        return parse_ets(str(Path(ets_path)))
    files = companion_ets_files(vsi_path)
    if not files:
        return None
    return parse_ets(str(files[0]))


def is_vsi_mosaic(path: Path, ets_path: str | Path | None = None) -> bool:
    info = dataset_info(path, ets_path=ets_path)
    return bool(info and (info.size_x > info.tile_width or info.size_y > info.tile_height))


def read_vsi_plane(
    path: Path,
    *,
    z: int = 0,
    c: int = 0,
    t: int = 0,
    s: int | None = None,
    ets_path: str | Path | None = None,
) -> np.ndarray:
    del t, s
    info = dataset_info(path, ets_path=ets_path)
    if info is None:
        return _read_tiff_preview_plane(path, c=c)
    return read_ets_region(info, x=0, y=0, width=info.size_x, height=info.size_y, z=z, c=c)


def read_vsi_mosaic_plane(
    path: Path,
    *,
    z: int = 0,
    c: int = 0,
    max_size: int = 2048,
    ets_path: str | Path | None = None,
) -> np.ndarray:
    info = dataset_info(path, ets_path=ets_path)
    if info is None:
        plane = read_vsi_plane(path, z=z, c=c)
    else:
        level = info.best_level_for_max_size(max_size)
        width, height = info.level_size(level)
        plane = read_ets_region(info, x=0, y=0, width=width, height=height, z=z, c=c, level=level)
    long_edge = max(plane.shape[:2])
    if long_edge <= max_size:
        return plane
    scale = float(max_size) / float(long_edge)
    width = max(1, int(round(plane.shape[1] * scale)))
    height = max(1, int(round(plane.shape[0] * scale)))
    return cv2.resize(plane, (width, height), interpolation=cv2.INTER_AREA)


def read_vsi_image(path: Path, *, z: int = 0, c: int = 0, t: int = 0, s: int | None = None) -> np.ndarray:
    return read_vsi_plane(path, z=z, c=c, t=t, s=s)


def read_ets_region(
    info: EtsInfo,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    z: int = 0,
    c: int = 0,
    level: int = 0,
) -> np.ndarray:
    if not info.is_supported_compression:
        raise NotImplementedError(f"ETS compression code {info.compression} is not implemented yet")
    if width <= 0 or height <= 0:
        return np.zeros((0, 0), dtype=info.dtype)

    z = _bounded("z", z, info.size_z)
    c = _bounded("channel", c, info.size_c)
    first_col = max(x // info.tile_width, 0)
    level_chunks = [chunk for chunk in info.chunks if info.chunk_level(chunk) == level]
    if not level_chunks:
        return np.zeros((height, width), dtype=info.dtype)
    last_col = min((x + width - 1) // info.tile_width, max(chunk.x_index for chunk in level_chunks))
    first_row = max(y // info.tile_height, 0)
    last_row = min((y + height - 1) // info.tile_height, max(chunk.y_index for chunk in level_chunks))
    out: np.ndarray | None = None

    for tile_y in range(first_row, last_row + 1):
        for tile_x in range(first_col, last_col + 1):
            chunk = info.chunk_for(x_tile=tile_x, y_tile=tile_y, z=z, c=c, level=level)
            if chunk is None:
                continue
            tile = _read_tile(info, chunk, c=c)
            if out is None:
                out_shape = (height, width) if tile.ndim == 2 else (height, width, tile.shape[2])
                out = np.zeros(out_shape, dtype=tile.dtype)
            src_x0 = max(x - tile_x * info.tile_width, 0)
            src_y0 = max(y - tile_y * info.tile_height, 0)
            src_x1 = min(x + width - tile_x * info.tile_width, tile.shape[1])
            src_y1 = min(y + height - tile_y * info.tile_height, tile.shape[0])
            dst_x0 = tile_x * info.tile_width + src_x0 - x
            dst_y0 = tile_y * info.tile_height + src_y0 - y
            out[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = tile[
                src_y0:src_y1, src_x0:src_x1
            ]
    if out is None:
        return np.zeros((height, width), dtype=info.dtype)
    return out


def create_vsi_preview(
    metadata: dict[str, Any],
    *,
    selected_s: int | None = None,
    preview_height: int = 512,
) -> Path | None:
    del selected_s
    source_file = metadata.get("source_file")
    if not source_file:
        return None
    source_path = Path(str(source_file))
    if not source_path.exists():
        return None

    try:
        stat = source_path.stat()
        digest = hashlib.sha1(
            json.dumps(
                [
                    "vsi-native-preview-v3-stack-aware",
                    str(source_path.resolve()),
                    stat.st_mtime_ns,
                    preview_height,
                    metadata.get("ets_file"),
                    metadata.get("ets_stack_name"),
                ],
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        out_path = _native_preview_cache_dir() / f"{digest}.png"
        if out_path.exists():
            return out_path

        info = dataset_info(source_path, ets_path=metadata.get("ets_file"))
        if info is None:
            preview = tifffile.imread(source_path)
        else:
            z = max(info.size_z // 2, 0)
            channels = []
            for channel in range(min(info.size_c, 3)):
                plane = read_vsi_mosaic_plane(
                    source_path,
                    z=z,
                    c=channel,
                    max_size=preview_height * 2,
                    ets_path=metadata.get("ets_file"),
                )
                channels.append(_normalize_plane(plane))
            preview = _compose_channels(channels, metadata)
        preview = _resize_preview(_ensure_rgb(preview), preview_height)
        if not cv2.imwrite(str(out_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
            return None
        return out_path
    except Exception:
        return None


def _read_tile(info: EtsInfo, chunk: EtsChunk, *, c: int = 0) -> np.ndarray:
    if info.compression == RAW:
        return _read_raw_tile(info, chunk)
    with info.path.open("rb") as fh:
        fh.seek(chunk.offset)
        raw = fh.read(chunk.byte_count)
    if info.compression in {JPEG, JPEG_LOSSLESS}:
        tile = _decode_jpeg(raw)
    elif info.compression == JPEG_2000:
        tile = _decode_jpeg2000(raw)
    elif info.compression in {PNG, BMP}:
        tile = _decode_cv2_image(raw)
    else:
        raise NotImplementedError(f"ETS compression code {info.compression} is not implemented yet")
    tile = np.asarray(tile)
    if info.is_rgb_encoded and tile.ndim == 3:
        return tile[..., min(max(c, 0), tile.shape[2] - 1)]
    return tile


def _read_raw_tile(info: EtsInfo, chunk: EtsChunk) -> np.ndarray:
    expected = info.tile_width * info.tile_height * info.dtype.itemsize
    byte_count = min(chunk.byte_count, expected)
    with info.path.open("rb") as fh:
        fh.seek(chunk.offset)
        raw = fh.read(byte_count)
    arr = np.frombuffer(raw, dtype=info.dtype)
    needed = info.tile_width * info.tile_height
    if arr.size < needed:
        padded = np.zeros(needed, dtype=info.dtype)
        padded[: arr.size] = arr
        arr = padded
    return arr[:needed].reshape(info.tile_height, info.tile_width)


def _decode_jpeg(raw: bytes) -> np.ndarray:
    if imagecodecs is not None:
        return np.asarray(imagecodecs.jpeg_decode(raw))
    return _decode_cv2_image(raw)


def _decode_jpeg2000(raw: bytes) -> np.ndarray:
    if imagecodecs is not None:
        return np.asarray(imagecodecs.jpeg2k_decode(raw))
    return _decode_cv2_image(raw)


def _decode_cv2_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Compressed ETS tile could not be decoded")
    if image.ndim == 3 and image.shape[2] >= 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _read_tiff_preview_plane(path: Path, *, c: int) -> np.ndarray:
    arr = tifffile.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[..., min(max(c, 0), arr.shape[-1] - 1)]
    return arr


def _compose_channels(channels: list[np.ndarray], metadata: dict[str, Any]) -> np.ndarray:
    if not channels:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if len(channels) == 1:
        names = metadata.get("channel_names") or []
        name = names[0] if names else ""
        color = _channel_color(str(name), 0)
        plane = channels[0].astype(np.float32)
        return np.dstack(
            [
                np.clip(plane * (color[0] / 255.0), 0, 255).astype(np.uint8),
                np.clip(plane * (color[1] / 255.0), 0, 255).astype(np.uint8),
                np.clip(plane * (color[2] / 255.0), 0, 255).astype(np.uint8),
            ]
        )
    colors = [_channel_color(name, i) for i, name in enumerate(metadata.get("channel_names") or [])]
    while len(colors) < len(channels):
        colors.append(_channel_color("", len(colors)))
    canvas = np.zeros((*channels[0].shape, 3), dtype=np.float32)
    for plane, color in zip(channels, colors, strict=False):
        plane_f = plane.astype(np.float32)
        canvas[..., 0] += plane_f * (color[0] / 255.0)
        canvas[..., 1] += plane_f * (color[1] / 255.0)
        canvas[..., 2] += plane_f * (color[2] / 255.0)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        plane = _normalize_plane(image)
        return np.dstack([plane] * 3)
    if image.shape[-1] >= 3:
        return _normalize_color_image(image[..., :3])
    return np.dstack([_normalize_plane(image[..., 0])] * 3)


def _normalize_plane(plane: np.ndarray) -> np.ndarray:
    arr = np.asarray(plane, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    lo, hi = np.percentile(arr, [1.0, 99.8])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)


def _normalize_color_image(image: np.ndarray) -> np.ndarray:
    return np.dstack([_normalize_plane(image[..., idx]) for idx in range(3)])


def _resize_preview(rgb: np.ndarray, preview_height: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        return rgb
    scale = float(preview_height) / float(height)
    out_height = max(64, int(round(height * scale)))
    out_width = max(64, int(round(width * scale)))
    return cv2.resize(rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)


def _channel_color(name: str, index: int) -> tuple[int, int, int]:
    low = str(name).strip().lower()
    if any(token in low for token in ("dapi", "hoechst", "405")):
        return 0, 128, 255
    if any(token in low for token in ("488", "fitc", "gfp")):
        return 0, 255, 0
    if any(token in low for token in ("tritc", "cy3", "568", "594")):
        return 255, 128, 0
    if any(token in low for token in ("cy5", "647", "660")):
        return 255, 0, 128
    return [(0, 255, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0)][index % 4]


def _bounded(name: str, value: int, size: int) -> int:
    value = int(value)
    if value < 0 or value >= max(size, 1):
        raise IndexError(f"{name} index {value} out of range for size {size}")
    return value


def _read_u32(fh) -> int:
    return int.from_bytes(fh.read(4), "little", signed=False)


def _read_i32(fh) -> int:
    return int.from_bytes(fh.read(4), "little", signed=True)


def _read_u64(fh) -> int:
    return int.from_bytes(fh.read(8), "little", signed=False)


def _native_preview_cache_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "olympus_browser_qt_native_preview_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
