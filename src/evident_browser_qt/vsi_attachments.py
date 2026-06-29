from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct


_FILE_HEADER_ATTACHMENT_DIR_OFFSET = 104
_FILE_HEADER_SIZE = 512
_ATTACHMENT_DIR_HEADER_SIZE = 288
_ATTACHMENT_ENTRY_SIZE = 128
_ATTACHMENT_SEGMENT_DATA_OFFSET = 288


@dataclass(frozen=True)
class CziAttachment:
    name: str
    file_type: str
    file_position: int
    data_size: int | None = None


def list_vsi_attachments(path: Path) -> list[CziAttachment]:
    data = path.read_bytes()
    if len(data) < _FILE_HEADER_SIZE:
        return []
    if not data.startswith(b"ZISRAWFILE"):
        return []

    attachment_dir_position = struct.unpack_from("<q", data, _FILE_HEADER_ATTACHMENT_DIR_OFFSET)[0]
    if attachment_dir_position <= 0 or attachment_dir_position >= len(data):
        return []

    if not data[attachment_dir_position : attachment_dir_position + 16].startswith(b"ZISRAWATTDIR"):
        return []

    entry_count = struct.unpack_from("<i", data, attachment_dir_position + 32)[0]
    if entry_count <= 0:
        return []

    attachments: list[CziAttachment] = []
    entries_offset = attachment_dir_position + _ATTACHMENT_DIR_HEADER_SIZE
    for index in range(entry_count):
        offset = entries_offset + index * _ATTACHMENT_ENTRY_SIZE
        if offset + _ATTACHMENT_ENTRY_SIZE > len(data):
            break
        schema = data[offset : offset + 2]
        if schema != b"A1":
            continue
        file_position = struct.unpack_from("<q", data, offset + 12)[0]
        file_type = _read_c_string(data[offset + 40 : offset + 48])
        name = _read_c_string(data[offset + 48 : offset + 128])
        if not name:
            continue
        attachments.append(
            CziAttachment(
                name=name,
                file_type=file_type.lower(),
                file_position=file_position,
                data_size=_read_attachment_data_size(data, file_position),
            )
        )
    return attachments


def extract_embedded_preview(path: Path, cache_dir: Path) -> Path | None:
    preferred = {"thumbnail", "slidepreview"}
    attachments = [att for att in list_vsi_attachments(path) if att.name.lower() in preferred]
    if not attachments:
        return None

    attachment = sorted(
        attachments,
        key=lambda att: (att.name.lower() != "thumbnail", att.name.lower() != "slidepreview"),
    )[0]
    data = path.read_bytes()
    payload = _read_attachment_payload(data, attachment)
    if payload is None:
        return None

    extension = _normalized_extension(attachment.file_type)
    digest = hashlib.sha1((str(path.resolve()) + attachment.name + extension).encode("utf-8")).hexdigest()
    out_path = cache_dir / f"{digest}.{extension}"
    if not out_path.exists():
        out_path.write_bytes(payload)
    return out_path


def _read_attachment_data_size(data: bytes, file_position: int) -> int | None:
    if file_position <= 0 or file_position + _ATTACHMENT_SEGMENT_DATA_OFFSET > len(data):
        return None
    return struct.unpack_from("<q", data, file_position + 32)[0]


def _read_attachment_payload(data: bytes, attachment: CziAttachment) -> bytes | None:
    file_position = attachment.file_position
    data_size = attachment.data_size
    if data_size is None or data_size <= 0:
        return None
    payload_offset = file_position + _ATTACHMENT_SEGMENT_DATA_OFFSET
    payload_end = payload_offset + data_size
    if file_position <= 0 or payload_end > len(data):
        return None
    return data[payload_offset:payload_end]


def _normalized_extension(file_type: str) -> str:
    cleaned = (file_type or "bin").strip("\x00 ").lower()
    if cleaned in {"jpg", "jpeg"}:
        return "jpg"
    if cleaned in {"png", "tif", "tiff", "bmp"}:
        return cleaned
    return "bin"


def _read_c_string(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", "ignore").strip()