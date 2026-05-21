"""Olympus path scanning and native VSI/ETS metadata adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .vsi_metadata import extract_vsi_xml_metadata
from .vsi_native import companion_ets_files
from .metadata import context_from_metadata
from .models import OlympusImageContext

VSI_EXTENSIONS = {".vsi"}
LEICA_EXTENSIONS = VSI_EXTENSIONS
IGNORED_NAME_PARTS = (
    "metadata",
    "_pmd_",
    "_histo",
    "_environmetalgraph",
    "iomanagerconfiguation",
    "iomanagerconfiguration",
)


@dataclass
class OlympusTreeNode:
    name: str
    kind: str
    path: Path | None = None
    internal_path: str = ""
    image_id: str | None = None
    context: OlympusImageContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    folder_metadata: dict[str, Any] | None = None
    metadata_loaded: bool = False
    warning: str | None = None
    children: list["OlympusTreeNode"] = field(default_factory=list)

    @property
    def is_image(self) -> bool:
        return self.context is not None


class OlympusAdapter:
    """Compatibility shim retained for callers that inject an old adapter."""

    def __init__(self) -> None:
        self._helpers = None

    def _load_helpers(self):
        return None

    def read_tree(self, path: Path, folder_uuid: str | None = None) -> dict[str, Any]:
        raise RuntimeError("Olympus VSI tree parsing is not implemented yet.")

    def read_image_metadata(
        self,
        path: Path,
        image_uuid: str | None,
        folder_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {}


class OlympusGateway:
    """Scan Olympus roots and bridge the UI to native VSI metadata."""

    def __init__(self, adapter: OlympusAdapter | None = None) -> None:
        self.adapter = adapter or OlympusAdapter()

    def scan_roots(self, roots: Iterable[str | Path] | None = None) -> list[OlympusTreeNode]:
        paths = [Path(p).expanduser() for p in roots] if roots else [Path.cwd()]
        nodes: list[OlympusTreeNode] = []
        for path in paths:
            nodes.extend(self.scan_path(path))
        return nodes

    def scan_path(self, path: str | Path) -> list[OlympusTreeNode]:
        root = Path(path).expanduser()
        if root.is_dir():
            return [self._scan_directory(root)]
        if root.is_file() and root.suffix.lower() in VSI_EXTENSIONS:
            return [self.container_node(root)]
        if not root.exists():
            return [
                OlympusTreeNode(
                    name=root.name or str(root),
                    kind="warning",
                    path=root,
                    warning=f"Path does not exist: {root}",
                )
            ]
        return []

    def _scan_directory(self, path: Path) -> OlympusTreeNode:
        node = OlympusTreeNode(name=path.name or str(path), kind="folder", path=path)
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            node.warning = str(exc)
            return node

        hidden_companion_dirs = {
            f"_{child.stem}_".lower()
            for child in entries
            if child.is_file() and child.suffix.lower() in VSI_EXTENSIONS
        }
        for child in entries:
            if self._ignore_name(child.name):
                continue
            if child.is_dir() and child.name.lower() in hidden_companion_dirs:
                continue
            if child.is_dir():
                node.children.append(self._scan_directory(child))
            elif child.suffix.lower() in VSI_EXTENSIONS:
                node.children.append(self.container_node(child))
        return node

    def container_node(self, path: str | Path) -> OlympusTreeNode:
        container = Path(path)
        node = OlympusTreeNode(
            name=container.name,
            kind="container",
            path=container,
            internal_path=container.name,
        )

        ets_files = companion_ets_files(container)
        if len(ets_files) > 1:
            for index, ets_file in enumerate(ets_files):
                stack_name = ets_file.parent.name
                image_name = f"{container.stem} / {stack_name}"
                image_id = f"{container.stem}:{stack_name}:{index}"
                internal_path = f"{container.name}/{stack_name}"
                metadata = self._safe_image_metadata(container, None, {}, {}, ets_file=ets_file)
                metadata["save_child_name"] = image_name
                metadata["name"] = image_name
                node.children.append(
                    OlympusTreeNode(
                        name=image_name,
                        kind=self._image_kind(container),
                        path=container,
                        internal_path=internal_path,
                        image_id=image_id,
                        context=context_from_metadata(
                            name=image_name,
                            container_path=container,
                            internal_path=internal_path,
                            image_id=image_id,
                            kind=self._image_kind(container),
                            metadata=metadata,
                        ),
                        metadata=metadata,
                        metadata_loaded=True,
                    )
                )
        else:
            metadata = self._safe_image_metadata(container, None, {}, {})
            image_name = container.stem or container.name
            internal_path = f"{container.name}/{image_name}"
            node.children.append(
                OlympusTreeNode(
                    name=image_name,
                    kind=self._image_kind(container),
                    path=container,
                    internal_path=internal_path,
                    image_id=image_name,
                    context=context_from_metadata(
                        name=image_name,
                        container_path=container,
                        internal_path=internal_path,
                        image_id=image_name,
                        kind=self._image_kind(container),
                        metadata=metadata,
                    ),
                    metadata=metadata,
                    metadata_loaded=True,
                )
            )
        return node

    def children_for_folder(
        self,
        container: Path,
        folder_uuid: str,
        parent_internal_path: str,
    ) -> list[OlympusTreeNode]:
        return []

    def _children_from_metadata(
        self,
        container: Path,
        folder_metadata: dict[str, Any],
        parent_internal_path: str,
    ) -> list[OlympusTreeNode]:
        return []

    def hydrate_image_node(self, node: OlympusTreeNode) -> OlympusImageContext | None:
        """Load full image metadata for an image node on demand."""

        if node.context is None:
            return None
        if node.metadata_loaded:
            return node.context
        if node.path is None:
            node.metadata_loaded = True
            return node.context

        metadata = self._safe_image_metadata(
            node.path,
            node.image_id,
            node.folder_metadata or {},
            node.metadata,
        )
        node.metadata = metadata
        node.metadata_loaded = True
        node.context = context_from_metadata(
            name=node.name,
            container_path=node.context.container_path,
            internal_path=node.internal_path,
            image_id=node.image_id,
            kind=node.kind,
            metadata=metadata,
        )
        return node.context

    def _lightweight_image_metadata(
        self,
        container: Path,
        source: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(source)
        metadata.setdefault("filetype", container.suffix.lower())
        metadata.setdefault("source_file", str(container))
        return metadata

    def _safe_image_metadata(
        self,
        container: Path,
        image_uuid: str | None,
        folder_metadata: dict[str, Any],
        fallback: dict[str, Any],
        ets_file: Path | None = None,
    ) -> dict[str, Any]:
        metadata = dict(fallback)
        try:
            stat = container.stat()
        except OSError as exc:
            metadata["warning"] = str(exc)
            stat = None
        try:
            metadata.update(extract_vsi_xml_metadata(container, ets_path=ets_file))
        except Exception as exc:
            metadata.setdefault("filetype", container.suffix.lower())
            metadata.setdefault("source_file", str(container))
            metadata.setdefault("save_child_name", container.stem or container.name)
            metadata.setdefault("name", container.stem or container.name)
            metadata.setdefault("size_x", 512)
            metadata.setdefault("size_y", 384)
            metadata.setdefault("size_z", 1)
            metadata.setdefault("size_c", 1)
            metadata.setdefault("size_t", 1)
            metadata.setdefault("size_s", 1)
            metadata.setdefault(
                "dimensions",
                {"x": metadata["size_x"], "y": metadata["size_y"], "z": 1, "c": 1, "t": 1, "s": 1},
            )
            metadata.setdefault("channel_names", ["Channel 1"])
            metadata.setdefault("pixel_type", "u1")
            metadata.setdefault("placeholder_size_x", 512)
            metadata.setdefault("placeholder_size_y", 384)
            metadata.setdefault("backend_status", "placeholder")
            metadata.setdefault(
                "warning",
                f"Embedded VSI XML metadata could not be parsed; using placeholder metadata. {exc}",
            )
        if stat is not None:
            metadata["file_size_bytes"] = int(stat.st_size)
        return metadata

    def _safe_vsi_metadata(self, container: Path) -> dict[str, Any]:
        return self._safe_image_metadata(container, None, {}, {})

    def read_thumbnail(self, context: OlympusImageContext, max_size: int = 512):
        from .preview import preview_png_from_metadata

        preview_path = preview_png_from_metadata(
            context.metadata,
            selected_s=context.selected_s,
            preview_height=max_size,
        )
        try:
            import cv2

            image = cv2.imread(str(preview_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                return image
        except ImportError:
            pass
        return np.asarray(preview_path)

    def read_plane(
        self,
        context: OlympusImageContext,
        z: int = 0,
        c: int = 0,
        t: int = 0,
        s: int | None = None,
    ):
        from .olympus_pixels import read_olympus_plane

        return read_olympus_plane(context, z=z, c=c, t=t, s=s)

    def read_array(self, context: OlympusImageContext, s: int | None = None):
        from .olympus_pixels import read_olympus_array

        return read_olympus_array(context, s=s)

    @staticmethod
    def _ignore_name(name: str) -> bool:
        low = name.lower()
        return any(part in low for part in IGNORED_NAME_PARTS)

    @staticmethod
    def _image_kind(container: Path) -> str:
        return {
            ".vsi": "vsi-image",
        }.get(container.suffix.lower(), "olympus-image")
