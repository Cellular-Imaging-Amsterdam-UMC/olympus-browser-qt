from __future__ import annotations

from dataclasses import replace
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import QEventLoop, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .olympus_gateway import VSI_EXTENSIONS, OlympusGateway, OlympusTreeNode
from .icons import make_app_icon
from .metadata import format_metadata_summary
from .models import OlympusImageContext, OlympusImageHandle
from .preview import preview_png_from_metadata

NODE_ROLE = int(Qt.ItemDataRole.UserRole)
CONTEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
PLACEHOLDER_TEXT = "..."
RECENT_ROOTS_KEY = "recent_roots"
MAX_RECENT_ROOTS = 10
SETTINGS_VENDOR = "NL-BioImaging"
SETTINGS_APP = "olympus-browser-qt"
LEGACY_SETTINGS_APP = "olympus-browser-qt"


class PreviewWorker(QThread):
    previewReady = pyqtSignal(int, int, str)
    error = pyqtSignal(int, str)

    def __init__(self, job_id: int, context: OlympusImageContext, heights: list[int]) -> None:
        super().__init__()
        self.job_id = job_id
        self.context = context
        self.heights = heights

    def run(self) -> None:
        try:
            for height in self.heights:
                if self.isInterruptionRequested():
                    break
                path = preview_png_from_metadata(
                    self.context.metadata,
                    selected_s=self.context.selected_s,
                    preview_height=height,
                )
                self.previewReady.emit(self.job_id, height, str(path))
                if height != self.heights[-1]:
                    QThread.msleep(120)
        except Exception:
            self.error.emit(self.job_id, traceback.format_exc())


class MetadataHydrateWorker(QThread):
    finishedHydrating = pyqtSignal(object, object)

    def __init__(self, gateway: OlympusGateway, nodes: list[OlympusTreeNode]) -> None:
        super().__init__()
        self.gateway = gateway
        self.nodes = nodes

    def run(self) -> None:
        try:
            contexts = []
            for node in self.nodes:
                if self.isInterruptionRequested():
                    return
                context = self.gateway.hydrate_image_node(node)
                if context is not None:
                    contexts.append(context)
            self.finishedHydrating.emit(contexts, None)
        except Exception:
            self.finishedHydrating.emit(None, traceback.format_exc())


class OlympusBrowserDialog(QDialog):
    """Reusable browser dialog for selecting Olympus image contexts."""

    def __init__(
        self,
        roots: Iterable[str | Path] | None = None,
        selection_mode: str = "single",
        parent: QWidget | None = None,
        gateway: OlympusGateway | None = None,
    ) -> None:
        super().__init__(parent)
        if selection_mode not in {"single", "multiple"}:
            raise ValueError("selection_mode must be 'single' or 'multiple'")

        root_list = list(roots) if roots is not None else None
        self.gateway = gateway or OlympusGateway()
        self.selection_mode = selection_mode
        self._preview_worker: PreviewWorker | None = None
        self._hydrate_worker: MetadataHydrateWorker | None = None
        self._stale_preview_workers: list[PreviewWorker] = []
        self._preview_job_id = 0
        self._current_file: Path | None = None
        self._accepted_contexts: list[OlympusImageContext] | None = None
        self._browser_s_selection: dict[str, int | None] = {}
        self._settings = QSettings(SETTINGS_VENDOR, SETTINGS_APP)
        self._legacy_settings = QSettings(SETTINGS_VENDOR, LEGACY_SETTINGS_APP)
        self._current_root = self._initial_root(root_list)
        self._initial_files = self._initial_file_roots(root_list)
        self._recent_roots = self._load_recent_roots()

        self.setWindowTitle("Browse Olympus VSI Images")
        self.setWindowIcon(make_app_icon())
        self.resize(1180, 760)
        self._build_ui()
        self._apply_style()
        self.populate_fs_root()
        if self._initial_files:
            self.load_file_images(self._initial_files[0])

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_browse_root = QPushButton("Browse...")
        self.btn_browse_root.setFixedWidth(100)
        self.btn_browse_root.clicked.connect(self.choose_root)
        top.addWidget(self.btn_browse_root)

        top.addWidget(QLabel("Recent:"))
        self.recent_roots_combo = QComboBox()
        self.recent_roots_combo.setMinimumWidth(220)
        self.recent_roots_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.recent_roots_combo.currentIndexChanged.connect(self.on_recent_root_selected)
        top.addWidget(self.recent_roots_combo)
        self._refresh_recent_roots_combo()

        self.lbl_root = QLabel(f"Root: {self._current_root}")
        self.lbl_root.setWordWrap(True)
        top.addWidget(self.lbl_root, 1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        top.addWidget(self.btn_refresh)

        top.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter Olympus contents")
        self.filter_edit.textChanged.connect(self.apply_content_filter)
        top.addWidget(self.filter_edit, 1)

        self.select_all_below_button = QPushButton("Select all images below")
        self.select_all_below_button.clicked.connect(self.select_all_images_below)
        self.select_all_below_button.setVisible(self.selection_mode == "multiple")
        top.addWidget(self.select_all_below_button)
        outer.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_folders = QLabel("Folders and VSI files:")
        left_layout.addWidget(self.lbl_folders)
        self.tree_fs = QTreeWidget()
        self.tree_fs.setHeaderHidden(True)
        self.tree_fs.itemExpanded.connect(self.on_fs_item_expanded)
        self.tree_fs.itemSelectionChanged.connect(self.on_fs_selection_changed)
        self.tree_fs.itemDoubleClicked.connect(self.on_fs_item_double_clicked)
        left_layout.addWidget(self.tree_fs, 1)
        splitter.addWidget(left)

        right_split = QSplitter(Qt.Orientation.Horizontal)
        contents = QWidget()
        contents_layout = QVBoxLayout(contents)
        contents_layout.setContentsMargins(0, 0, 0, 0)
        contents_layout.addWidget(QLabel("Contents of selected VSI file:"))
        self.tree_images = QTreeWidget()
        self.tree_images.setHeaderHidden(True)
        self.tree_images.setSelectionMode(
            QTreeWidget.SelectionMode.SingleSelection
            if self.selection_mode == "single"
            else QTreeWidget.SelectionMode.ExtendedSelection
        )
        self.tree_images.itemExpanded.connect(self.on_content_item_expanded)
        self.tree_images.itemSelectionChanged.connect(self.on_image_selection_changed)
        self.tree_images.itemDoubleClicked.connect(self.on_image_double_clicked)
        contents_layout.addWidget(self.tree_images, 1)
        right_split.addWidget(contents)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel("Select an image to preview it")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(420, 300)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.preview_label, 2)

        self._s_controls = QWidget()
        s_controls_layout = QHBoxLayout(self._s_controls)
        s_controls_layout.setContentsMargins(0, 0, 0, 0)
        s_controls_layout.addWidget(QLabel("S"))
        self._s_mode_combo = QComboBox()
        self._s_mode_combo.addItems(["All", "Fixed"])
        self._s_mode_combo.setMinimumContentsLength(7)
        self._s_mode_combo.setMinimumWidth(88)
        self._s_mode_combo.currentIndexChanged.connect(self._on_browser_s_mode_changed)
        s_controls_layout.addWidget(self._s_mode_combo)
        self._s_slider = QSlider(Qt.Orientation.Horizontal)
        self._s_slider.setRange(0, 0)
        self._s_slider.setSingleStep(1)
        self._s_slider.setPageStep(1)
        self._s_slider.valueChanged.connect(self._on_browser_s_slider_changed)
        s_controls_layout.addWidget(self._s_slider, 1)
        self._s_spin = QSpinBox()
        self._s_spin.setRange(1, 1)
        self._s_spin.setMinimumWidth(72)
        self._s_spin.valueChanged.connect(self._on_browser_s_spin_changed)
        s_controls_layout.addWidget(self._s_spin)
        self._s_value_label = QLabel("All")
        s_controls_layout.addWidget(self._s_value_label)
        self._s_controls.setVisible(False)
        preview_layout.addWidget(self._s_controls, 0)

        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setPlaceholderText("Image metadata will appear here")
        self.metadata_text.setMaximumHeight(150)
        preview_layout.addWidget(self.metadata_text, 0)
        right_split.addWidget(preview)
        right_split.setStretchFactor(0, 36)
        right_split.setStretchFactor(1, 64)
        right_split.setSizes([265, 480])

        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 36)
        splitter.setStretchFactor(1, 64)
        splitter.setSizes([425, 745])
        outer.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("No image selected")
        footer.addWidget(self.status_label, 1)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        footer.addWidget(self.button_box)
        outer.addLayout(footer)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            "QDialog { background: #202020; color: #f0f0f0; }"
            "QLabel { color: #f0f0f0; }"
            "QTreeWidget, QTableWidget, QTextEdit, QLineEdit, QComboBox {"
            "background: #2d2d2d; color: #f7f7f7; border: 1px solid #555;"
            "selection-background-color: #3d5068; }"
            "QHeaderView::section { background: #1e293b; color: #e2e8f0; border: 0; padding: 4px; }"
            "QPushButton { background: #0d47a1; color: white; border: 1px solid #1e5bb8;"
            "border-radius: 4px; padding: 6px 10px; }"
            "QPushButton:hover { background: #1565c0; }"
            "QPushButton:disabled { background: #444; color: #888; border-color: #555; }"
        )

    # ------------------------------------------------------------------
    # Root and filesystem tree
    # ------------------------------------------------------------------

    def _initial_root(self, roots: Iterable[str | Path] | None) -> Path:
        if roots:
            first = Path(next(iter(roots))).expanduser()
            if first.is_file():
                return first.parent
            return first
        remembered = self._settings.value("last_root", "", str)
        if remembered:
            path = Path(remembered).expanduser()
            if self._is_usable_root(path):
                return path
        return Path.home()

    def _initial_file_roots(self, roots: Iterable[str | Path] | None) -> list[Path]:
        files: list[Path] = []
        if roots:
            for root in roots:
                path = Path(root).expanduser()
                if path.is_file() and path.suffix.lower() in VSI_EXTENSIONS:
                    files.append(path)
        return files

    def choose_root(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose root folder", str(self._current_root))
        if chosen:
            self._current_root = Path(chosen)
            self._remember_root(self._current_root)
            self.lbl_root.setText(f"Root: {self._current_root}")
            self.refresh()

    def on_recent_root_selected(self, index: int) -> None:
        path_text = self.recent_roots_combo.itemData(index)
        if not path_text:
            return
        path = Path(str(path_text)).expanduser()
        if path == self._current_root:
            return
        if not self._is_usable_root(path):
            self._recent_roots = [
                root for root in self._recent_roots if self._path_key(root) != self._path_key(path)
            ]
            self._store_recent_roots()
            self._refresh_recent_roots_combo()
            return
        self._current_root = path
        self._remember_root(self._current_root)
        self.lbl_root.setText(f"Root: {self._current_root}")
        self.refresh()

    def refresh(self) -> None:
        self.populate_fs_root()
        self.tree_images.clear()
        self._clear_preview()
        self.metadata_text.clear()
        self._current_file = None
        self._update_ok_state()

    def populate_fs_root(self) -> None:
        self.tree_fs.clear()
        item = QTreeWidgetItem([str(self._current_root)])
        item.setIcon(0, self.icon_folder())
        item.setData(0, NODE_ROLE, self._current_root)
        item.addChild(QTreeWidgetItem([PLACEHOLDER_TEXT]))
        self.tree_fs.addTopLevelItem(item)
        self.tree_fs.expandItem(item)

    def on_fs_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() == 1 and item.child(0).text(0) == PLACEHOLDER_TEXT:
            self._populate_fs_children(item)

    def _populate_fs_children(self, parent_item: QTreeWidgetItem) -> None:
        parent_item.takeChildren()
        parent_path = parent_item.data(0, NODE_ROLE)
        if not parent_path:
            return
        path = Path(parent_path)
        if not path.is_dir():
            return

        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            parent_item.addChild(QTreeWidgetItem([f"Warning: {exc}"]))
            return

        hidden_companion_dirs = {
            f"_{entry.stem}_".lower()
            for entry in entries
            if entry.is_file() and entry.suffix.lower() in VSI_EXTENSIONS
        }
        for entry in entries:
            if self.gateway._ignore_name(entry.name):
                continue
            if entry.is_dir() and entry.name.lower() in hidden_companion_dirs:
                continue
            if entry.is_dir():
                item = QTreeWidgetItem([entry.name])
                item.setIcon(0, self.icon_folder())
                item.setData(0, NODE_ROLE, entry)
                item.addChild(QTreeWidgetItem([PLACEHOLDER_TEXT]))
                parent_item.addChild(item)
            elif entry.suffix.lower() in VSI_EXTENSIONS:
                self._add_fs_file(parent_item, entry)

    def _add_fs_file(self, parent_item: QTreeWidgetItem, path: Path) -> None:
        item = QTreeWidgetItem([path.name])
        item.setIcon(0, self.icon_for_file(path.suffix))
        item.setData(0, NODE_ROLE, path)
        parent_item.addChild(item)

    def on_fs_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        path = item.data(0, NODE_ROLE)
        if not path:
            return
        path = Path(path)
        if path.is_dir():
            self._current_root = path
            self._remember_root(self._current_root)
            self.lbl_root.setText(f"Root: {self._current_root}")
            self.refresh()
        elif self._is_vsi_container(path):
            self.load_file_images(path)

    def on_fs_selection_changed(self) -> None:
        items = self.tree_fs.selectedItems()
        if not items:
            return
        path = items[0].data(0, NODE_ROLE)
        if path and self._is_vsi_container(Path(path)):
            self.load_file_images(Path(path))

    # ------------------------------------------------------------------
    # Olympus content tree
    # ------------------------------------------------------------------

    def load_file_images(self, path: str | Path) -> None:
        container = Path(path)
        self._current_file = container
        self._remember_root(container.parent if container.is_file() else container)
        self.tree_images.clear()
        self._clear_preview()
        self.metadata_text.clear()
        self.preview_label.setText(f"Loading {container.name}...")
        QApplication.processEvents()

        node = self.gateway.container_node(container)
        root_item = self._content_item_from_node(node, is_root=True)
        self.tree_images.addTopLevelItem(root_item)
        self.tree_images.expandItem(root_item)
        self.preview_label.setText("Select an image to preview it")
        self.apply_content_filter()
        self._auto_select_single_root_image(root_item)
        self._update_ok_state()

    def _content_item_from_node(self, node: OlympusTreeNode, *, is_root: bool = False) -> QTreeWidgetItem:
        text = node.name if not node.warning else f"{node.name}  [{node.warning}]"
        item = QTreeWidgetItem([text])
        item.setData(0, NODE_ROLE, node)
        if node.context is not None:
            item.setData(0, CONTEXT_ROLE, node.context)
            item.setIcon(0, self.icon_image())
        elif node.kind == "container":
            item.setIcon(0, self.icon_for_file(Path(node.name).suffix))
        elif node.kind == "folder":
            item.setIcon(0, self.icon_folder())
        else:
            item.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))

        if node.context is None and not is_root:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)

        for child in node.children:
            item.addChild(self._content_item_from_node(child))
        if node.kind == "folder" and not node.children:
            item.addChild(QTreeWidgetItem([PLACEHOLDER_TEXT]))
        return item

    def _auto_select_single_root_image(self, root_item: QTreeWidgetItem) -> None:
        if root_item.childCount() != 1:
            return
        image_item = root_item.child(0)
        if not isinstance(image_item.data(0, CONTEXT_ROLE), OlympusImageContext):
            return
        self.tree_images.clearSelection()
        self.tree_images.setCurrentItem(image_item)
        image_item.setSelected(True)

    def on_content_item_expanded(self, item: QTreeWidgetItem) -> None:
        node = item.data(0, NODE_ROLE)
        if not isinstance(node, OlympusTreeNode) or node.kind != "folder":
            return
        if item.childCount() != 1 or item.child(0).text(0) != PLACEHOLDER_TEXT:
            return
        if not node.path or not node.image_id:
            return
        item.takeChildren()
        try:
            children = self.gateway.children_for_folder(node.path, node.image_id, node.internal_path)
            node.children = children
            for child in children:
                item.addChild(self._content_item_from_node(child))
        except Exception as exc:
            warn = QTreeWidgetItem([f"Warning: {exc}"])
            warn.setIcon(0, self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
            item.addChild(warn)

    def selected_contexts(self) -> list[OlympusImageContext]:
        if self._accepted_contexts is not None:
            return list(self._accepted_contexts)
        return self._selected_contexts(hydrate=True)

    def _selected_contexts(self, *, hydrate: bool) -> list[OlympusImageContext]:
        contexts: list[OlympusImageContext] = []
        for item in self.tree_images.selectedItems():
            context = self._context_for_item(item, hydrate=hydrate)
            if isinstance(context, OlympusImageContext):
                context = self._derive_context_with_browser_selection(context)
                if context not in contexts:
                    contexts.append(context)
        return contexts

    def selected_context(self) -> OlympusImageContext | None:
        contexts = self.selected_contexts()
        return contexts[0] if contexts else None

    def on_image_selection_changed(self) -> None:
        items = [
            item
            for item in self.tree_images.selectedItems()
            if self._context_for_item(item, hydrate=False)
        ]
        contexts = [self._context_for_item(item, hydrate=False) for item in items]
        self._update_browser_s_controls(items[0] if len(items) == 1 else None)
        if len(items) == 1 and isinstance(contexts[0], OlympusImageContext):
            self.show_context(items[0])
        elif len(contexts) > 1:
            self._cancel_preview_worker()
            self.preview_label.setText(f"{len(contexts)} images selected")
            self.metadata_text.clear()
        else:
            self._cancel_preview_worker()
            self._clear_preview()
            self.metadata_text.clear()
        self._update_ok_state()

    def on_image_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if self.selection_mode == "single" and isinstance(item.data(0, CONTEXT_ROLE), OlympusImageContext):
            self.accept()

    def select_all_images_below(self) -> None:
        if self.selection_mode != "multiple":
            return
        roots = self.tree_images.selectedItems() or [
            self.tree_images.topLevelItem(i) for i in range(self.tree_images.topLevelItemCount())
        ]
        self.tree_images.blockSignals(True)
        self.tree_images.clearSelection()

        def select_images(item: QTreeWidgetItem) -> None:
            if isinstance(item.data(0, CONTEXT_ROLE), OlympusImageContext):
                item.setSelected(True)
            for idx in range(item.childCount()):
                select_images(item.child(idx))

        for root in roots:
            select_images(root)
        self.tree_images.blockSignals(False)
        self.on_image_selection_changed()

    def apply_content_filter(self) -> None:
        text = self.filter_edit.text().strip().lower()

        def update(item: QTreeWidgetItem) -> bool:
            own_match = not text or text in item.text(0).lower()
            child_match = False
            for idx in range(item.childCount()):
                child_match = update(item.child(idx)) or child_match
            visible = own_match or child_match
            item.setHidden(not visible)
            return visible

        for idx in range(self.tree_images.topLevelItemCount()):
            update(self.tree_images.topLevelItem(idx))

    # ------------------------------------------------------------------
    # Preview and metadata
    # ------------------------------------------------------------------

    def show_context(self, item: QTreeWidgetItem) -> None:
        context = self._context_for_item(item, hydrate=False)
        if context is None:
            return
        context = self._derive_context_with_browser_selection(context)
        self._populate_metadata(context.metadata)
        self._start_preview(context)

    def _populate_metadata(self, metadata: dict) -> None:
        self.metadata_text.setPlainText(format_metadata_summary(metadata))

    def _context_for_item(
        self,
        item: QTreeWidgetItem,
        *,
        hydrate: bool,
    ) -> OlympusImageContext | None:
        context = item.data(0, CONTEXT_ROLE)
        if not isinstance(context, OlympusImageContext):
            return None
        if not hydrate:
            return context

        node = item.data(0, NODE_ROLE)
        if isinstance(node, OlympusTreeNode) and not node.metadata_loaded:
            loaded = self.gateway.hydrate_image_node(node)
            if loaded is not None:
                item.setData(0, CONTEXT_ROLE, loaded)
                item.setData(0, NODE_ROLE, node)
                context = loaded
        return context

    def _start_preview(self, context: OlympusImageContext) -> None:
        self._cancel_preview_worker()
        self._preview_job_id += 1
        job_id = self._preview_job_id
        heights = self._preview_heights(context)
        self.preview_label.setText(f"Loading preview {heights[0]}px...")
        self._preview_worker = PreviewWorker(job_id, context, heights)
        self._preview_worker.previewReady.connect(self._on_preview_ready)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_ready(self, job_id: int, height: int, path: str) -> None:
        if job_id != self._preview_job_id:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview_label.setText("Preview unavailable")
            return
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.preview_label.setToolTip(f"Preview height: {height}px")

    def _on_preview_error(self, job_id: int, message: str) -> None:
        if job_id != self._preview_job_id:
            return
        last_line = message.strip().splitlines()[-1] if message.strip() else "Preview unavailable"
        self.preview_label.setText(last_line)

    def _clear_preview(self) -> None:
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setToolTip("")
        self.preview_label.setText("Select an image to preview it")

    def _preview_heights(self, context: OlympusImageContext) -> list[int]:
        steps = [24, 112, 256, 512]
        sx = context.size_x
        sy = context.size_y
        if sx is not None and sy is not None and sx <= 2048 and sy <= 2048:
            return [max(steps)]
        return steps

    def _context_selection_key(self, context: OlympusImageContext) -> str:
        return f"{context.container_path}|{context.internal_path}"

    def _selected_s_for_context(self, context: OlympusImageContext) -> int | None:
        key = self._context_selection_key(context)
        if key in self._browser_s_selection:
            return self._browser_s_selection[key]
        return context.selected_s

    def _derive_context_with_browser_selection(self, context: OlympusImageContext) -> OlympusImageContext:
        selected_s = self._selected_s_for_context(context)
        metadata = dict(context.metadata)
        if selected_s is None:
            metadata.pop("selected_s", None)
        else:
            metadata["selected_s"] = int(selected_s)
        return replace(context, selected_s=selected_s, metadata=metadata)

    def _set_browser_s_for_context(self, context: OlympusImageContext, selected_s: int | None) -> None:
        key = self._context_selection_key(context)
        if selected_s == context.selected_s:
            self._browser_s_selection.pop(key, None)
        else:
            self._browser_s_selection[key] = selected_s

    def _current_single_selected_item(self) -> QTreeWidgetItem | None:
        items = self.tree_images.selectedItems()
        if len(items) != 1:
            return None
        if not isinstance(self._context_for_item(items[0], hydrate=False), OlympusImageContext):
            return None
        return items[0]

    def _update_browser_s_controls(self, item: QTreeWidgetItem | None) -> None:
        context = self._context_for_item(item, hydrate=False) if item is not None else None
        size_s = context.size_s if isinstance(context, OlympusImageContext) else None
        if not isinstance(context, OlympusImageContext) or (size_s or 1) <= 1:
            self._s_controls.setVisible(False)
            return

        current_selected_s = self._selected_s_for_context(context)
        fixed = current_selected_s is not None
        slider_value = current_selected_s if current_selected_s is not None else max((size_s or 1) // 2, 0)

        self._s_controls.setVisible(True)
        self._s_mode_combo.blockSignals(True)
        self._s_slider.blockSignals(True)
        self._s_spin.blockSignals(True)
        self._s_mode_combo.setCurrentIndex(1 if fixed else 0)
        self._s_slider.setRange(0, (size_s or 1) - 1)
        self._s_slider.setSingleStep(1)
        self._s_slider.setPageStep(1)
        self._s_slider.setValue(slider_value)
        self._s_slider.setEnabled(fixed)
        self._s_spin.setRange(1, size_s or 1)
        self._s_spin.setValue(slider_value + 1)
        self._s_spin.setEnabled(fixed)
        self._s_mode_combo.blockSignals(False)
        self._s_slider.blockSignals(False)
        self._s_spin.blockSignals(False)
        self._refresh_browser_s_label(size_s or 1, current_selected_s, slider_value)

    def _refresh_browser_s_label(self, size_s: int, selected_s: int | None, slider_value: int | None = None) -> None:
        if selected_s is None:
            self._s_value_label.setText(f"All ({max(size_s // 2, 0) + 1}/{size_s} preview)")
            return
        value = selected_s if slider_value is None else slider_value
        self._s_value_label.setText(f"{value + 1}/{size_s}")

    def _apply_browser_fixed_s_value(self, context: OlympusImageContext, value: int) -> None:
        self._set_browser_s_for_context(context, value)
        self._s_slider.blockSignals(True)
        self._s_spin.blockSignals(True)
        self._s_slider.setValue(value)
        self._s_spin.setValue(value + 1)
        self._s_slider.blockSignals(False)
        self._s_spin.blockSignals(False)
        self._refresh_browser_s_label(context.size_s or 1, value, value)
        self._refresh_selected_preview()

    def _refresh_selected_preview(self) -> None:
        item = self._current_single_selected_item()
        if item is not None:
            self.show_context(item)

    def _on_browser_s_mode_changed(self, _index: int) -> None:
        item = self._current_single_selected_item()
        if item is None:
            return
        context = self._context_for_item(item, hydrate=False)
        if context is None:
            return
        if self._s_mode_combo.currentText() == "All":
            self._set_browser_s_for_context(context, None)
            self._s_slider.setEnabled(False)
            self._s_spin.setEnabled(False)
            self._refresh_browser_s_label(context.size_s or 1, None)
        else:
            value = self._s_slider.value()
            self._s_slider.setEnabled(True)
            self._s_spin.setEnabled(True)
            self._apply_browser_fixed_s_value(context, value)
            return
        self._refresh_selected_preview()

    def _on_browser_s_slider_changed(self, value: int) -> None:
        item = self._current_single_selected_item()
        if item is None:
            return
        context = self._context_for_item(item, hydrate=False)
        if context is None or self._s_mode_combo.currentText() != "Fixed":
            return
        self._apply_browser_fixed_s_value(context, value)

    def _on_browser_s_spin_changed(self, value: int) -> None:
        item = self._current_single_selected_item()
        if item is None:
            return
        context = self._context_for_item(item, hydrate=False)
        if context is None or self._s_mode_combo.currentText() != "Fixed":
            return
        self._apply_browser_fixed_s_value(context, value - 1)

    # ------------------------------------------------------------------
    # Icons and helpers
    # ------------------------------------------------------------------

    def _asset_icon(self, name: str, fallback: QStyle.StandardPixmap | None = None) -> QIcon:
        path = Path(__file__).with_name("images") / name
        if path.exists():
            return QIcon(str(path))
        if fallback is not None:
            return self.style().standardIcon(fallback)
        return QIcon()

    def icon_folder(self) -> QIcon:
        return self._asset_icon("folder.svg", QStyle.StandardPixmap.SP_DirIcon)

    def icon_image(self) -> QIcon:
        return self._asset_icon("image.svg", QStyle.StandardPixmap.SP_FileIcon)

    def icon_for_file(self, ext: str) -> QIcon:
        ext = ext.lower().lstrip(".")
        mapping = {"vsi": "file-vsi.svg"}
        return self._asset_icon(mapping.get(ext, "image.svg"), QStyle.StandardPixmap.SP_FileIcon)

    @staticmethod
    def _is_vsi_container(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in VSI_EXTENSIONS

    def _remember_root(self, root: Path) -> None:
        if self._is_usable_root(root):
            self._settings.setValue("last_root", str(root))
            self._remember_recent_root(root)

    def _load_recent_roots(self) -> list[Path]:
        raw_roots = self._settings.value(RECENT_ROOTS_KEY, [])
        if not raw_roots:
            raw_roots = self._legacy_settings.value(RECENT_ROOTS_KEY, [])
        if isinstance(raw_roots, str):
            values = [raw_roots] if raw_roots else []
        else:
            try:
                values = list(raw_roots)
            except TypeError:
                values = []

        roots: list[Path] = []
        seen: set[str] = set()
        for value in values:
            path = Path(str(value)).expanduser()
            key = self._path_key(path)
            if key in seen or not self._is_usable_root(path):
                continue
            seen.add(key)
            roots.append(path)
            if len(roots) >= MAX_RECENT_ROOTS:
                break
        return roots

    def _remember_recent_root(self, root: Path) -> None:
        if not self._is_usable_root(root):
            return
        root_key = self._path_key(root)
        self._recent_roots = [
            existing for existing in self._recent_roots if self._path_key(existing) != root_key
        ]
        self._recent_roots.insert(0, root)
        self._recent_roots = self._recent_roots[:MAX_RECENT_ROOTS]
        self._store_recent_roots()
        self._refresh_recent_roots_combo()

    def _store_recent_roots(self) -> None:
        self._settings.setValue(RECENT_ROOTS_KEY, [str(root) for root in self._recent_roots])

    def _refresh_recent_roots_combo(self) -> None:
        self.recent_roots_combo.blockSignals(True)
        self.recent_roots_combo.clear()
        if not self._recent_roots:
            self.recent_roots_combo.addItem("No recent folders", None)
            self.recent_roots_combo.setEnabled(False)
        else:
            self.recent_roots_combo.setEnabled(True)
            current_key = self._path_key(self._current_root)
            current_index = -1
            for root in self._recent_roots:
                self.recent_roots_combo.addItem(str(root), str(root))
                if self._path_key(root) == current_key:
                    current_index = self.recent_roots_combo.count() - 1
            self.recent_roots_combo.setCurrentIndex(current_index)
        self.recent_roots_combo.blockSignals(False)

    @staticmethod
    def _path_key(path: Path) -> str:
        try:
            return str(path.expanduser().resolve())
        except OSError:
            return str(path.expanduser())

    def _is_usable_root(self, root: Path) -> bool:
        """Return True when a remembered folder still looks readable/useful."""

        if not root.exists() or not root.is_dir() or self._is_source_checkout(root):
            return False
        if not os.access(root, os.R_OK):
            return False
        try:
            entries = list(root.iterdir())
        except OSError:
            return False

        for entry in entries:
            if entry.is_file() and entry.suffix.lower() in VSI_EXTENSIONS:
                try:
                    with entry.open("rb"):
                        pass
                except OSError:
                    return False
        return True

    @staticmethod
    def _is_source_checkout(path: Path) -> bool:
        """Avoid persisting this package checkout as a data browsing root."""

        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        markers = [
            resolved / "pyproject.toml",
            resolved / "src" / "olympus_browser_qt",
            resolved / "plan_olympus_browser_qt.md",
        ]
        return all(marker.exists() for marker in markers)

    def _update_ok_state(self) -> None:
        count = len(self._selected_contexts(hydrate=False))
        valid = count == 1 if self.selection_mode == "single" else count > 0
        self.ok_button.setEnabled(valid)
        self.status_label.setText(f"{count} image selected" if count == 1 else f"{count} images selected")

    def _hydrate_selected_contexts_for_accept(self) -> bool:
        items = [
            item
            for item in self.tree_images.selectedItems()
            if isinstance(item.data(0, CONTEXT_ROLE), OlympusImageContext)
        ]
        nodes = [item.data(0, NODE_ROLE) for item in items]
        image_nodes = [node for node in nodes if isinstance(node, OlympusTreeNode)]
        if not image_nodes:
            return True
        if all(node.metadata_loaded for node in image_nodes):
            self._accepted_contexts = self._selected_contexts(hydrate=False)
            return True

        progress = QProgressDialog("Loading selected image metadata...", None, 0, 0, self)
        progress.setWindowTitle("Preparing Selection")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()

        loop = QEventLoop(self)
        result: dict[str, object] = {"contexts": None, "error": None}
        worker = MetadataHydrateWorker(self.gateway, image_nodes)
        self._hydrate_worker = worker

        def finish(contexts, error) -> None:
            result["contexts"] = contexts
            result["error"] = error
            loop.quit()

        worker.finishedHydrating.connect(finish)
        worker.start()
        loop.exec()
        worker.wait(1500)
        self._hydrate_worker = None
        progress.close()

        if result["error"]:
            QMessageBox.warning(
                self,
                "Metadata Load Failed",
                f"Could not load selected image metadata:\n{result['error']}",
            )
            return False

        for item, node in zip(items, image_nodes, strict=False):
            if node.context is not None:
                item.setData(0, CONTEXT_ROLE, node.context)
                item.setData(0, NODE_ROLE, node)
        self._accepted_contexts = self._selected_contexts(hydrate=False)
        return True

    def _cancel_preview_worker(self) -> None:
        if self._preview_worker is not None:
            worker = self._preview_worker
            worker.requestInterruption()
            self._stale_preview_workers.append(worker)
            worker.finished.connect(lambda w=worker: self._forget_stale_worker(w))
            self._preview_worker = None

    def _forget_stale_worker(self, worker: PreviewWorker) -> None:
        try:
            self._stale_preview_workers.remove(worker)
        except ValueError:
            pass

    def _shutdown_workers(self) -> None:
        workers = [self._preview_worker, self._hydrate_worker, *self._stale_preview_workers]
        for worker in workers:
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(1500)
        self._preview_worker = None
        self._hydrate_worker = None
        self._stale_preview_workers.clear()

    def accept(self) -> None:
        if not self._hydrate_selected_contexts_for_accept():
            return
        self._shutdown_workers()
        super().accept()

    def reject(self) -> None:
        self._shutdown_workers()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._shutdown_workers()
        super().closeEvent(event)

    @classmethod
    def select_image_context(
        cls,
        roots: Iterable[str | Path] | None = None,
        parent: QWidget | None = None,
    ) -> OlympusImageContext | None:
        ensure_app()
        dialog = cls(roots=roots, selection_mode="single", parent=parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_context()
        return None

    @classmethod
    def select_image_contexts(
        cls,
        roots: Iterable[str | Path] | None = None,
        parent: QWidget | None = None,
    ) -> list[OlympusImageContext]:
        ensure_app()
        dialog = cls(roots=roots, selection_mode="multiple", parent=parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_contexts()
        return []

    @classmethod
    def select_image(
        cls,
        roots: Iterable[str | Path] | None = None,
        parent: QWidget | None = None,
    ) -> OlympusImageHandle | None:
        context = cls.select_image_context(roots=roots, parent=parent)
        return context.open() if context is not None else None

    @classmethod
    def select_images(
        cls,
        roots: Iterable[str | Path] | None = None,
        parent: QWidget | None = None,
    ) -> list[OlympusImageHandle]:
        return [context.open() for context in cls.select_image_contexts(roots=roots, parent=parent)]


_APP_REF: QApplication | None = None


def ensure_app() -> QApplication:
    global _APP_REF
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    _APP_REF = app
    return app


def run_dialog_as_json(paths: list[str], *, multiple: bool) -> int:
    ensure_app()
    dialog = OlympusBrowserDialog(roots=paths or None, selection_mode="multiple" if multiple else "single")
    if dialog.exec() == QDialog.DialogCode.Accepted:
        contexts = dialog.selected_contexts()
        payload = [ctx.to_dict() for ctx in contexts] if multiple else (
            contexts[0].to_dict() if contexts else None
        )
        print(json.dumps(payload, indent=2))
        return 0
    print("[]" if multiple else "null")
    return 1

