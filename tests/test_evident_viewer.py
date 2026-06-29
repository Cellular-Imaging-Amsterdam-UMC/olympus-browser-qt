import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from olympus_browser_qt import OlympusImageContext, OlympusViewerWindow

_APP = None


def app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication([])
    return _APP


def test_viewer_instantiates_without_real_olympus_data():
    app()
    ctx = OlympusImageContext(
        name="Preview",
        container_path=Path("sample.vsi"),
        internal_path="sample.vsi/Preview",
        image_id="preview",
        kind="vsi-image",
        size_x=128,
        size_y=96,
        size_z=3,
        size_c=2,
        size_t=1,
        pixel_size_x_um=0.25,
        channel_names=["Green", "Magenta"],
        metadata={"xs": 128, "ys": 96, "zs": 3, "channels": 2, "filetype": ".vsi"},
    )

    win = OlympusViewerWindow(ctx)
    try:
        app().processEvents()
        assert win.windowTitle() == "Olympus Viewer"
        assert win._provider is not None
        assert len(win._channel_buttons) == 2
    finally:
        win.close()


def test_viewer_shows_s_slider_for_unpinned_multi_s_context():
    app()
    ctx = OlympusImageContext(
        name="Preview",
        container_path=Path("sample.vsi"),
        internal_path="sample.vsi/Preview",
        image_id="preview",
        kind="vsi-image",
        size_x=128,
        size_y=96,
        size_z=3,
        size_c=2,
        size_t=1,
        size_s=5,
        pixel_size_x_um=0.25,
        channel_names=["Green", "Magenta"],
        metadata={"xs": 128, "ys": 96, "zs": 3, "channels": 2, "size_s": 5, "filetype": ".vsi"},
    )

    win = OlympusViewerWindow(ctx)
    try:
        app().processEvents()
        assert not win._s_controls.isHidden()
        assert win._s_slider.maximum() == 4
        assert win._s_slider.value() == 2
    finally:
        win.close()


def test_viewer_hides_s_slider_for_browser_pinned_s_context():
    app()
    ctx = OlympusImageContext(
        name="Preview",
        container_path=Path("sample.vsi"),
        internal_path="sample.vsi/Preview",
        image_id="preview",
        kind="vsi-image",
        size_x=128,
        size_y=96,
        size_z=3,
        size_c=2,
        size_t=1,
        size_s=5,
        selected_s=3,
        pixel_size_x_um=0.25,
        channel_names=["Green", "Magenta"],
        metadata={"xs": 128, "ys": 96, "zs": 3, "channels": 2, "size_s": 5, "filetype": ".vsi"},
    )

    win = OlympusViewerWindow(ctx)
    try:
        app().processEvents()
        assert win._s_controls.isHidden()
        assert win._current_s_value() == 3
    finally:
        win.close()
