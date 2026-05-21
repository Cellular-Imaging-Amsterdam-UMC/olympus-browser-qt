from pathlib import Path

import cv2
import numpy as np

from olympus_browser_qt.models import OlympusImageContext
from olympus_browser_qt.preview import preview_png_from_metadata
from olympus_browser_qt.olympus_image_loader import OlympusPreviewPlaneProvider


def test_preview_png_from_metadata_autocontrasts_explicit_preview(tmp_path):
    source = tmp_path / "dark.png"
    gradient = np.linspace(30, 60, 64, dtype=np.uint8).reshape(8, 8)
    image = np.dstack([gradient, gradient, gradient])
    assert cv2.imwrite(str(source), image)

    out = preview_png_from_metadata({"preview_png": str(source)}, preview_height=128)
    adjusted = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)

    assert out.exists()
    assert adjusted is not None
    assert int(adjusted.min()) <= 5
    assert int(adjusted.max()) >= 250


def test_preview_provider_uses_preview_when_real_pixels_absent(tmp_path):
    source = tmp_path / "preview.png"
    red_gradient = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    image_bgr = np.dstack([np.zeros_like(red_gradient), np.zeros_like(red_gradient), red_gradient])
    assert cv2.imwrite(str(source), image_bgr)

    context = OlympusImageContext(
        name="Preview",
        container_path=Path("sample.vsi"),
        internal_path="sample.vsi/Preview",
        image_id="preview",
        kind="vsi-image",
        size_x=2,
        size_y=2,
        size_z=3,
        size_c=2,
        size_t=1,
        metadata={
            "preview_png": str(source),
            "backend_status": "xml-metadata",
            "xs": 2,
            "ys": 2,
            "zs": 3,
            "channels": 2,
        },
    )

    provider = OlympusPreviewPlaneProvider(context)
    preview_path = preview_png_from_metadata(context.metadata, preview_height=768)
    expected_rgb = cv2.cvtColor(cv2.imread(str(preview_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    plane0 = provider.get_plane(0, 0)
    plane1 = provider.get_plane(1, 1)

    assert plane0.shape == (2, 2)
    assert np.array_equal(plane0, expected_rgb[..., 0])
    assert np.array_equal(plane1, expected_rgb[..., 1])