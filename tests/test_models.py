from pathlib import Path

from olympus_browser_qt import OlympusImageContext
from olympus_browser_qt.metadata import (
    context_from_metadata,
    format_metadata_summary,
    normalize_resolution_metadata,
)


def test_context_serializes_path_and_metadata():
    ctx = OlympusImageContext(
        name="Image 1",
        container_path=Path("sample.vsi"),
        internal_path="sample.vsi/Folder/Image 1",
        image_id="abc",
        kind="vsi-image",
        size_x=10,
        size_y=20,
        size_s=4,
        selected_s=2,
        channel_names=["DAPI"],
        metadata={"nested": {"path": Path("sample.vsi")}},
    )

    data = ctx.to_dict()

    assert data["container_path"] == "sample.vsi"
    assert data["size_s"] == 4
    assert data["selected_s"] == 2
    assert data["metadata"]["nested"]["path"] == "sample.vsi"


def test_metadata_summary_prefers_convertleica_fields():
    summary = format_metadata_summary(
        {
            "save_child_name": "Scene 1",
            "uuid": "abc",
            "xs": 512,
            "ys": 256,
            "zs": 4,
            "ts": 2,
            "channels": 3,
            "tiles": 5,
            "xres2": 0.25,
            "yres2": 0.25,
            "channelResolution": [16, 16, 16],
            "experiment_datetime": "2026-05-18T10:20:30.123Z",
        }
    )

    assert "Name: Scene 1" in summary
    assert "UUID:" not in summary
    assert summary.splitlines()[1] == "Date: 2026-05-18 10:20:30"
    assert "Dimensions: 512 x 256  Z=4  T=2  C=3  S=5" in summary
    assert "Pixel size: X=0.25 micrometer, Y=0.25 micrometer" in summary
    assert "FOV size: 8192 um^2" in summary
    assert "Image size: 6.3 MB" in summary
    assert "Pixel type: 16-bit" in summary


def test_resolution_metadata_fallback_converts_native_spatial_units():
    metadata = normalize_resolution_metadata(
        {
            "xres": 2.5e-7,
            "yres": 5e-7,
            "zres": 1e-6,
            "xres2": 2.5e-7,
            "yres2": 0,
            "resunit": "meter",
        }
    )

    assert metadata["xres2"] == 0.25
    assert metadata["yres2"] == 0.5
    assert metadata["zres2"] == 1.0
    assert metadata["resunit2"] == "micrometer"


def test_context_uses_normalized_resolution_metadata():
    ctx = context_from_metadata(
        name="Image 1",
        container_path=Path("sample.vsi"),
        internal_path="sample.vsi/Image 1",
        image_id="abc",
        kind="vsi-image",
        metadata={
            "xres": 2.5e-7,
            "yres": 2.5e-7,
            "xres2": 2.5e-7,
            "yres2": 2.5e-7,
            "resunit": "m",
        },
    )

    assert ctx.pixel_size_x_um == 0.25
    assert ctx.pixel_size_y_um == 0.25
    assert format_metadata_summary(ctx.metadata).splitlines()[2] == (
        "Pixel size: X=0.25 micrometer, Y=0.25 micrometer"
    )


def test_context_populates_size_s_from_metadata_dimensions():
    ctx = context_from_metadata(
        name="Image 1",
        container_path=Path("sample.vsi"),
        internal_path="sample.vsi/Image 1",
        image_id="abc",
        kind="vsi-image",
        metadata={
            "dimensions": {"x": 32, "y": 16, "s": 7},
        },
    )

    assert ctx.size_s == 7
