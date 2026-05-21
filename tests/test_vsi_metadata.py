from pathlib import Path

import numpy as np

from olympus_browser_qt.olympus_gateway import OlympusGateway
from olympus_browser_qt.preview import preview_png_from_metadata
from olympus_browser_qt.vsi_metadata import extract_vsi_xml_metadata
from olympus_browser_qt.vsi_native import parse_ets, read_ets_region, read_vsi_mosaic_plane


SAMPLE_VSI = Path("localdata/20231102_TRITC, Cy5, DAPI_007_current_current.vsi")
SAMPLE_ETS = Path(
    "localdata/_20231102_TRITC, Cy5, DAPI_007_current_current_/stack1/frame_t_0.ets"
)
COMPRESSED_VSI = Path("localdata/24-63 p2w3_Cing_20x.vsi")


def test_extract_vsi_metadata_from_tiff_and_ets_sample():
    metadata = extract_vsi_xml_metadata(SAMPLE_VSI)

    assert metadata["xs"] == 2048
    assert metadata["ys"] == 2048
    assert metadata["zs"] == 35
    assert metadata["channels"] == 3
    assert metadata["channel_names"] == ["TRITC", "Cy5", "DAPI"]
    assert metadata["pixel_backend"] == "native"
    assert metadata["backend_status"] == "native-ets-raw"
    assert metadata["ets_file_count"] == 1


def test_parse_ets_sample_header_and_read_region():
    info = parse_ets(str(SAMPLE_ETS))

    assert info.size_x == 2048
    assert info.size_y == 2048
    assert info.size_z == 35
    assert info.size_c == 3
    assert info.tile_width == 2048
    assert info.tile_height == 2048
    assert info.is_raw

    region = read_ets_region(info, x=0, y=0, width=32, height=16, z=0, c=0)
    assert region.shape == (16, 32)
    assert region.dtype == np.dtype("<u2")


def test_mosaic_preview_reader_honors_z_index():
    z0 = read_vsi_mosaic_plane(SAMPLE_VSI, z=0, c=0, max_size=128)
    z_mid = read_vsi_mosaic_plane(SAMPLE_VSI, z=17, c=0, max_size=128)

    assert z0.shape == z_mid.shape
    assert not np.array_equal(z0, z_mid)


def test_gateway_hides_vsi_companion_folder_when_vsi_exists(tmp_path):
    (tmp_path / "sample.vsi").write_bytes(b"not a real vsi")
    (tmp_path / "_sample_").mkdir()
    (tmp_path / "other").mkdir()

    root = OlympusGateway().scan_path(tmp_path)[0]
    names = [child.name for child in root.children]

    assert "sample.vsi" in names
    assert "other" in names
    assert "_sample_" not in names


def test_compressed_vsi_exposes_and_reads_all_ets_stacks():
    node = OlympusGateway().container_node(COMPRESSED_VSI)

    assert [child.metadata["ets_stack_name"] for child in node.children] == [
        "stack1",
        "stack10000",
        "stack10002",
    ]
    assert [child.metadata["ets_compression"] for child in node.children] == [2, 2, 3]

    for child in node.children:
        ctx = child.context
        assert ctx is not None
        plane = ctx.open().read_plane(z=min((ctx.size_z or 1) // 2, (ctx.size_z or 1) - 1), c=0)
        assert plane.ndim == 2
        assert plane.size > 0


def test_compressed_stack_previews_use_stack_specific_cache_keys():
    node = OlympusGateway().container_node(COMPRESSED_VSI)

    previews = [
        preview_png_from_metadata(child.context.metadata, preview_height=128)
        for child in node.children
        if child.context is not None
    ]

    assert len(previews) == 3
    assert len({path.name for path in previews}) == 3
