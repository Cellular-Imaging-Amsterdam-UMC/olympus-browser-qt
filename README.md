# olympus-browser-qt

Reusable PyQt6 dialog and viewer for browsing Olympus / Evident CellSens `.vsi`
datasets.

This package follows the same public shape as the Leica, Zeiss, and Nikon Qt
browsers, but uses a native Python VSI/ETS reader. It does not use Bio-Formats
or Java.

## Install

```bash
pip install -e .
```

For this checkout, `run_viewer.cmd` uses the local `deconvolve` conda
environment when it is available.

## Current Status

- Scans folders for `.vsi` files and exposes Olympus image contexts.
- Reads TIFF preview/SIS metadata from the `.vsi` container with `tifffile`.
- Discovers companion ETS files in `_<vsi stem>_/stack*/frame_*.ets`.
- Natively parses the Olympus `SIS`/`ETS` chunk table for raw uncompressed ETS
  files.
- Reads real `uint8`, `uint16`, integer, and float raw ETS planes and regions.
- Generates real previews from ETS planes, with TIFF preview fallback.

Compressed ETS variants such as JPEG, JPEG2000, PNG, BMP, and lossless JPEG are
recognized by compression code but are not decoded yet.

## Single Select

```python
from PyQt6.QtWidgets import QApplication
from olympus_browser_qt import OlympusBrowserDialog

app = QApplication([])
ctx = OlympusBrowserDialog.select_image_context(roots=[r"D:\data"])
if ctx is not None:
    print(ctx.name, ctx.container_path, ctx.size_x, ctx.size_y)
```

## Direct Pixel Reads

```python
from olympus_browser_qt import OlympusGateway

node = OlympusGateway().container_node(r"D:\data\sample.vsi")
ctx = node.children[0].context

plane = ctx.open().read_plane(z=0, c=0)
stack = ctx.open().read_stack(c=0)
arr = ctx.open().read_array()
```

The public handle returns NumPy arrays. For the sample raw ETS dataset, planes
are real `uint16` arrays read from the external `.ets` file.

## CLI

```bash
olympus_browser D:\data
olympus_browser D:\data\sample.vsi --multi
olympus_viewer
run_viewer.cmd
```
