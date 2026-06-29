"""Helpers for choosing a readable microscopy scale bar.

Adapted from omero-browser-qt 0.2.5, Copyright (c) 2026 Ron Hoebe,
MIT License.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class ScaleBarSpec:
    """Description of a scale bar to draw on screen."""

    image_pixels: float
    screen_pixels: float
    label: str
    physical_um: float


def compute_scale_bar(
    um_per_pixel: float | None,
    screen_pixels_per_image_pixel: float,
    *,
    target_screen_px: float = 120.0,
    min_screen_px: float = 70.0,
    max_screen_px: float = 180.0,
) -> ScaleBarSpec | None:
    """Choose a human-friendly scale bar size for the current zoom."""

    if (
        um_per_pixel is None
        or um_per_pixel <= 0
        or screen_pixels_per_image_pixel <= 0
    ):
        return None

    raw_um = target_screen_px * um_per_pixel / screen_pixels_per_image_pixel
    if raw_um <= 0:
        return None

    magnitude = 10 ** math.floor(math.log10(raw_um))
    for factor in (1, 2, 5, 10):
        physical_um = factor * magnitude
        screen_px = physical_um / um_per_pixel * screen_pixels_per_image_pixel
        if min_screen_px <= screen_px <= max_screen_px:
            break
    else:
        physical_um = raw_um
        screen_px = target_screen_px

    return ScaleBarSpec(
        image_pixels=physical_um / um_per_pixel,
        screen_pixels=screen_px,
        label=_format_physical_length(physical_um),
        physical_um=physical_um,
    )


def _format_physical_length(physical_um: float) -> str:
    if physical_um >= 1000:
        mm = physical_um / 1000.0
        if mm >= 10:
            return f"{mm:.0f} mm"
        if mm >= 1:
            return f"{mm:.1f} mm"
        return f"{mm:.2f} mm"
    if physical_um >= 10:
        return f"{physical_um:.0f} um"
    if physical_um >= 1:
        return f"{physical_um:.1f} um"
    return f"{physical_um:.2f} um"

