"""Reusable PyQt6 browser for Olympus VSI image containers."""

from .api import (
    OlympusBrowserDialog,
    OlympusGateway,
    OlympusImageContext,
    OlympusImageHandle,
)

__all__ = [
    "OlympusBrowserDialog",
    "OlympusGateway",
    "OlympusImageContext",
    "OlympusImageHandle",
    "OlympusViewerWindow",
]


def __getattr__(name: str):
    if name == "OlympusViewerWindow":
        from .olympus_viewer import OlympusViewerWindow

        return OlympusViewerWindow
    raise AttributeError(name)
