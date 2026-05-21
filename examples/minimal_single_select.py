from PyQt6.QtWidgets import QApplication

from olympus_browser_qt import OlympusBrowserDialog


app = QApplication([])
ctx = OlympusBrowserDialog.select_image_context()
if ctx is not None:
    print(ctx.to_dict())

