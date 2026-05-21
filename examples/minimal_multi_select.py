from PyQt6.QtWidgets import QApplication

from olympus_browser_qt import OlympusBrowserDialog


app = QApplication([])
contexts = OlympusBrowserDialog.select_image_contexts()
for ctx in contexts:
    print(ctx.to_dict())
