from olympus_browser_qt import OlympusBrowserDialog


def open_olympus_single(parent):
    ctx = OlympusBrowserDialog.select_image_context(parent=parent)
    if ctx is None:
        return None
    handle = ctx.open()
    arr = handle.read_array()
    metadata = ctx.metadata
    return arr, metadata


def open_olympus_multiple(parent):
    contexts = OlympusBrowserDialog.select_image_contexts(parent=parent)
    results = []
    for ctx in contexts:
        handle = ctx.open()
        results.append((handle.read_array(), ctx.metadata))
    return results

