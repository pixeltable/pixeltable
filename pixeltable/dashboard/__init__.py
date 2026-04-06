"""
Pixeltable Dashboard - Local Web UI for exploring Pixeltable data.

Usage:
    import pixeltable as pxt
    pxt.dashboard.serve()
"""

import time


def serve(open_browser: bool = True) -> None:
    """
    Ensure the Pixeltable dashboard is running and optionally open it in a browser.

    The dashboard is started automatically when Pixeltable initializes. This function
    ensures it is running (starting it if necessary) and optionally opens a browser tab.

    Args:
        open_browser: Whether to automatically open the browser (default: True)

    Example:
        >>> import pixeltable as pxt
        >>> pxt.dashboard.serve()  # Opens browser to http://localhost:{port}
    """
    from pixeltable.env import Env

    harness = Env.get().dashboard_harness
    harness.start()
    if open_browser:
        import webbrowser  # Intentionally scope-limited since this is the only place `webbrowser` is used

        time.sleep(0.5)
        webbrowser.open(f'http://localhost:{harness.port}')
