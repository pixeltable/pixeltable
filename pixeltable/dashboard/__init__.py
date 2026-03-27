"""
Pixeltable Dashboard - Local Web UI for exploring Pixeltable data.

Usage:
    import pixeltable as pxt
    pxt.dashboard.serve()
"""

import logging

from pixeltable.config import Config

_logger = logging.getLogger('pixeltable.dashboard')


def serve(open_browser: bool = True) -> None:
    """
    Start the Pixeltable dashboard UI server.

    Args:
        open_browser: Whether to automatically open the browser (default: True)

    Example:
        >>> import pixeltable as pxt
        >>> pxt.dashboard.serve()  # Opens browser to http://localhost:{port}
    """
    from pixeltable.dashboard.server import run_server

    port = Config.get('dashboard_port')

    url = f'http://localhost:{port}'
    _logger.info(f'Starting Pixeltable Dashboard at {url}')

    print(f'\n  Pixeltable Dashboard running at {url}\n')

    if open_browser:
        import webbrowser

        webbrowser.open(url)

    run_server(port=port)
