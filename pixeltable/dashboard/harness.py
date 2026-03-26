import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Literal


_logger = logging.getLogger('pixeltable')

class DashboardHarness:
    port: int
    dashboard_thread: threading.Thread | None
    watchdog_thread: threading.Thread | None

    def __init__(self, port: int) -> None:
        self.port = port
        self.dashboard_thread = None
        self.watchdog_thread = None

    def start(self) -> None:
        probe = self._probe_port()
        match probe:
            case 'pixeltable':
                print(f'Found an existing Pixeltable dashboard at: http://localhost:{self.port}')
                self._start_watchdog()
            case 'other':
                print(f'Warning: Dashboard port {self.port} is in use by another application.')
                print(
                    'To use a different port, set the PIXELTABLE_DASHBOARD_PORT environment '
                    'variable and restart Python.'
                )
            case 'none':
                self._start_dashboard()

    def _probe_port(self) -> Literal['pixeltable', 'other', 'none']:
        """Check if a Pixeltable dashboard is already listening on `self.port`."""
        try:
            resp = urllib.request.urlopen(f'http://localhost:{self.port}/api/pixeltable-health', timeout=1)
            data = json.loads(resp.read().decode())
            if data.get('status') == 'ok':
                return 'pixeltable'
            return 'other'
        except urllib.error.HTTPError:
            return 'other'
        except urllib.error.URLError as e:
            if isinstance(e.reason, ConnectionRefusedError):
                return 'none'
            return 'other'
        except Exception:
            return 'other'

    def _start_dashboard(self) -> None:
        """Start the dashboard server in a daemon thread."""
        from pixeltable.dashboard.server import run_server

        assert self.dashboard_thread is None or not self.dashboard_thread.is_alive(), (
            'Dashboard server is unexpectedly running'
        )

        def run() -> None:
            try:
                run_server(port=self.port)
            except OSError as e:
                # EADDRINUSE (Address already in use) can happen despite the earlier probe, if we lost a race
                # condition with another process starting on the same port in parallel.
                if 'Address already in use' in str(e):
                    self.dashboard_thread = None
                    self._start_watchdog()
                else:
                    _logger.error(f'Dashboard server error: {e}')
            except Exception as e:
                _logger.error(f'Dashboard server error: {e}')

        self.dashboard_thread = threading.Thread(target=run, daemon=True, name='pixeltable-dashboard')
        self.dashboard_thread.start()

        time.sleep(1.0)
        if self.dashboard_thread.is_alive():
            print(f'Pixeltable dashboard available at: http://localhost:{self.port}')
        else:
            print(f'Pixeltable dashboard failed to start on port {self.port}; check logs for details.')

    def _start_watchdog(self) -> None:
        """Start a background thread to monitor the primary dashboard server and take over if it dies."""
        if self.watchdog_thread is not None and self.watchdog_thread.is_alive():
            return

        def watchdog_loop() -> None:
            while True:
                time.sleep(3.0)

                # If we somehow started our own server in the meantime, stop watching
                if self.dashboard_thread is not None and self.dashboard_thread.is_alive():
                    break

                probe = self._probe_port()
                match probe:
                    case 'pixeltable':
                        continue  # primary server is alive, keep watching
                    case 'other':
                        # This can happen in rare cases where we lost a race condition with a different
                        # (non-Pixeltable) process attempting to start on the same port. This will
                        # probably never happen, but we handle it gracefully just in case.
                        _logger.error(f'Dashboard port {self.port} is in use by another application.')
                        break
                    case 'none':
                        # Primary server is dead; take over by starting our own server
                        self.watchdog_thread = None
                        self._start_dashboard()
                        break

        self.watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True, name='pixeltable-dashboard-watchdog')
        self.watchdog_thread.start()
