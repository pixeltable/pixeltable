"""In-process daemon state shared across request handlers.

The dashboard feature flag lives here as a single-cell list (boxed so handlers don't
need a `global` statement). The default at daemon startup is disabled; `pxt dashboard
start` flips it to enabled via POST /api/dashboard/control.
"""

_dashboard_enabled: list[bool] = [False]


def dashboard_enabled() -> bool:
    return _dashboard_enabled[0]


def set_dashboard_enabled(enabled: bool) -> None:
    _dashboard_enabled[0] = enabled
