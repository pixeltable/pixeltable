from typing import Any

from pixeltable_cli.utils import identity

# taken before the daemon imports pixeltable, so that it cannot postdate the daemon's imports
STARTUP_IDENTITY: dict[str, Any] = identity()
