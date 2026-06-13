"""OpenTelemetry integration for Pixeltable.

This subpackage requires the `pixeltable[otel]` extra; importing it without those dependencies raises
ImportError. When the extra is installed, instrumentation is enabled automatically at Env initialization
(opt out with `PIXELTABLE_OTEL=0` or `otel.enabled = false` in the config). Use [init][pixeltable.otel.init]
or [PixeltableInstrumentor][pixeltable.otel.PixeltableInstrumentor] for manual control.
"""

from ._bridge import PixeltableInstrumentor
from ._sdk import init

__all__ = ['PixeltableInstrumentor', 'init']
