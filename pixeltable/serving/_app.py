from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.app_module import load_app_module, module_routers, service_spec, services_by_name, visible_models

if TYPE_CHECKING:
    import fastapi

    from pixeltable_cli.types import ServiceSpec

_logger = logging.getLogger(__name__)


def create_app(app_file: str, name: str, base_path: str = '') -> tuple['fastapi.FastAPI', 'ServiceSpec']:
    """Create or return the FastAPI app derived from app_file, plus its spec."""
    Env.get().require_package('fastapi')
    import fastapi

    from pixeltable.serving import FastAPIRouter

    module = load_app_module(app_file, subject='application file')
    services = services_by_name(module, app_file)
    if name not in services:
        declared = ', '.join(sorted(services))
        raise excs.NotFoundError(
            excs.ErrorCode.SERVICE_NOT_FOUND, f'{app_file} declares no service named {name!r}; it declares: {declared}'
        )

    routers = module_routers(module)
    for router in routers:
        router.bind(base_path)
    # bind every model visible in the file, not just the ones created in it directly
    for model in visible_models(module).values():
        model._bind(base_path)

    service = services[name]
    spec = service_spec(name, service, routers)
    _logger.info(f'serving {name!r} from {app_file}')
    if not isinstance(service, FastAPIRouter):
        return service, spec

    app = fastapi.FastAPI(title=name)
    app.include_router(service)
    return app, spec


_OTEL_NOT_INSTALLED = (
    "OpenTelemetry tracing requires the instrumentation package; install: `pip install 'pixeltable[otel]'`"
)


def init_instrumentation(**kwargs: Any) -> None:
    """Start emitting telemetry, with kwargs overriding the [otel] config and its OTEL_* environment variables.

    Call before the first Pixeltable operation: init() configures the providers the instrumentation emits to.
    """
    Env.get().require_package('opentelemetry.instrumentation.pixeltable', not_installed_msg=_OTEL_NOT_INSTALLED)
    import opentelemetry.instrumentation.pixeltable as pxt_otel

    pxt_otel.init(**kwargs)


def instrument_app(app: 'fastapi.FastAPI') -> None:
    """Instrument a FastAPI application, so that Pixeltable spans nest under its request spans."""
    import opentelemetry.instrumentation.pixeltable as pxt_otel

    pxt_otel.instrument_fastapi(app)
