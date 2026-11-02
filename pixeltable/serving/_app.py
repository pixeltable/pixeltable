from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.app_module import load_app_module, module_routers, service_spec, services_by_name, visible_models

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter
    from pixeltable_cli.types import ServiceSpec

_logger = logging.getLogger(__name__)


def create_app(
    app_file: str, name: str, base_path: str = ''
) -> tuple['fastapi.FastAPI', 'ServiceSpec', list['FastAPIRouter']]:
    """Create or return the FastAPI app derived from app_file, plus its spec and the routers in app_file."""
    Env.get().require_package('fastapi')
    import fastapi

    from pixeltable.serving import FastAPIRouter

    module = load_app_module(app_file, subject='application file')
    services = services_by_name(module, app_file)
    if name not in services:
        defined = ', '.join(sorted(services))
        raise excs.NotFoundError(
            excs.ErrorCode.SERVICE_NOT_FOUND, f'{app_file} defines no service named {name!r}; it defines: {defined}'
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
        return service, spec, routers

    app = fastapi.FastAPI(title=name)
    app.include_router(service)
    return app, spec, routers


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


def instrument_app(app: 'fastapi.FastAPI') -> Any:
    """Instrument a FastAPI application, so that Pixeltable spans nest under its request spans.

    Returns the ASGI app to serve, which also starts the trace of a request at the time the Pixeltable Cloud
    gateway received it.
    """
    import opentelemetry.instrumentation.pixeltable as pxt_otel
    from opentelemetry.instrumentation.pixeltable._sdk import _gateway_span_app

    # the fastapi instrumentation ignores these variables when it receives patterns from the caller, and this keeps
    # their patterns next to /health$
    env_excluded_urls = os.environ.get(
        'OTEL_PYTHON_FASTAPI_EXCLUDED_URLS', os.environ.get('OTEL_PYTHON_EXCLUDED_URLS', '')
    )
    # a blank pattern would match every URL, as in '/ping,' or '   '
    env_patterns = [pattern.strip() for pattern in env_excluded_urls.split(',') if pattern.strip() != '']
    # k8s probes call /health every few seconds, and a trace per probe is noise
    excluded_urls = ','.join([*env_patterns, '/health$'])
    # a span per ASGI message sent or received would double the spans of every request
    pxt_otel.instrument_fastapi(app, excluded_urls=excluded_urls, exclude_spans=['receive', 'send'])
    return _gateway_span_app(app, excluded_urls)
