from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.app_module import load_services

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter

_logger = logging.getLogger(__name__)


def load_service_routers(app_file: str) -> dict[str, 'FastAPIRouter']:
    """The FastAPIRouter instances in app_file, keyed by service name.

    An application object the file supplies itself is a service that Pixeltable declared no routes for, so
    there is nothing to bind and nothing to serve; that is a RequestError here, whereas load_services()
    returns it, for the sake of reporting it in a diff.
    """
    from pixeltable.serving import FastAPIRouter

    routers: dict[str, FastAPIRouter] = {}
    for service_name, service in load_services(app_file).items():
        if not isinstance(service, FastAPIRouter):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'{app_file}: {service_name!r} is an application object of its own, which Pixeltable cannot '
                'serve; declare the service with FastAPIRouter instead.',
            )
        routers[service_name] = service
    return routers


def create_app(app_file: str, *, base_path: str = '', service_name: str | None = None) -> 'fastapi.FastAPI':
    """Build the application that serves app_file's services, with their models bound against base_path.

    Args:
        app_file: the application file declaring the services.
        base_path: the catalog directory the services' models bind against.
        service_name: serve only the FastAPIRouter instance with that name, rather than every FastAPIRouter instance.

    Raises NotFoundError if app_file does not contain a FastAPIRouter instance with that name, and RequestError if the
    file supplies an application object of its own, or if a FastAPIRouter instance cannot bind against base_path.
    """
    return create_app_for_services(
        load_service_routers(app_file), app_file=app_file, base_path=base_path, service_name=service_name
    )


def create_app_for_services(
    services: dict[str, 'FastAPIRouter'], *, app_file: str, base_path: str = '', service_name: str | None = None
) -> 'fastapi.FastAPI':
    """Build the application that serves already-loaded routers, with their models bound against base_path.

    Args:
        services: the routers, keyed by service name.
        app_file: the application file they were loaded from, named in errors and log messages.
        base_path: the catalog directory the services' models bind against.
        service_name: serve only the service of that name, rather than every service in services.
    """
    Env.get().require_package('fastapi')
    import fastapi

    if service_name is not None:
        if service_name not in services:
            declared = ', '.join(sorted(services))
            raise excs.NotFoundError(
                excs.ErrorCode.SERVICE_NOT_FOUND,
                f'{app_file} declares no FastAPIRouter named {service_name!r}; it declares: {declared}',
            )
        services = {service_name: services[service_name]}  # a single service, without disturbing the caller's mapping

    app = fastapi.FastAPI(title=service_name if service_name is not None else 'pixeltable')
    for name, service in services.items():
        service.bind(base_path)
        app.include_router(service)
        _logger.info(f'serving {name!r} from {app_file}')
    return app


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
