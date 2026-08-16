"""Assembling the FastAPI application that serves the services an application file declares."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.app_module import load_services

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter

_logger = logging.getLogger(__name__)


def load_service_routers(app_file: str) -> dict[str, 'FastAPIRouter']:
    """The routers app_file declares, keyed by service name.

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


def build_app(app_file: str, *, base_path: str = '', name: str | None = None) -> 'fastapi.FastAPI':
    """Build the application that serves app_file's services, with their models bound against base_path.

    Args:
        app_file: the application file declaring the services.
        base_path: the catalog directory the services' models bind against.
        name: serve only the service of that name, rather than every service the file declares.

    Raises NotFoundError if name is not a service the file declares, and RequestError if the file supplies
    an application object of its own, or if a service cannot bind against base_path (a table it names is
    missing, or lacks a column one of its routes needs).
    """
    return build_app_for_services(load_service_routers(app_file), app_file=app_file, base_path=base_path, name=name)


def build_app_for_services(
    services: dict[str, 'FastAPIRouter'], *, app_file: str, base_path: str = '', name: str | None = None
) -> 'fastapi.FastAPI':
    """Build the application that serves already-loaded routers, with their models bound against base_path.

    Args:
        services: the routers, keyed by service name.
        app_file: the application file they were loaded from, named in errors and log messages.
        base_path: the catalog directory the services' models bind against.
        name: serve only the service of that name, rather than every service in services.
    """
    Env.get().require_package('fastapi')
    import fastapi

    if name is not None:
        if name not in services:
            declared = ', '.join(sorted(services))
            raise excs.NotFoundError(
                excs.ErrorCode.SERVICE_NOT_FOUND,
                f'{app_file} declares no service named {name!r}; it declares: {declared}',
            )
        services = {name: services[name]}  # a single service, without disturbing the caller's mapping

    app = fastapi.FastAPI(title=name if name is not None else 'pixeltable')
    for service_name, service in services.items():
        service.bind(base_path)
        app.include_router(service)
        _logger.info(f'serving {service_name!r} from {app_file}')
    return app
