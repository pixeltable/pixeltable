"""Utilities for app modules (those containing table models and FastAPIRouter instances)."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from pixeltable import exceptions as excs
from pixeltable.catalog import ProhibitedWriteError, is_valid_identifier, model
from pixeltable.runtime import get_runtime
from pixeltable.utils import module_loader

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter


def load_app_module(file: str, *, subject: str, reload: bool = False) -> ModuleType:
    """Load a user-supplied module, discarding what a previous load of its directory produced if reload is set."""
    path = Path(file)
    if not path.is_file():
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{subject} not found: {file}')
    # resolve the catalog first: initializing it writes, which freeze() would refuse
    catalog = get_runtime().catalog
    try:
        with catalog.freeze():
            return module_loader.load_file(path, reload=reload)
    except ProhibitedWriteError as e:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'{file}: this {subject} modifies the catalog while it is imported; it declares tables, and '
            'creating or populating them happens when the declaration is applied',
        ) from e
    except Exception as e:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'error loading {file}: {e}') from e


def load_model_bases(schema_file: str) -> list[model.TableModelMeta]:
    """The model bases declared by a class-based schema file.

    Raises RequestError if the file is missing, fails to import, or declares no model base.
    """
    module = load_app_module(schema_file, subject='schema file', reload=True)

    # a model base carries __registered_models__ as its own class attribute, whereas the models defined
    # on it merely inherit it
    bases = [
        v
        for v in vars(module).values()
        if isinstance(v, model.TableModelMeta) and '__registered_models__' in v.__dict__
    ]
    if len(bases) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f"no model_base() found in {schema_file}; run 'pxt schema example' for a file to start from",
        )
    return bases


def load_services(app_file: str) -> dict[str, FastAPIRouter | fastapi.FastAPI]:
    """The services an application file declares, keyed by service name.

    A router names the service it declares; one that does not is named after the variable holding it. An
    application object the file supplies itself is a service too, named after its variable, whose routes
    Pixeltable did not declare and therefore cannot compare.

    Raises RequestError if the file is missing, fails to import, declares no service, declares two services
    under one name, or holds a router in a variable whose name a service cannot have.
    """
    # imported here rather than at module scope: pixeltable.serving pulls in fastapi, an optional dependency
    from pixeltable.serving import FastAPIRouter

    try:
        import fastapi

        app_type: type | None = fastapi.FastAPI
    except ImportError:
        app_type = None  # without fastapi, nothing in the file can be an application object

    module = load_app_module(app_file, subject='application file', reload=True)

    services: dict[str, FastAPIRouter | fastapi.FastAPI] = {}
    # the objects already collected, so that two variables naming one router declare a single service
    seen: set[int] = set()
    for var_name, value in vars(module).items():
        if isinstance(value, FastAPIRouter):
            name = var_name if value.name is None else value.name
        elif app_type is not None and isinstance(value, app_type):
            name = var_name
        else:
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        if not is_valid_identifier(name, allow_hyphens=True):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'{app_file}: {name!r} is not a name a service can have; name the service with FastAPIRouter(name=...)',
            )
        if name in services:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{app_file}: declares more than one service named {name!r}'
            )
        services[name] = value

    if len(services) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'no service found in {app_file}; a service is declared by creating a FastAPIRouter and adding '
            'routes to it',
        )
    return services
