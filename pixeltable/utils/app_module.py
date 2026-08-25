"""Utilities for app modules (those containing table models and FastAPIRouter instances)."""

from __future__ import annotations

import importlib
import keyword
import linecache
import traceback
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from pixeltable import exceptions as excs
from pixeltable.catalog import ProhibitedWriteError, is_valid_identifier, model
from pixeltable.env import Env
from pixeltable.runtime import get_runtime

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter


def load_app_module(file: str, *, subject: str) -> ModuleType:
    """Import file under the module path it has below this process's project root, and return the module.

    The project root is the one Env resolved at startup, so that every module this process imports comes
    from a single project. A file below a different root reaches this process by a restart.
    """
    path = Path(file).resolve()
    if not path.is_file():
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{subject} not found: {file}')

    env = Env.get()
    root = env.project_root
    if root is None or env.find_project_root(path.parent) != root:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, _no_root_msg(path, subject, root))
    module_name = _module_name(path, root, subject)

    # resolve the catalog first: initializing it writes, which freeze() would refuse
    catalog = get_runtime().catalog
    try:
        with catalog.freeze():
            return importlib.import_module(module_name)
    except ProhibitedWriteError as e:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, _prohibited_write_msg(str(path), subject, e)
        ) from e
    except Exception as e:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'error loading {file}: {e}') from e


def _no_root_msg(path: Path, subject: str, root: Path | None) -> str:
    """Report which project root this process serves, and what to do about a file outside it."""
    file_root = Env.get().find_project_root(path.parent)
    rule = (
        f'A UDF is recorded as a module path relative to the project root, so this {subject} has to sit '
        f'under the root that this process serves, and the directories from that root down to the file '
        f'become part of the path.'
    )
    if file_root is None:
        serving = 'no project' if root is None else f'the project at {root}'
        return (
            f'{path}: this process serves {serving}, and the file belongs to none. Searched {path.parent} '
            f'and every directory above it for a pixeltable.toml, or a pyproject.toml with a '
            f'[tool.pixeltable] section.\n'
            f'{rule}\n'
            f"Run 'pxt init' in the directory that holds the file, then 'pxt daemon restart' to serve it."
        )
    if root is None:
        return (
            f'{path}: this process serves no project, and the file belongs to the one at {file_root}.\n'
            f'{rule}\n'
            f"That project was marked after this process started: run 'pxt daemon restart' to serve it."
        )
    return (
        f'{path}: this process serves the project at {root}, and the file belongs to the one at '
        f'{file_root}.\n'
        f'{rule}\n'
        f"Run this command from within {root}, or run 'pxt daemon restart' from {file_root} to serve that "
        f'project instead.'
    )


def _module_name(path: Path, root: Path, subject: str) -> str:
    """The dotted name an import of path reaches it by, with root on sys.path."""
    relative = path.relative_to(root).with_suffix('')
    for part in relative.parts:
        if not part.isidentifier() or keyword.iskeyword(part):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'{path}: {part!r} is not a module name, so this {subject} cannot be imported; rename it, or '
                f'the directory holding it, to a Python identifier',
            )
    return '.'.join(relative.parts)


def _prohibited_write_msg(file: str, subject: str, exc: ProhibitedWriteError) -> str:
    """Report which statement in file modified the catalog, and what to write instead."""
    location = ''
    for frame, lineno in traceback.walk_tb(exc.__traceback__):
        if frame.f_code.co_filename == file:
            statement = (linecache.getline(file, lineno) or '').strip()
            location = f'line {lineno}: {statement}\n' if statement != '' else f'line {lineno}\n'
    return (
        f'{file}: this {subject} modifies the catalog while it is imported.\n'
        f'{location}'
        'Declare a table with a model class, and insert rows from a route or a script; '
        "'pxt schema update' then creates and populates them."
    )


def load_model_bases(schema_file: str) -> list[model.TableModelMeta]:
    """The model bases declared by a class-based schema file.

    Raises RequestError if the file is missing, fails to import, or declares no model base.
    """
    module = load_app_module(schema_file, subject='schema file')

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

    module = load_app_module(app_file, subject='application file')

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
