"""Loading the schema and application files an application is written in, and the declarations they hold."""

from __future__ import annotations

import importlib.util
import sys
import threading
import uuid
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from pixeltable import exceptions as excs
from pixeltable.catalog import is_valid_identifier, model

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter

# A loaded user file's directory is placed on the process-global sys.path so it can import sibling modules.
# _sys_path_lock guards the (un)registration; _sys_path_added refcounts each directory we add, so concurrent loads
# of the same directory share one entry that is removed only once the last of them finishes.
_sys_path_lock = threading.Lock()
_sys_path_added: dict[str, int] = {}


def _add_to_sys_path(entry: str) -> bool:
    """Register a directory on sys.path; return True if the caller must later release it.

    Refcounts directories we add so concurrent loads share one entry; a directory already present externally is
    left untouched (and not released).
    """
    with _sys_path_lock:
        if entry in _sys_path_added:
            _sys_path_added[entry] += 1
            return True
        if entry in sys.path:
            return False
        sys.path.append(entry)
        _sys_path_added[entry] = 1
        return True


def _remove_from_sys_path(entry: str) -> None:
    """Release a directory registered via _add_to_sys_path(); remove it once the last holder releases it."""
    with _sys_path_lock:
        n = _sys_path_added.get(entry)
        if n is None:
            return
        if n > 1:
            _sys_path_added[entry] = n - 1
        else:
            del _sys_path_added[entry]
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


def load_app_module(file: str, *, subject: str) -> ModuleType:
    """Load a user-supplied module under a unique module name."""
    path = Path(file)
    if not path.is_file():
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{subject} not found: {file}')

    # load under a unique key so a user file can't shadow an existing module (eg, one named json.py); the
    # key is unique per load, so this needs no synchronization
    module_name = f'pxt_user_file_{uuid.uuid4().hex}'
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'cannot load {subject}: {file}')
    module = importlib.util.module_from_spec(spec)

    # put the file's own directory on sys.path so it can import sibling modules next to it; only the
    # sys.path (un)registration is synchronized, so concurrent loads still run exec_module() in parallel
    sys_path_entry = str(path.parent)
    needs_remove = _add_to_sys_path(sys_path_entry)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'error loading {file}: {e}') from e
    finally:
        sys.modules.pop(module_name, None)
        if needs_remove:
            _remove_from_sys_path(sys_path_entry)
    return module


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
