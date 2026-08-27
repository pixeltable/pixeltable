"""Utilities for app modules (those containing table models and FastAPIRouter instances)."""

from __future__ import annotations

import importlib
import keyword
import linecache
import sys
import threading
import traceback
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from pixeltable import exceptions as excs
from pixeltable.catalog import ProhibitedWriteError, is_valid_identifier, model
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.func import FunctionRegistry
from pixeltable.runtime import get_runtime

_lock = threading.RLock()

# the packages this process is running, which stay loaded even when the project holds a checkout of them:
# re-importing them would leave two of every class
_RUNNING_PACKAGES = ('pixeltable', 'pixeltable_cli')

# TODO: Catalog needs to discard cached TableVersion that references reloaded modules

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter


def module_name(file: str, *, subject: str) -> str:
    """The dotted import path of 'file', relative to the project root."""
    path = Path(file).resolve()
    if not path.is_file():
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{subject} not found: {file}')

    # Env.get() establishes the project root
    Env.get()
    root = Config.get().project_root
    if root is None or not path.is_relative_to(root):
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, _no_root_msg(path, subject, root))
    relative = path.relative_to(root).with_suffix('')
    for part in relative.parts:
        if not part.isidentifier() or keyword.iskeyword(part):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'{path}: {part!r} is not a module name, so this {subject} cannot be imported; rename it, or '
                f'the directory holding it, to a Python identifier',
            )
    return '.'.join(relative.parts)


def load_app_module(file: str, *, subject: str) -> ModuleType:
    """Import file under the module path relative to the project root."""
    path = Path(file).resolve()
    name = module_name(file, subject=subject)
    root = Config.get().project_root
    assert root is not None  # module_name() refuses a file outside a project root

    # resolve the catalog first: initializing it writes, which freeze() would refuse
    catalog = get_runtime().catalog
    try:
        with _lock, catalog.freeze():
            _evict_project_modules(root)
            # a file written after this process started is invisible to a finder that cached its directory
            importlib.invalidate_caches()
            return importlib.import_module(name)
    except ProhibitedWriteError as e:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, _prohibited_write_msg(str(path), subject, e)
        ) from e
    except Exception as e:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'error loading {file}: {e}') from e


def _no_root_msg(path: Path, subject: str, root: Path | None) -> str:
    """Report the project root, and what to do about a file outside it."""
    rule = (
        f'A UDF is recorded as a module path relative to the project root, which is the directory holding '
        f'the project configuration, so this {subject} has to sit under that root.'
    )
    if root is None:
        return (
            f'{path}: there is no project root. Searched {Path.cwd()} and every directory above it for '
            f'a project configuration: a pixeltable.toml, or a pyproject.toml with a [tool.pixeltable] '
            f'section.\n'
            f'{rule}\n'
            f"Run 'pxt init' in the directory that holds your project, then run this command again."
        )
    return (
        f'{path}: the project root is {root}, which the file does not sit under.\n'
        f'{rule}\n'
        "Run this command from the file's own project."
    )


def _evict_project_modules(root: Path) -> None:
    """Discard every loaded module read from root, and the udfs they registered, so this load reads them again.

    Scanning sys.modules rather than tracking what an earlier load imported: resolving a stored udf reference
    imports a project module too, and a module missing from the eviction set is never read again.
    """
    registry = FunctionRegistry.get()
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, '__file__', None)
        if module_file is None or name == '__main__':
            continue
        if name.split('.', maxsplit=1)[0] in _RUNNING_PACKAGES:
            continue
        if Path(module_file).resolve().is_relative_to(root):
            registry.deregister_module(name)
            sys.modules.pop(name, None)


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


def get_model_bases(module: ModuleType) -> list[model.TableModelMeta]:
    """Returns the model bases found in module."""
    # a model base carries __registered_models__ as its own class attribute, whereas the models defined
    # on it merely inherit it
    bases = [
        v
        for v in vars(module).values()
        if isinstance(v, model.TableModelMeta) and '__registered_models__' in v.__dict__
    ]
    return bases


def check_udf_references(bases: list[model.TableModelMeta]) -> list[str]:
    """Returns error strings for udf references in 'bases' that cannot be resolved."""
    project_root = Config.get().project_root
    assert project_root is not None
    errors: list[str] = []
    fn_paths = {fn.self_path for base in bases for cls in base.declared_models() for fn in cls.referenced_functions()}
    for fn_path in sorted(p for p in fn_paths if p is not None):
        resolved = _resolved_module(fn_path)
        if resolved is None:
            errors.append(f'{fn_path}: cannot be resolved to a known module')
            continue
        file = getattr(resolved, '__file__', None)
        if file is None or not Path(file).resolve().is_relative_to(project_root):
            continue  # an installed module
        top_level = fn_path.split('.', maxsplit=1)[0]
        if _first_on_path(top_level) is None:
            errors.append(f'{fn_path}: {top_level!r} is not on sys.path')
    return errors


def shadowed_project_modules() -> list[str]:
    """Report the project's top-level modules that an import of the same name reads from somewhere else.

    Env appends the project root to sys.path, so an installed distribution of the same name wins. The
    generic names a single-file recipe hands out collide most often.
    """
    root = Config.get().project_root
    assert root is not None  # only a process serving a project has modules to report
    warnings: list[str] = []
    for entry in sorted(root.iterdir()):
        name = entry.stem if entry.suffix == '.py' else entry.name
        if not name.isidentifier() or keyword.iskeyword(name):
            continue
        if entry.is_file() and entry.suffix != '.py':
            continue
        if entry.is_dir() and not any(entry.glob('*.py')):
            continue
        origin = _first_on_path(name)
        if origin is not None and not origin.is_relative_to(root):
            warnings.append(
                f'{entry.name}: an import of {name!r} reads {origin}, so this project cannot record a udf '
                f'under {name!r}; rename it'
            )
    return warnings


def _resolved_module(dotted_path: str) -> ModuleType | None:
    """Return the most specific (longest prefix) module reference in dotted_path that's in sys.modules."""
    parts = dotted_path.split('.')
    for i in range(len(parts) - 1, 0, -1):
        module = sys.modules.get('.'.join(parts[:i]))
        if module is not None:
            return module
    return None


def _first_on_path(module_name: str) -> Path | None:
    """Return the file read by 'import <module_name>'."""
    spec = importlib.machinery.PathFinder.find_spec(module_name, sys.path)
    if spec is None:
        return None
    if spec.origin is not None and spec.origin != 'namespace':
        return Path(spec.origin).resolve()
    locations = list(spec.submodule_search_locations or [])
    return Path(locations[0]).resolve() if len(locations) > 0 else None


def load_services(app_file: str) -> dict[str, FastAPIRouter | fastapi.FastAPI]:
    """The FastAPIRouter/FastAPI instances in an app file, keyed by service name.

    - FastAPIRouter instances are either explicitly named (name parameter) or implicitly via variable assignment
    - a FastAPI instance is named via variable assignment
    """
    return get_module_services(load_app_module(app_file, subject='application file'), app_file)


def get_module_services(module: ModuleType, file: str) -> dict[str, FastAPIRouter | fastapi.FastAPI]:
    """
    Returns the FastAPIRouter/FastAPI instances found in module, keyed by service name.
    """
    # imported here rather than at module scope: pixeltable.serving pulls in fastapi, an optional dependency
    from pixeltable.serving import FastAPIRouter

    try:
        import fastapi

        app_type: type | None = fastapi.FastAPI
    except ImportError:
        app_type = None  # without fastapi, nothing in the file can be an application object

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
                f'{file}: {name!r} is not a name a service can have; name the service with FastAPIRouter(name=...)',
            )
        if name in services:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{file}: declares more than one service named {name!r}'
            )
        services[name] = value

    if len(services) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'no service found in {file}; a service is declared by creating a FastAPIRouter and adding routes to it',
        )
    return services
