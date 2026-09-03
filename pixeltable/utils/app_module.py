"""Utilities for app modules (those containing table models and FastAPIRouter instances)."""

from __future__ import annotations

import importlib
import keyword
import linecache
import re
import sys
import threading
import traceback
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from pixeltable import exceptions as excs
from pixeltable.catalog import ProhibitedWriteError, is_valid_identifier, model
from pixeltable.catalog.model import diff
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.func import FunctionRegistry
from pixeltable.runtime import get_runtime
from pixeltable.utils.project import in_environment
from pixeltable_cli.types import CheckReport, RouteSpec, ServiceSpec

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

    # resolve the catalog first: initializing it writes, which freeze() would refuse
    catalog = get_runtime().catalog
    try:
        registry = FunctionRegistry.get()
        with _lock, catalog.freeze():
            _evict_project_modules()
            # a file written after this process started is invisible to a finder that cached its directory
            importlib.invalidate_caches()
            registered = set(registry.module_fns)
            try:
                return importlib.import_module(name)
            except BaseException:
                # a module that raises partway through leaves the udfs it already defined registered, and
                # Python drops it from sys.modules, so _evict_project_modules() cannot reach them again
                registry.deregister_functions(set(registry.module_fns) - registered)
                raise
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


def _evict_project_modules() -> None:
    """Remove the project's own modules from sys.modules, and their udfs from the registry.

    Removing project modules allows us to have them re-imported, in response to changes to the source files.

    The standard library and installed packages stay loaded, including when the environment sits inside the
    project root (eg, a .venv).
    """
    root = Config.get().project_root
    assert root is not None  # module_name() refuses a file outside a project root
    registry = FunctionRegistry.get()
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, '__file__', None)
        if module_file is None or name == '__main__':
            continue
        if name.split('.', maxsplit=1)[0] in _RUNNING_PACKAGES:
            continue
        # the file path lets us distinguish between a project module and a standard library module
        resolved = Path(module_file).resolve()
        if in_environment(resolved):
            continue
        if resolved.is_relative_to(root):
            registry.deregister_module(name)
            sys.modules.pop(name, None)


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
        'Define a table with a model class, and insert rows from a route or a script; '
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


def visible_models(module: ModuleType) -> dict[str, model.TableModelMeta]:
    """Every model the module reaches by name, keyed by the table name each defines."""
    models: dict[str, model.TableModelMeta] = {}
    for value in vars(module).values():
        if not isinstance(value, model.TableModelMeta):
            continue
        # a model base carries __registered_models__ as its own class attribute
        if '__registered_models__' in value.__dict__:
            models.update({m.__table_spec__['name']: m for m in value.defined_models()})
        else:
            models[value.__table_spec__['name']] = value
    return models


def model_mismatch_error_str(models: dict[str, model.TableModelMeta], base_path: str) -> str | None:
    """Return an error string explaining a mismatch between the models and their corresponding tables in base_path, or
    None if there is no mismatch."""
    diffs = diff.validate_models(models, base_path)
    mismatched = {name: d for name, d in diffs.items() if d.resolution != 'up_to_date'}
    if len(mismatched) == 0:
        return None

    detail = '\n'.join(line for name, d in mismatched.items() for line in diff.format_diff(name, d))
    target = '' if base_path == '' else f' {base_path}'
    unsupported = sorted(name for name, d in mismatched.items() if d.resolution == 'unsupported')
    if len(unsupported) == 0:
        hint = f'Run `pxt schema update <app file>{target}` first.'
    else:
        hint = (
            f'No schema update can reconcile {", ".join(repr(name) for name in unsupported)}: adjust the '
            'existing table(s) manually, or adjust the models to be consistent with the catalog.'
        )
        if len(unsupported) < len(mismatched):
            hint += f'\nRun `pxt schema update <app file>{target}` for the rest.'
    names = ', '.join(repr(name) for name in sorted(mismatched))
    return f'Cannot serve {names}:\n{detail}\n{hint}'


def check_report(file: str, bases: list[model.TableModelMeta]) -> CheckReport:
    """What checking a file on its own reports: whether it is valid, and what to fix or to know about."""
    errors = check_udf_references(bases)
    return CheckReport(file=file, valid=len(errors) == 0, errors=errors, warnings=shadowed_project_modules())


def check_udf_references(bases: list[model.TableModelMeta]) -> list[str]:
    """Returns error strings for udf references in 'bases' that cannot be resolved."""
    project_root = Config.get().project_root
    assert project_root is not None
    errors: list[str] = []
    fn_paths = {fn.self_path for base in bases for cls in base.defined_models() for fn in cls.referenced_functions()}
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


def get_module_services(module: ModuleType, file: str) -> tuple[fastapi.FastAPI | None, dict[str, FastAPIRouter]]:
    """
    Returns from the module: (FastAPI instance, FastAPIRouter instances, keyed by service name).
    """
    # imported here rather than at module scope: pixeltable.serving pulls in fastapi, an optional dependency
    from pixeltable.serving import FastAPIRouter

    try:
        import fastapi

        app_type: type | None = fastapi.FastAPI
    except ImportError:
        app_type = None  # without fastapi, nothing in the file can be an application object

    routers: dict[str, FastAPIRouter] = {}
    apps: dict[str, fastapi.FastAPI] = {}
    # the objects already collected, so that two variables naming one router define a single service
    seen: set[int] = set()
    seen_names: set[str] = set()
    for var_name, value in vars(module).items():
        name: str
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
                f'{file}: {name!r} is not a valid name for a service; name the service with FastAPIRouter(name=...)',
            )
        if name in seen_names:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{file}: defines more than one service named {name!r}'
            )
        seen_names.add(name)

        if isinstance(value, FastAPIRouter):
            routers[name] = value
        else:
            apps[name] = value

    if len(routers) == 0 and len(apps) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'{file}: needs to have at least one FastAPIRouter or FastAPI application object',
        )
    if len(apps) > 1:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'{file} contains more than one FastAPI application object'
        )
    app = apps.popitem()[1] if len(apps) > 0 else None

    if app is not None:
        # make sure every router is included in the application
        app_endpoints = {id(route.endpoint) for route in app.routes if hasattr(route, 'endpoint')}
        for name, router in routers.items():
            router_endpoints = {id(route.endpoint) for route in router.routes if hasattr(route, 'endpoint')}
            if not router_endpoints.issubset(app_endpoints):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'{file}: the FastAPI application does not include the router {name!r}; '
                    f'include it with include_router(), or move it to a separate file',
                )

    return app, routers


def services_by_name(module: ModuleType, file: str) -> dict[str, FastAPIRouter | fastapi.FastAPI]:
    """The module's services, keyed by service name."""
    app, routers = get_module_services(module, file)
    if app is None:
        return dict(routers)
    return {module.__name__.rpartition('.')[2]: app}


def service_spec(name: str, service: FastAPIRouter | fastapi.FastAPI, routers: list[FastAPIRouter]) -> ServiceSpec:
    """The spec of the named service."""
    from pixeltable.serving import FastAPIRouter

    if isinstance(service, FastAPIRouter):
        return service.service_spec(name)
    routes: list[RouteSpec] = []
    for router in routers:
        prefix = _include_prefix(service, router)
        routes += [
            route.model_copy(update={'path': f'{prefix}{route.path}'}) for route in router.service_spec(name).routes
        ]
    return ServiceSpec(name=name, routes=routes, app_paths=_app_paths(service, routers))


def _include_prefix(app: fastapi.FastAPI, router: FastAPIRouter) -> str:
    """The prefix that app adds to router's paths, empty when absent."""
    app_paths = {id(getattr(route, 'endpoint', None)): getattr(route, 'path', '') for route in app.routes}
    for route in router.routes:
        path = getattr(route, 'path', None)
        app_path = app_paths.get(id(getattr(route, 'endpoint', None)))
        if path is not None and app_path is not None and app_path.endswith(path):
            return app_path[: len(app_path) - len(path)]
    return ''


def _app_paths(app: fastapi.FastAPI, routers: list[FastAPIRouter]) -> list[str]:
    """The paths which app serves itself, minus the ones from routers."""
    from_routers = {
        id(endpoint) for router in routers for endpoint in (getattr(r, 'endpoint', None) for r in router.routes)
    }
    included: set[str] = set()
    for route in app.routes:
        path = getattr(route, 'path', None)
        if path is not None and id(getattr(route, 'endpoint', None)) in from_routers:
            # a route spells a path parameter with its converter, '/media/{path:path}', where the document
            # names it '/media/{path}'
            included.add(re.sub(r'{([^}:]+):[^}]+}', r'{\1}', path))
    return sorted(set(app.openapi()['paths']) - included)


def module_routers(module: ModuleType) -> list[FastAPIRouter]:
    from pixeltable.serving import FastAPIRouter

    routers: list[FastAPIRouter] = []
    for value in vars(module).values():
        if isinstance(value, FastAPIRouter) and not any(value is router for router in routers):
            routers.append(value)
    return routers
