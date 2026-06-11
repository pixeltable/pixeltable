import io
import json
import logging
import os
import subprocess
import tarfile
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

import toml
from pathspec import PathSpec

import pixeltable as pxt
from pixeltable import config, exceptions as excs, metadata
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.serving._config import lookup_deployment_config, lookup_service_config

_logger = logging.getLogger(__name__)


def build_deploy_bundle(deployment_name: str) -> Path:
    """
    Packages the current Pixeltable project into a tarball, along with:
    - conda environment, if present
    - catalog metadata for any relevant tables
    - deployment and service configuration
    """
    Env.get().require_package('fastapi')

    cfg = lookup_deployment_config(deployment_name)
    Env.get().console_logger.info(f'Deploying {deployment_name!r} ...')

    # Process the service, validating it and collecting its config and table MD.
    service_cfg, md_export = _process_service(cfg)

    Env.get().console_logger.info(f'The following service will be deployed: {cfg.service}')

    config_export = {'deployment': [cfg.model_dump(mode='json')]}
    if service_cfg is not None:
        config_export['service'] = [service_cfg.model_dump(mode='json')]
    conda_export = _export_conda_env()
    lockfile = _find_lockfile()

    if conda_export is None and len(cfg.env_dependencies) == 0:
        Env.get().console_logger.warning(
            'No conda environment was detected and no dependencies are specified in config.\n'
            'The deployment may not have the necessary dependencies to run correctly.'
        )
    if lockfile is None and len(cfg.python_dependencies) == 0:
        Env.get().console_logger.warning(
            'No dependency lockfile was found and no Python dependencies are specified in config.\n'
            'The deployment may not have the necessary dependencies to run correctly.'
        )

    bundle_path = package(cfg, config_export=config_export, md_export=md_export, conda_export=conda_export)
    Env.get().console_logger.info(f'Built project bundle: {bundle_path}')
    return bundle_path


def _process_service(cfg: config.DeploymentConfig) -> tuple[config.ServiceConfig | None, dict[str, Any]]:
    """
    Validate the service referenced by the deployment config.

    Returns: (service config, table md export)
    """
    service_cfg: config.ServiceConfig | None = None
    table_paths: set[str] = set()
    service_name = cfg.service
    if ':' in service_name:
        # "module:attribute" references a service defined as a FastAPI app in user code.
        paths = _tables_from_fastapi_app(cfg, service_name)
        table_paths.update(paths)
    else:
        # Otherwise, it references a service defined in config.
        service_cfg = lookup_service_config(service_name)
        for route in service_cfg.routes:
            if route.type != 'compute':
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'Service {service_name!r} referenced in deployment {cfg.name!r} has a route {route.path!r} '
                    f"of type {route.type!r}. Currently, only 'compute' routes are supported for deployment.",
                )
            assert isinstance(route, config.InsertRouteConfig)
            table_paths.add(route.table)
        _logger.info(f'Validated service {service_name!r} with {len(service_cfg.routes)} route(s).')

    md_export = _export_tables_md(table_paths)
    return service_cfg, md_export


def _tables_from_fastapi_app(env_cfg: config.DeploymentConfig, module_attr: str) -> set[str]:
    """
    Given a "module:attribute" reference to a FastAPI app, import the module and inspect the app's routes to find
    all tables mentioned by any compute routes.

    Validate the config to ensure there are no non-compute routes and that all routes refer to valid FastAPI apps.
    """
    from fastapi import FastAPI
    from starlette.routing import Route

    from pixeltable.serving._fastapi import PxtEndpoint

    module_path, attr_name = module_attr.split(':', 1)
    try:
        module = import_module(module_path)
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'Could not import module `{module_path}` referenced in deployment {env_cfg.name!r}: {e}',
        ) from e
    if not hasattr(module, attr_name):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'Module `{module_path}` referenced in deployment {env_cfg.name!r} has no attribute `{attr_name}`',
        )
    app = getattr(module, attr_name)
    if not isinstance(app, FastAPI):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'Service `{module_attr}` referenced in deployment {env_cfg.name!r} is not a FastAPI app',
        )
    table_paths: set[str] = set()
    for route in app.routes:
        if isinstance(route, Route) and isinstance(route.endpoint, PxtEndpoint):
            if route.endpoint.route_type != 'compute':
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'Service `{module_attr}` referenced in deployment {env_cfg.name!r} has a route {route.path!r} '
                    f"of type {route.endpoint.route_type!r}. Currently, only 'compute' routes are supported for "
                    'deployment.',
                )
            assert route.endpoint.tbl is not None  # It's always non-None for 'compute' routes
            table_paths.add(route.endpoint.tbl._path())

    _logger.info(
        f'Validated service {module_attr!r} with {len(table_paths)} table(s) referenced by {len(app.routes)} route(s).'
    )
    return table_paths


def _resolve_patterns(project_dir: Path, patterns: list[str]) -> set[Path]:
    """Resolve gitignore-style patterns against a project directory, returning matching file paths."""
    spec = PathSpec.from_lines('gitignore', patterns)
    return {p for p in project_dir.rglob('*') if p.is_file() and spec.match_file(p.relative_to(project_dir))}


def _read_gitignore(project_dir: Path) -> list[str]:
    gitignore = project_dir / '.gitignore'
    if not gitignore.is_file():
        return []
    return gitignore.read_text().splitlines()


def _collect_project_files(project_dir: Path, include: list[str] | None, exclude: list[str] | None) -> list[Path]:
    if include is not None:
        files = _resolve_patterns(project_dir, include)
    else:
        files = {p for p in project_dir.rglob('*') if p.is_file()}

    # Automatically exclude everything in the project's .gitignore
    files -= _resolve_patterns(project_dir, _read_gitignore(project_dir))
    if exclude is not None:
        files -= _resolve_patterns(project_dir, exclude)
    return sorted(files)


def _export_tables_md(table_paths: set[str]) -> dict[str, Any]:
    # Get all tables mentioned by any route contained in this deployment.
    tables = [pxt.get_table(path) for path in sorted(table_paths)]

    # Get the md for all ancestors of all such tables.
    catalog = get_runtime().catalog
    with catalog.begin_xact(for_write=False):
        tables_md = [catalog.load_md_for_export(tbl, as_replica=False) for tbl in tables]

    # The ancestor md is returned as: primary table first, followed by ancestors in descending order.
    # Reverse so that ancestors come first, then flatten and de-duplicate (since some tables might have common
    # ancestors). Use a dict for deduplicating, so that we preserve ancestor order to get a topologically
    # sorted list at the end.
    flattened_md = {md.tbl_md.tbl_id: md for md_list in tables_md for md in reversed(md_list)}
    bundle_md = {
        'pxt_version': pxt.__version__,
        'pxt_md_version': metadata.VERSION,
        'tables_md': [md.as_dict() for md in flattened_md.values()],
    }
    return bundle_md


def _export_conda_env() -> bytes | None:
    """Export the active conda environment as YAML, without platform-specific build strings.

    Returns the conda-env.yml content as bytes, or None if not running in a conda environment.
    """
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe is None or 'CONDA_DEFAULT_ENV' not in os.environ:
        return None

    Env.get().console_logger.info(f'Found a conda environment: {os.environ["CONDA_DEFAULT_ENV"]}')
    try:
        result = subprocess.run((conda_exe, 'env', 'export', '--no-builds'), capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        Env.get().console_logger.warning(f'Failed to export conda environment: {exc}')
        return None
    return result.stdout


def _find_lockfile() -> Path | None:
    cwd = Path.cwd()
    for name in ('uv.lock', 'poetry.lock', 'requirements.txt'):
        path = cwd / name
        if path.is_file():
            Env.get().console_logger.info(f'Found a dependency lockfile: {path}')
            return path
    return None


def package(
    deployment_config: config.DeploymentConfig,
    config_export: dict[str, Any],
    md_export: dict[str, Any],
    conda_export: bytes | None = None,
    project_dir: Path | None = None,
) -> Path:
    """Bundle the contents of a Pixeltable project directory into a tarball.

    Args:
        deployment_config: Deployment configuration.
        config_export: The deployment and service configuration to include in the bundle, as a dict.
        md_export: The table metadata to include in the bundle, as a dict.
        conda_export: Output of `conda env export --no-builds`, included as `conda-env.yml`
            in the bundle when provided.
        project_dir: Path to the project directory. Defaults to the current working directory.

    Returns:
        Path to the generated tarball.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()

    if not project_dir.is_dir():
        raise FileNotFoundError(f'Project directory does not exist: {project_dir}')

    _logger.info(f'Packaging project directory: {project_dir}')
    fd, name = tempfile.mkstemp(suffix='.tar.bz2', prefix='pxt_deploy_')
    os.close(fd)
    bundle_path = Path(name)

    files = _collect_project_files(project_dir, deployment_config.include, deployment_config.exclude)
    with tarfile.open(bundle_path, 'w:bz2') as tf:
        __add_tarfile(tf, 'config.toml', toml.dumps(config_export).encode('utf-8'))
        __add_tarfile(tf, 'metadata.json', json.dumps(md_export).encode('utf-8'))
        if conda_export is not None:
            __add_tarfile(tf, 'conda-env.yml', conda_export)
        for f in files:
            relpath = f.relative_to(project_dir)
            tf.add(f, arcname=f'project/{relpath}')

    _logger.info(f'Packaging complete: {bundle_path}')
    return bundle_path


def __add_tarfile(tf: tarfile.TarFile, name: str, content: bytes) -> None:
    """Helper function to add a file with the given content to the tarfile being built in `package()`."""
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tf.addfile(info, fileobj=io.BytesIO(content))
