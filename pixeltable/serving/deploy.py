import io
import logging
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from pixeltable import config
from pixeltable.env import Env
from pixeltable.serving._config import lookup_deployment_config

_logger = logging.getLogger('pixeltable')


def deploy(deployment_name: str) -> None:
    cfg = lookup_deployment_config(deployment_name)
    Env.get().console_logger.info(f'Deploying {deployment_name!r} ...')
    conda_export = _export_conda_env()
    lockfile = _find_lockfile()
    if conda_export is None and len(cfg.env_dependencies) == 0:
        Env.get().console_logger.warning(
            'No conda environment was detected and no environment dependencies are specified in config.\n'
            'The deployment may not have the necessary dependencies to run correctly.'
        )
    if lockfile is None and len(cfg.python_dependencies) == 0:
        Env.get().console_logger.warning(
            'No dependency lockfile was found and no Python dependencies are specified in config.\n'
            'The deployment may not have the necessary dependencies to run correctly.'
        )
    bundle_path = package(cfg, conda_export=conda_export)
    Env.get().console_logger.info(f'Built project bundle: {bundle_path}')


def _resolve_patterns(project_dir: Path, patterns: list[str]) -> set[Path]:
    """Resolve include/exclude patterns against a project directory, returning matching file paths.

    Pattern semantics (like pypi wheel specifications):
    - 'dist' matches any path component named 'dist' (and contents of matching directories)
    - '/dist' matches only 'dist' at the project root
    - Glob characters (*, ?, [...]) are supported
    """
    matched: set[Path] = set()
    for pattern in patterns:
        if pattern.startswith('/'):
            glob_pattern = pattern[1:]
        else:
            glob_pattern = f'**/{pattern}'
        for path in project_dir.glob(glob_pattern):
            if path.is_file():
                matched.add(path)
            elif path.is_dir():
                matched.update(p for p in path.rglob('*') if p.is_file())
    return matched


def _collect_project_files(project_dir: Path, include: list[str] | None, exclude: list[str] | None) -> list[Path]:
    if include is not None:
        files = _resolve_patterns(project_dir, include)
    else:
        files = {p for p in project_dir.rglob('*') if p.is_file()}
    if exclude is not None:
        files -= _resolve_patterns(project_dir, exclude)
    return sorted(files)


def _export_conda_env() -> bytes | None:
    """Export the active conda environment as YAML, without platform-specific build strings.

    Returns the environment.yml content as bytes, or None if not running in a conda environment.
    """
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        return None

    Env.get().console_logger.info(f'Found a conda environment: {os.environ["CONDA_DEFAULT_ENV"]}')
    try:
        result = subprocess.run(('conda', 'env', 'export', '--no-builds'), capture_output=True, check=True)
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
    deploy_config: config.DeploymentConfig, project_dir: Path | None = None, conda_export: bytes | None = None
) -> Path:
    """Bundle the contents of a Pixeltable project directory into a tarball.

    Args:
        deploy_config: Deployment configuration.
        project_dir: Path to the project directory. Defaults to the current working directory.
        conda_export: Output of ``conda env export --no-builds``, included as ``environment.yml``
            in the bundle when provided.

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

    files = _collect_project_files(project_dir, deploy_config.include, deploy_config.exclude)
    with tarfile.open(bundle_path, 'w:bz2') as tf:
        if conda_export is not None:
            info = tarfile.TarInfo(name='environment.yml')
            info.size = len(conda_export)
            tf.addfile(info, fileobj=io.BytesIO(conda_export))
        for f in files:
            relpath = f.relative_to(project_dir)
            tf.add(f, arcname=f'project/{relpath}')

    _logger.info(f'Packaging complete: {bundle_path}')
    return bundle_path
