import io
import logging
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pydantic

from pixeltable import exceptions as excs
from pixeltable.config import Config

_logger = logging.getLogger('pixeltable')


class DeploymentConfig(pydantic.BaseModel):
    name: str
    include: list[str] | None = None
    exclude: list[str] | None = None


def get_deployment_configs() -> dict[str, DeploymentConfig]:
    value = Config.get().get_value('deployment', list)
    if value is None:
        return {}
    try:
        configs = [DeploymentConfig(**entry) for entry in value]
    except pydantic.ValidationError as exc:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid deployment configuration:\n{exc}'
        ) from exc
    return {config.name: config for config in configs}


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
    try:
        result = subprocess.run(['conda', 'env', 'export', '--no-builds'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        _logger.warning(f'Failed to export conda environment: {exc}')
        return None
    return result.stdout


def package(deploy_config: DeploymentConfig, project_dir: Path | None = None) -> Path:
    """Bundle the contents of a Pixeltable project directory into a tarball.

    Args:
        deployment: Name of a deployment configuration.
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

    conda_env = _export_conda_env()
    files = _collect_project_files(project_dir, deploy_config.include, deploy_config.exclude)
    with tarfile.open(bundle_path, 'w:bz2') as tf:
        if conda_env is not None:
            info = tarfile.TarInfo(name='environment.yml')
            info.size = len(conda_env)
            tf.addfile(info, fileobj=io.BytesIO(conda_env))
        for f in files:
            tf.add(f, arcname=str(f.relative_to(project_dir)))

    _logger.info(f'Packaging complete: {bundle_path}')
    return bundle_path
