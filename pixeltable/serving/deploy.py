import logging
import os
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
        raise excs.Error(f'Invalid deployment configuration:\n{exc}') from exc
    return {config.name: config for config in configs}


def package(project_dir: Path | None = None) -> Path:
    """Bundle the contents of a Pixeltable project directory into a tarball.

    Args:
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

    with tarfile.open(bundle_path, 'w:bz2') as tf:
        tf.add(project_dir, arcname='.')

    _logger.info(f'Packaging complete: {bundle_path}')
    return bundle_path
