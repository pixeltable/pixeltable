import io
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from pathspec import PathSpec

from pixeltable import metadata
from pixeltable.env import Env
from pixeltable.serving._config import lookup_database_runtime_config

_logger = logging.getLogger(__name__)


def _resolve_patterns(project_dir: Path, patterns: list[str]) -> set[Path]:
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

    files -= _resolve_patterns(project_dir, _read_gitignore(project_dir))
    if exclude is not None:
        files -= _resolve_patterns(project_dir, exclude)
    return sorted(files)


def _export_conda_env() -> bytes | None:
    """Export the active conda environment as a cross-platform YAML (no build strings).

    Returns None if conda is not active or the export fails.
    Strips pixeltable* pip entries — the server installs pixeltable separately.
    """
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        return None
    try:
        result = subprocess.run(
            ['conda', 'env', 'export', '--no-builds', '--prefix', conda_prefix], capture_output=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    filtered = [
        line
        for line in result.stdout.decode('utf-8').splitlines(keepends=True)
        if not re.match(r'^\s+-\s+pixeltable', line)
    ]
    return ''.join(filtered).encode('utf-8')


def __add_tarfile(tf: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tf.addfile(info, fileobj=io.BytesIO(content))


def build_db_runtime_bundle(project_dir: Path | None = None) -> Path:
    """Package the current project into a tarball for updating a cloud-hosted database runtime.

    Bundle layout:
        metadata.json   (always) — pxt_md_version, python_version
        project/        (always) — all project source files including uv.lock, pyproject.toml, etc.

    The server reads project/uv.lock and runs `uv sync --frozen` to install Python packages.
    System dependencies declared in pixeltable.toml [pixeltable.database] system_dependencies
    are included in metadata.json for the server-side Dockerfile builder to install via conda-forge.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()

    if not project_dir.is_dir():
        raise FileNotFoundError(f'Project directory does not exist: {project_dir}')

    if (project_dir / 'pyproject.toml').exists():
        if not (project_dir / 'uv.lock').exists():
            _logger.info('uv.lock not found — running uv lock')
            subprocess.run(['uv', 'lock'], cwd=project_dir, check=True)
        else:
            result = subprocess.run(['uv', 'lock', '--check'], cwd=project_dir, capture_output=True, check=False)
            if result.returncode != 0:
                Env.get().console_logger.warning(
                    'uv.lock is out of sync with pyproject.toml. '
                    'Run `uv lock` in your project directory to update it before deploying.'
                )

    runtime_cfg = lookup_database_runtime_config()
    include = runtime_cfg.include if runtime_cfg else None
    exclude = runtime_cfg.exclude if runtime_cfg else None
    system_dependencies: list[str] = (runtime_cfg.system_dependencies or []) if runtime_cfg else []

    python_version = f'{sys.version_info.major}.{sys.version_info.minor}'

    files_set = set(_collect_project_files(project_dir, include, exclude))
    # Lock files are always bundled regardless of .gitignore — they control reproducible installs.
    for lock_name in ('uv.lock', 'poetry.lock'):
        lock_path = project_dir / lock_name
        if lock_path.is_file():
            files_set.add(lock_path)
    files = sorted(files_set)

    fd, name = tempfile.mkstemp(suffix='.tar.bz2', prefix='pxt_runtime_')
    os.close(fd)
    bundle_path = Path(name)

    meta: dict = {'pxt_md_version': metadata.VERSION, 'python_version': python_version}
    if system_dependencies:
        meta['system_dependencies'] = system_dependencies
    if runtime_cfg and runtime_cfg.pixeltable_source:
        meta['pixeltable_source'] = runtime_cfg.pixeltable_source.model_dump(exclude_none=True)

    conda_env_yaml = _export_conda_env()

    with tarfile.open(bundle_path, 'w:bz2') as tf:
        __add_tarfile(tf, 'metadata.json', json.dumps(meta).encode('utf-8'))
        if conda_env_yaml is not None:
            __add_tarfile(tf, 'project/conda-environment.yaml', conda_env_yaml)
        for f in sorted(files):
            relpath = f.relative_to(project_dir)
            tf.add(f, arcname=f'project/{relpath}')

    _logger.info(f'Runtime bundle created: {bundle_path}')
    return bundle_path
