import io
import json
import logging
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import toml
from pathspec import PathSpec

from pixeltable import config, metadata
from pixeltable.env import Env

_logger = logging.getLogger(__name__)


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


def _export_conda_env() -> bytes | None:
    """Export the active conda environment as YAML, without platform-specific build strings.

    Returns the environment.yml content as bytes, or None if not running in a conda environment.
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

    # Strip pixeltable* entries from the pip section — dev-installed versions aren't on PyPI.
    # The runtime Dockerfile installs pixeltable[serve] explicitly after env creation.
    env_text = result.stdout.decode('utf-8')
    filtered = [line for line in env_text.splitlines(keepends=True) if not re.match(r'^\s+-\s+pixeltable', line)]
    return ''.join(filtered).encode('utf-8')


def _find_lockfile(project_dir: Path | None = None) -> Path | None:
    if project_dir is None:
        project_dir = Path.cwd()
    for name in ('uv.lock', 'poetry.lock', 'requirements.txt'):
        path = project_dir / name
        if path.is_file():
            Env.get().console_logger.info(f'Found a dependency lockfile: {path}')
            return path
    return None


def __add_tarfile(tf: tarfile.TarFile, name: str, content: bytes) -> None:
    """Helper function to add a file with the given content to the tarfile being built in `package()`."""
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tf.addfile(info, fileobj=io.BytesIO(content))


def _load_database_runtime_config(project_dir: Path) -> config.DatabaseRuntimeConfig | None:
    """Read [database] config from project_dir/pixeltable.toml; fall back to Config singleton.

    When project_dir is passed explicitly (e.g. in tests), the project_dir's own
    pixeltable.toml takes precedence over the CWD-based Config singleton.
    """
    toml_path = project_dir / 'pixeltable.toml'
    if toml_path.is_file():
        raw_toml = toml.load(toml_path)
        db_raw = raw_toml.get('pixeltable', {}).get('database')
        if db_raw is not None:
            return config.DatabaseRuntimeConfig.model_validate(db_raw)
    from pixeltable.serving._config import lookup_database_runtime_config
    return lookup_database_runtime_config()


def build_db_runtime_bundle(project_dir: Path | None = None) -> Path:
    """Package the current project into a tarball for updating a cloud-hosted database runtime.

    Bundles project source files, dependency lockfiles, and optionally a conda environment
    or a pixeltable source override. Service configuration is not included — services are
    managed independently via the CLI.

    Bundle layout:
        environment.yml      (optional — present when running inside a conda environment)
        runtime_config.json  (optional — present when [database.pixeltable_source] is configured)
        project/             (all project source files: UDFs, lockfiles, pyproject.toml, etc.)
    """
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()

    if not project_dir.is_dir():
        raise FileNotFoundError(f'Project directory does not exist: {project_dir}')

    runtime_cfg = _load_database_runtime_config(project_dir)
    include = runtime_cfg.include if runtime_cfg else None
    exclude = runtime_cfg.exclude if runtime_cfg else None
    pxt_source = runtime_cfg.pixeltable_source if runtime_cfg else None

    lockfile = _find_lockfile(project_dir)
    if lockfile is None:
        Env.get().console_logger.warning(
            'No dependency lockfile was found (uv.lock, poetry.lock, requirements.txt).\n'
            'The runtime may not have the necessary dependencies.'
        )

    conda_export = _export_conda_env()
    files = _collect_project_files(project_dir, include, exclude)

    fd, name = tempfile.mkstemp(suffix='.tar.bz2', prefix='pxt_runtime_')
    os.close(fd)
    bundle_path = Path(name)

    with tarfile.open(bundle_path, 'w:bz2') as tf:
        manifest = {'pxt_md_version': metadata.VERSION}
        __add_tarfile(tf, 'metadata.json', json.dumps(manifest).encode('utf-8'))
        if conda_export is not None:
            __add_tarfile(tf, 'environment.yml', conda_export)
        if pxt_source is not None:
            rt_config = {'pixeltable_source': pxt_source.model_dump(exclude_none=True)}
            __add_tarfile(tf, 'runtime_config.json', json.dumps(rt_config).encode('utf-8'))
        for f in sorted(files):
            relpath = f.relative_to(project_dir)
            tf.add(f, arcname=f'project/{relpath}')

    _logger.info(f'Runtime bundle created: {bundle_path}')
    return bundle_path
