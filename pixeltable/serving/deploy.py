import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import toml
from tqdm import tqdm

from pixeltable import exceptions as excs, metadata
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.serving._config import DatabaseConfig
from pixeltable.utils.project import _LOCK_FILES, selected_files

_logger = logging.getLogger(__name__)


def _conda_exe() -> str | None:
    """Path to the executable that manages the active environment, or None if there is none to find.

    CONDA_EXE/MAMBA_EXE point at the manager that activated the environment; conda and micromamba are
    only on PATH for some installations (micromamba in particular ships no `conda`).
    """
    for var in ('CONDA_EXE', 'MAMBA_EXE'):
        exe = os.environ.get(var)
        if exe is not None and exe != '':
            return exe
    return shutil.which('conda') or shutil.which('micromamba')


def _export_conda_env() -> bytes | None:
    """Export the active conda environment as a cross-platform YAML (no build strings).

    Returns None if no conda environment is active. Raises if one is active but cannot be exported:
    building the bundle as though there were no environment would silently drop its dependencies.
    Strips the pixeltable dependency line (both conda and pip); the server installs pixeltable separately.
    """
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix is None or conda_prefix == '':
        return None
    exe = _conda_exe()
    if exe is None:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            f'A conda environment is active ({conda_prefix}) but no conda or micromamba executable was found.\n'
            'Set CONDA_EXE or MAMBA_EXE, or deactivate the environment to build without it.',
        )
    try:
        result = subprocess.run(
            [exe, 'env', 'export', '--no-builds', '--prefix', conda_prefix], capture_output=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        stderr = (
            e.stderr.decode('utf-8', errors='replace').strip() if isinstance(e, subprocess.CalledProcessError) else ''
        )
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            f'Failed to export the active conda environment ({conda_prefix}) with {exe}: {e}\n{stderr}\n'
            'Fix the environment, or deactivate it to build without it.',
        ) from e
    # a dependency line names pixeltable itself only if what follows is a version spec or nothing;
    # `pixeltable-yolox` and friends are ordinary dependencies and stay
    filtered = [
        line
        for line in result.stdout.decode('utf-8').splitlines(keepends=True)
        if not re.match(r'^\s+-\s+pixeltable\s*([=<>!~]|$)', line)
    ]
    return ''.join(filtered).encode('utf-8')


def _load_database_config(project_dir: Path, name: str | None = None) -> DatabaseConfig | None:
    """The project's configuration for the named database, or for its only one when name is absent."""
    pxt_toml = _load_database_config_from_toml(project_dir / 'pixeltable.toml', ['pixeltable', 'database'], name)
    if pxt_toml is not None:
        return pxt_toml

    py_toml = _load_database_config_from_toml(project_dir / 'pyproject.toml', ['tool', 'pixeltable', 'database'], name)
    if py_toml is not None:
        return py_toml

    # Fall back on system config.
    # TODO: This should be removed, but doing it now will break a bunch of tests
    return _select_database(Config.get().get_value('database', list), name)


def _load_database_config_from_toml(toml_path: Path, resolution: list[str], name: str | None) -> DatabaseConfig | None:
    if not toml_path.is_file():
        return None

    try:
        cfg = toml.load(toml_path)
    except Exception as e:
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid TOML in {toml_path.name}: {e}') from e

    for key in resolution:
        if not isinstance(cfg, dict) or key not in cfg:
            return None
        cfg = cfg[key]

    entries = cfg if isinstance(cfg, list) else [cfg]  # a single table is one entry, written [pixeltable.database]
    try:
        validated = [DatabaseConfig.model_validate(entry) for entry in entries]
    except Exception as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid [[pixeltable.database]] in {toml_path.name}: {e}'
        ) from e
    # a name addresses one entry, so two entries sharing one leave the target ambiguous; Config enforces the
    # same rule for the entries it reads
    seen: set[str | None] = set()
    for db in validated:
        if db.name in seen:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'Duplicate [[pixeltable.database]] name {db.name!r} in {toml_path.name}',
            )
        seen.add(db.name)
    return _select_database(validated, name)


def _select_database(databases: list[DatabaseConfig] | None, name: str | None) -> DatabaseConfig | None:
    """The entry name addresses; the only entry when name addresses none, so a lone entry configures any target."""
    if databases is None or len(databases) == 0:
        return None
    if name is not None:
        named = next((db for db in databases if db.name == name), None)
        if named is not None:
            return named
    return databases[0] if len(databases) == 1 else None


def _abbrev_path(path: str, max_len: int = 40) -> str:
    """Truncate path from the left, so that a long path doesn't wrap the progress bar line."""
    return path if len(path) <= max_len else '…' + path[-(max_len - 1) :]


def __add_tarfile(tf: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tf.addfile(info, fileobj=io.BytesIO(content))


def build_db_runtime_bundle(
    project_dir: Path | None = None, show_progress: bool = False, db_name: str | None = None
) -> Path:
    """Package the current project into a tarball for updating a hosted database runtime.

    If show_progress is True, a progress bar tracking the number of project files added, and naming the
    file most recently added, is displayed.

    Bundle layout:
        metadata.json   (always) — pxt_md_version, python_version
        project/        (always) — all project source files including uv.lock, pyproject.toml, etc.

    The server reads project/uv.lock and runs `uv sync --frozen` to install Python packages.
    System dependencies declared in the project's [[pixeltable.database]] entry as system_dependencies
    are included in metadata.json for the server-side Dockerfile builder to install via conda-forge.

    Lockfiles are never generated here — the developer is expected to have run `uv lock` (or provided
    a requirements.txt) in their project. If no lockfile and no conda environment is found, a warning
    is emitted but the bundle is still built.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()

    if not project_dir.is_dir():
        raise FileNotFoundError(f'Project directory does not exist: {project_dir}')

    runtime_cfg = _load_database_config(project_dir, db_name)
    system_dependencies: list[str] = (runtime_cfg.system_dependencies or []) if runtime_cfg else []

    # Config override wins; otherwise use the deploy environment's version.
    python_version = (runtime_cfg.python_version if runtime_cfg else None) or (
        f'{sys.version_info.major}.{sys.version_info.minor}'
    )

    files = selected_files(project_dir, runtime_cfg)
    has_lockfile = any(file.parent == project_dir and file.name in _LOCK_FILES for file in files)

    print(f'A runtime bundle will be built containing {len(files)} files from {project_dir}.')
    print(
        'By default, all files not ignored by .gitignore are included; '
        'you can adjust this behavior with include/exclude in pixeltable.toml.'
    )

    conda_env_yaml = _export_conda_env()

    # No lockfile and no conda export means the image has no source for Python dependencies.
    if not has_lockfile and conda_env_yaml is None:
        Env.get().console_logger.warning(
            'No dependency lockfile (uv.lock, poetry.lock, requirements.txt) was found and no conda '
            'environment was detected.\nThe deployed runtime may not have the necessary Python '
            'dependencies to run correctly.'
        )

    fd, name = tempfile.mkstemp(suffix='.tar.bz2', prefix='pxt_runtime_')
    os.close(fd)
    bundle_path = Path(name)

    meta: dict = {'pxt_md_version': metadata.VERSION, 'python_version': python_version}
    if system_dependencies:
        meta['system_dependencies'] = system_dependencies

    with (
        tarfile.open(bundle_path, 'w:bz2') as tf,
        tqdm(desc='Building runtime bundle', total=len(files), unit=' files', disable=not show_progress) as bar,
    ):
        __add_tarfile(tf, 'metadata.json', json.dumps(meta).encode('utf-8'))
        if conda_env_yaml is not None:
            __add_tarfile(tf, 'project/conda-environment.yaml', conda_env_yaml)
        for f in files:
            relpath = f.relative_to(project_dir)
            # refresh=False: the postfix is drawn by the following update(), which respects tqdm's redraw interval
            bar.set_postfix_str(_abbrev_path(str(relpath)), refresh=False)
            tf.add(f, arcname=f'project/{relpath}')
            bar.update(1)
        bar.set_postfix_str('', refresh=False)

    _logger.info(f'Runtime bundle created: {bundle_path}')
    return bundle_path
