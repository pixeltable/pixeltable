import io
import json
import logging
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

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


def _is_stable_pypi_version(version: str) -> bool:
    """Return True if version is a stable PyPI-installable release (no local label or dev segment)."""
    return '+' not in version and '.dev' not in version


def _export_conda_env() -> tuple[bytes, str | None] | None:
    """Export the active conda environment as YAML, without platform-specific build strings.

    Returns (cleaned_yaml, detected_pixeltable_version) or None if not in a conda environment.
    detected_pixeltable_version is the version string found in the pip section (e.g. "0.6.7" or
    "0.0.1.dev1600+76a4f45"). It is always stripped from the YAML — local dev versions cannot be
    pip-installed from PyPI; stable versions are re-added to runtime_config.json by the caller.
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

    # Strip all pixeltable* entries from the pip section and capture the pixeltable version.
    # pixeltable dev builds (version contains '+' or '.dev') cannot be pip-installed from PyPI.
    env_text = result.stdout.decode('utf-8')
    detected_version: str | None = None
    filtered = []
    for line in env_text.splitlines(keepends=True):
        if re.match(r'^\s+-\s+pixeltable==', line):
            detected_version = line.strip().split('==', 1)[1].strip()
        elif re.match(r'^\s+-\s+pixeltable', line):
            pass  # strip pixeltable-pgserver, pixeltable-cloud, etc.
        else:
            filtered.append(line)
    return ''.join(filtered).encode('utf-8'), detected_version


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

    from pixeltable.serving._config import lookup_database_runtime_config

    runtime_cfg = lookup_database_runtime_config()
    include = runtime_cfg.include if runtime_cfg else None
    exclude = runtime_cfg.exclude if runtime_cfg else None
    pxt_source = runtime_cfg.pixeltable_source if runtime_cfg else None

    lockfile = _find_lockfile(project_dir)
    if lockfile is None:
        Env.get().console_logger.warning(
            'No dependency lockfile was found (uv.lock, poetry.lock, requirements.txt).\n'
            'The runtime may not have the necessary dependencies.'
        )

    conda_result = _export_conda_env()
    conda_export: bytes | None = None
    conda_pxt_version: str | None = None
    if conda_result is not None:
        conda_export, conda_pxt_version = conda_result

    # Determine the effective pixeltable source for runtime_config.json.
    # Priority: explicit pixeltable.toml config > stable conda version > warn (dev build).
    effective_pxt_source = pxt_source
    if effective_pxt_source is None and conda_pxt_version is not None:
        if _is_stable_pypi_version(conda_pxt_version):
            effective_pxt_source = config.PixeltableSourceConfig(version=conda_pxt_version)
        else:
            Env.get().console_logger.warning(
                f'Detected a local dev pixeltable build ({conda_pxt_version}) that cannot be '
                'installed from PyPI. Add a [pixeltable.database.pixeltable_source] section to '
                'pixeltable.toml with a git ref pointing to your branch so the runtime image '
                'uses the correct pixeltable version.'
            )

    files = _collect_project_files(project_dir, include, exclude)

    fd, name = tempfile.mkstemp(suffix='.tar.bz2', prefix='pxt_runtime_')
    os.close(fd)
    bundle_path = Path(name)

    with tarfile.open(bundle_path, 'w:bz2') as tf:
        manifest = {'pxt_md_version': metadata.VERSION}
        __add_tarfile(tf, 'metadata.json', json.dumps(manifest).encode('utf-8'))
        if conda_export is not None:
            __add_tarfile(tf, 'environment.yml', conda_export)
        if effective_pxt_source is not None:
            rt_config = {'pixeltable_source': effective_pxt_source.model_dump(exclude_none=True)}
            __add_tarfile(tf, 'runtime_config.json', json.dumps(rt_config).encode('utf-8'))
        for f in sorted(files):
            relpath = f.relative_to(project_dir)
            tf.add(f, arcname=f'project/{relpath}')

    _logger.info(f'Runtime bundle created: {bundle_path}')
    return bundle_path
