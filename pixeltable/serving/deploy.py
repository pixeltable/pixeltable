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
from pixeltable.serving._config import lookup_database_runtime_config

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


def _detect_pxt_version() -> str | None:
    """Return the installed pixeltable version string, or None if not installed."""
    try:
        import importlib.metadata

        return importlib.metadata.version('pixeltable')
    except Exception:
        return None


def _export_conda_env() -> bytes | None:
    """Export the active conda environment as YAML, stripping pixeltable* packages.

    Pixeltable is installed separately via runtime_config.json, so it must be excluded here.
    The exported environment.yml is included in the bundle for reference but the server-side
    Dockerfile builder no longer uses it — uv export (requirements.txt) takes priority.
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
    env_text = result.stdout.decode('utf-8')
    filtered = [line for line in env_text.splitlines(keepends=True) if not re.match(r'^\s+-\s+pixeltable', line)]
    return ''.join(filtered).encode('utf-8')


def _export_uv_requirements(project_dir: Path) -> bytes | None:
    """Generate a cross-platform requirements.txt from uv.lock.

    Runs `uv export --frozen --no-dev --no-emit-project --no-hashes` targeting the project's
    uv.lock. The --no-hashes flag produces plain package==version lines with no wheel URLs or
    checksums, so pip on CodeBuild (linux/amd64) resolves the correct platform wheels from PyPI
    regardless of what platform the client is running on.

    pixeltable is excluded via --no-emit-package because it is installed separately from
    runtime_config.json (with the [serve] extra and potentially a git ref).
    Returns None if uv.lock is absent or uv export fails.
    """
    if not (project_dir / 'uv.lock').exists():
        return None
    try:
        result = subprocess.run(
            [
                'uv',
                'export',
                '--frozen',
                '--no-dev',
                '--no-emit-project',
                '--no-hashes',
                '--no-emit-package',
                'pixeltable',
            ],
            capture_output=True,
            check=True,
            cwd=project_dir,
        )
        Env.get().console_logger.info('Generated cross-platform requirements.txt from uv.lock')
        return result.stdout
    except FileNotFoundError:
        Env.get().console_logger.warning('uv not found; skipping uv export (install uv to include lockfile deps)')
        return None
    except subprocess.CalledProcessError as exc:
        Env.get().console_logger.warning(f'uv export failed: {exc.stderr.decode(errors="replace").strip()}')
        return None


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

    Bundles project source files, dependency lockfiles, and an optional pixeltable source
    override. Service configuration is not included — services are managed independently.

    Bundle layout:
        metadata.json        (always present)
        environment.yml      (optional — conda env export, included for reference, not used to build)
        requirements.txt     (optional — generated from uv.lock via uv export; used to build image)
        runtime_config.json  (optional — present when [pixeltable.database.pixeltable_source] is configured)
        project/             (all project source files: UDFs, lockfiles, pyproject.toml, etc.)

    Dependency resolution on the server (priority order):
        requirements.txt (root) → project/uv.lock → project/poetry.lock → project/requirements.txt → none
    environment.yml is intentionally ignored by the server-side Dockerfile builder.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()

    if not project_dir.is_dir():
        raise FileNotFoundError(f'Project directory does not exist: {project_dir}')

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

    conda_export = _export_conda_env()
    uv_requirements = _export_uv_requirements(project_dir)

    # Determine the effective pixeltable source for runtime_config.json.
    # Priority: explicit pixeltable.toml config > stable installed version > warn (dev build).
    effective_pxt_source = pxt_source
    if effective_pxt_source is None:
        installed_version = _detect_pxt_version()
        if installed_version is not None:
            if _is_stable_pypi_version(installed_version):
                effective_pxt_source = config.PixeltableSourceConfig(version=installed_version)
            else:
                Env.get().console_logger.warning(
                    f'Detected a local dev pixeltable build ({installed_version}) that cannot be '
                    'installed from PyPI. Add a [pixeltable.database.pixeltable_source] section to '
                    'pixeltable.toml with a git ref pointing to your branch so the runtime image '
                    'uses the correct pixeltable version.'
                )

    files = _collect_project_files(project_dir, include, exclude)

    fd, name = tempfile.mkstemp(suffix='.tar.bz2', prefix='pxt_runtime_')
    os.close(fd)
    bundle_path = Path(name)

    with tarfile.open(bundle_path, 'w:bz2') as tf:
        __add_tarfile(tf, 'metadata.json', json.dumps({'pxt_md_version': metadata.VERSION}).encode('utf-8'))
        if conda_export is not None:
            __add_tarfile(tf, 'environment.yml', conda_export)
        if uv_requirements is not None:
            __add_tarfile(tf, 'requirements.txt', uv_requirements)
        if effective_pxt_source is not None:
            rt_config = {'pixeltable_source': effective_pxt_source.model_dump(exclude_none=True)}
            __add_tarfile(tf, 'runtime_config.json', json.dumps(rt_config).encode('utf-8'))
        for f in sorted(files):
            relpath = f.relative_to(project_dir)
            tf.add(f, arcname=f'project/{relpath}')

    _logger.info(f'Runtime bundle created: {bundle_path}')
    return bundle_path
