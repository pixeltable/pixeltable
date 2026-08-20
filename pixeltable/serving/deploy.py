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
from pathspec import PathSpec
from tqdm import tqdm

from pixeltable import config, exceptions as excs, metadata
from pixeltable.env import Env
from pixeltable.serving._config import lookup_database_config

_logger = logging.getLogger(__name__)


def _resolve_patterns(project_dir: Path, patterns: list[str]) -> set[Path]:
    """Files under project_dir matching patterns, which use git wildmatch syntax (`*`, `**`, `!`, `dir/`)."""
    # 'gitignore' names the pattern dialect to parse `patterns` with; no .gitignore file is read here
    spec = PathSpec.from_lines('gitignore', patterns)
    return {p for p in project_dir.rglob('*') if p.is_file() and spec.match_file(p.relative_to(project_dir))}


def _gitignore_spec(dir_path: Path) -> PathSpec | None:
    """The PathSpec for dir_path's own .gitignore, or None if it has none."""
    gitignore = dir_path / '.gitignore'
    if not gitignore.is_file():
        return None
    return PathSpec.from_lines('gitignore', gitignore.read_text().splitlines())


def _is_gitignored(path: Path, is_dir: bool, specs: list[tuple[Path, PathSpec]]) -> bool:
    """Whether path is ignored by specs, a list of (directory, its .gitignore) ordered outermost first.

    The innermost .gitignore that has anything to say about path decides, since git lets a nested
    .gitignore override the directories above it. Patterns are matched relative to the directory the
    .gitignore lives in, and a trailing slash is what makes a directory-only pattern (`build/`) match.
    """
    for base, spec in reversed(specs):
        rel = path.relative_to(base).as_posix() + ('/' if is_dir else '')
        include = spec.check_file(rel).include
        if include is not None:
            return include
    return False


def _collect_unignored_files(project_dir: Path) -> set[Path]:
    """All files under project_dir that git would not ignore.

    Honors the .gitignore at every level of the tree, not just project_dir's: tools such as ruff, mypy and
    pytest keep their caches out of git by writing a `.gitignore` containing `*` into the cache directory
    itself, so a root-only scan bundles those caches even though `git status` reports a clean tree.

    An ignored directory is not descended into, matching git's rule that a nested negation cannot
    re-include a file whose parent directory is excluded. Directory symlinks are not followed (git stores
    them as symlinks rather than recursing).

    .git is skipped here, as git itself does, but only by default: an `include` pattern of `.git/**` still
    reaches it, which a project that derives its version from VCS metadata needs.
    """
    files: set[Path] = set()

    def visit(dir_path: Path, specs: list[tuple[Path, PathSpec]]) -> None:
        spec = _gitignore_spec(dir_path)
        if spec is not None:
            specs = [*specs, (dir_path, spec)]
        for entry in dir_path.iterdir():
            if entry.name == '.git':
                continue
            is_dir = entry.is_dir() and not entry.is_symlink()
            if _is_gitignored(entry, is_dir, specs):
                continue
            if is_dir:
                visit(entry, specs)
            elif entry.is_file():
                files.add(entry)

    visit(project_dir, [])
    return files


def _collect_project_files(
    project_dir: Path, exclude: list[str] | None, include: list[str] | None, include_only: list[str] | None
) -> list[Path]:
    if include_only is not None:
        if include is not None or exclude is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                'Cannot specify both include_only and include/exclude in [pixeltable.database] configuration',
            )
        files = _resolve_patterns(project_dir, include_only)

    else:
        # Apply .gitignore excludes, at every level of the tree
        files = _collect_unignored_files(project_dir)
        # Apply explicit excludes
        if exclude is not None:
            files -= _resolve_patterns(project_dir, exclude)
        # Apply explicit includes (which override excludes)
        if include is not None:
            files |= _resolve_patterns(project_dir, include)

    return sorted(files)


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


def _load_database_config(project_dir: Path) -> config.DatabaseConfig | None:
    """Read [pixeltable.database] config from project_dir/pixeltable.toml; fall back to Config singleton."""
    toml_path = project_dir / 'pixeltable.toml'
    if toml_path.is_file():
        try:
            parsed = toml.load(toml_path)
        except Exception as e:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid TOML in {toml_path.name}: {e}'
            ) from e
        db_raw = parsed.get('pixeltable', {}).get('database')
        if db_raw is not None:
            try:
                return config.DatabaseConfig.model_validate(db_raw)
            except Exception as e:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid [pixeltable.database] configuration: {e}'
                ) from e
    return lookup_database_config()


def _abbrev_path(path: str, max_len: int = 40) -> str:
    """Truncate path from the left, so that a long path doesn't wrap the progress bar line."""
    return path if len(path) <= max_len else '…' + path[-(max_len - 1) :]


def __add_tarfile(tf: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    tf.addfile(info, fileobj=io.BytesIO(content))


def build_db_runtime_bundle(project_dir: Path | None = None, show_progress: bool = False) -> Path:
    """Package the current project into a tarball for updating a hosted database runtime.

    If show_progress is True, a progress bar tracking the number of project files added, and naming the
    file most recently added, is displayed.

    Bundle layout:
        metadata.json   (always) — pxt_md_version, python_version
        project/        (always) — all project source files including uv.lock, pyproject.toml, etc.

    The server reads project/uv.lock and runs `uv sync --frozen` to install Python packages.
    System dependencies declared in pixeltable.toml [pixeltable.database] system_dependencies
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

    runtime_cfg = _load_database_config(project_dir)
    exclude = runtime_cfg.exclude if runtime_cfg else None
    include = runtime_cfg.include if runtime_cfg else None
    include_only = runtime_cfg.include_only if runtime_cfg else None
    system_dependencies: list[str] = (runtime_cfg.system_dependencies or []) if runtime_cfg else []

    # Config override wins; otherwise use the deploy environment's version.
    python_version = (runtime_cfg.python_version if runtime_cfg else None) or (
        f'{sys.version_info.major}.{sys.version_info.minor}'
    )

    files_set = set(_collect_project_files(project_dir, exclude, include, include_only))
    # Lock files are always bundled regardless of .gitignore — they control reproducible installs.
    has_lockfile = False
    for lock_name in ('uv.lock', 'poetry.lock', 'requirements.txt'):
        lock_path = project_dir / lock_name
        if lock_path.is_file():
            files_set.add(lock_path)
            has_lockfile = True
    files = sorted(files_set)

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
    if runtime_cfg and runtime_cfg.pixeltable_source:
        meta['pixeltable_source'] = runtime_cfg.pixeltable_source.model_dump(exclude_none=True)

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
