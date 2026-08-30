"""The files a project consists of, and the fingerprint describing them."""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pydantic
from pathspec import PathSpec
from tqdm import tqdm

import pixeltable
from pixeltable import exceptions as excs
from pixeltable.config import DatabaseConfig
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')

# how an image build installs the project's packages
DepsType = Literal['uv', 'poetry', 'pip', 'none']

# a project declares its packages in one of these, each installed by the tool it names
LOCK_FILES: dict[str, DepsType] = {'uv.lock': 'uv', 'poetry.lock': 'poetry', 'requirements.txt': 'pip'}


class ProjectPart(enum.StrEnum):
    """The parts that make up a project's fingerprint."""

    IMAGE = 'image'

    ARCHIVE = 'archive'

    # vars and secrets
    BINDINGS = 'bindings'


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


def _archive_files(project_root: Path, config: DatabaseConfig | None) -> list[Path]:
    """The files that go into an image, and into the fingerprint.

    Everything git would not ignore, adjusted by the entry's include/exclude patterns, plus the lockfile,
    which declares the project's packages and is selected whatever the patterns say.
    """
    exclude = config.exclude if config is not None else None
    include = config.include if config is not None else None
    include_only = config.include_only if config is not None else None

    if include_only is not None:
        if include is not None or exclude is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                'Cannot specify both include_only and include/exclude in a [[pixeltable.database]] entry',
            )
        files = _resolve_patterns(project_root, include_only)

    else:
        # Apply .gitignore excludes, at every level of the tree
        files = _collect_unignored_files(project_root)
        # Apply explicit excludes
        if exclude is not None:
            files -= _resolve_patterns(project_root, exclude)
        # Apply explicit includes (which override excludes)
        if include is not None:
            files |= _resolve_patterns(project_root, include)

    files |= {project_root / name for name in LOCK_FILES if (project_root / name).is_file()}
    return sorted(files)


def create_project_archive(
    project_dir: Path | None = None, db_config: DatabaseConfig | None = None, show_progress: bool = False
) -> Path:
    """Produce an archive (tar file) of the project files, as selected by db_config.

    Includes every git-recognized file below the project root, plus the lockfile.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()

    if not project_dir.is_dir():
        raise FileNotFoundError(f'Project directory does not exist: {project_dir}')

    files = _archive_files(project_dir, db_config)
    has_lockfile = any(file.parent == project_dir and file.name in LOCK_FILES for file in files)

    print(f'Packaging {len(files)} files from {project_dir}.')
    print(
        'By default, all files not ignored by .gitignore are included; '
        'you can adjust this behavior with include/exclude in pixeltable.toml.'
    )

    if not has_lockfile:
        Env.get().console_logger.warning(
            'No dependency lockfile (uv.lock, poetry.lock, requirements.txt) was found in '
            f'{project_dir}.\nThe image will hold Pixeltable and nothing else, so it may not have the '
            'Python dependencies the project needs. An active conda environment is not a substitute: '
            "run 'uv lock', or write a requirements.txt."
        )

    fd, name = tempfile.mkstemp(suffix='.tar.bz2', prefix='pxt_project_')
    os.close(fd)
    archive_path = Path(name)

    max_pathlen = 40
    with (
        tarfile.open(archive_path, 'w:bz2') as tf,
        tqdm(desc='Packaging project', total=len(files), unit=' files', disable=not show_progress) as bar,
    ):
        for f in files:
            relpath = str(f.relative_to(project_dir))
            abbrev_path = relpath if len(relpath) <= max_pathlen else '…' + relpath[-(max_pathlen - 1) :]
            # refresh=False: the postfix is drawn by the following update(), which respects tqdm's redraw interval
            bar.set_postfix_str(abbrev_path, refresh=False)
            tf.add(f, arcname=f'project/{relpath}')
            bar.update(1)
        bar.set_postfix_str('', refresh=False)

    _logger.info(f'Project archive created: {archive_path}')
    return archive_path


class ProjectFingerprint(pydantic.BaseModel):
    """
    Captures the environment that affects a Pixeltable process started from a project, for the purpose of comparison.
    """

    model_config = pydantic.ConfigDict(extra='ignore')

    # path relative to the project root -> sha256 of the file's contents
    files: dict[str, str]

    python_version: str
    system_dependencies: list[str]
    pixeltable_version: str
    uv_options: str | None = None

    # bindings, never resolved values: a secret names the source of its value
    vars: dict[str, str]
    secrets: dict[str, str]

    def compare(self, other: ProjectFingerprint, *, own_files_only: bool = False) -> set[ProjectPart]:
        """The parts that differ from other.

        own_files_only compares only the files in this fingerprint and excludes files that exist only in other.
        """
        parts: set[ProjectPart] = set()
        if self._image_inputs() != other._image_inputs():
            parts.add(ProjectPart.IMAGE)
        files_differ = len(self._added_or_changed(other)) > 0 if own_files_only else self.files != other.files
        if files_differ:
            parts.add(ProjectPart.ARCHIVE)
        if (self.vars, self.secrets) != (other.vars, other.secrets):
            parts.add(ProjectPart.BINDINGS)
        return parts

    def image_digest(self) -> str:
        """The identity of an image built for this environment.

        Two fingerprints share it exactly when compare() reports no IMAGE difference, so an environment
        that has been built once is never built again, whichever project declares it.
        """
        return _digest(self._image_inputs())

    def archive_digest(self) -> str:
        """The identity of the archive this project's files package into.

        Two fingerprints share it exactly when compare() reports no ARCHIVE difference.
        """
        return _digest(self.files)

    def changes(
        self, other: ProjectFingerprint, parts: set[ProjectPart] | None = None, *, own_files_only: bool = False
    ) -> list[str]:
        """What differs from other in the given parts, one printable line each; every part by default.

        own_files_only compares only the files in this fingerprint and excludes files that exist only in other.
        """
        if parts is None:
            parts = set(ProjectPart)
        lines: list[str] = []
        if ProjectPart.ARCHIVE in parts:
            lines += self._added_or_changed(other)
            if not own_files_only:
                lines += [f'{path} removed' for path in sorted(set(other.files) - set(self.files))]
        if ProjectPart.IMAGE in parts:
            if ProjectPart.ARCHIVE not in parts:
                # make sure to include the lock files
                lines += _changed_paths(self._lock_files(), other._lock_files())
            for field in ('python_version', 'pixeltable_version'):
                was, now = getattr(other, field), getattr(self, field)
                if was != now:
                    lines.append(f'{field} {was} -> {now}')
            if self.system_dependencies != other.system_dependencies:
                lines.append('system_dependencies changed')
            if self.uv_options != other.uv_options:
                lines.append('uv_options changed')
        if ProjectPart.BINDINGS in parts:
            lines += [f'var {name} changed' for name in _changed_keys(self.vars, other.vars)]
            lines += [f'secret {name} changed' for name in _changed_keys(self.secrets, other.secrets)]
        return lines

    def _added_or_changed(self, other: ProjectFingerprint) -> list[str]:
        """The files in this fingerprint that are changed or absent in other."""
        return _changed_paths(self.files, other.files) + [
            f'{path} added' for path in sorted(set(self.files) - set(other.files))
        ]

    def deps_type(self) -> DepsType:
        """The tool that installs the project's packages."""
        return next((tool for name, tool in LOCK_FILES.items() if name in self.files), 'none')

    def _image_inputs(self) -> tuple:
        return (
            self._lock_files(),
            self.python_version,
            self.system_dependencies,
            self.pixeltable_version,
            self.uv_options,
        )

    def _lock_files(self) -> dict[str, str]:
        return {path: content_hash for path, content_hash in self.files.items() if path in LOCK_FILES}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _changed_paths(now: dict[str, str], was: dict[str, str]) -> list[str]:
    """One line per path both hold with different contents."""
    return [f'{path} changed' for path in sorted(set(now) & set(was)) if now[path] != was[path]]


def _changed_keys(now: dict[str, str], was: dict[str, str]) -> list[str]:
    return sorted({name for name in set(now) | set(was) if now.get(name) != was.get(name)})


def project_fingerprint(project_root: Path, config: DatabaseConfig | None) -> ProjectFingerprint:
    """Fingerprint every file an image built from project_root would hold.

    This decides whether an image is out of date, since an image holds the whole project.
    """
    return _fingerprint(_archive_files(project_root, config), project_root, config)


def loaded_fingerprint(project_root: Path, config: DatabaseConfig | None) -> ProjectFingerprint:
    """Fingerprint the files the loaded application reached under project_root, plus the lockfile.

    This decides whether a running service is out of date, so that a service restarts for a file it imports
    and not for one its neighbour in the same project imports.

    Call it after load_app_module(), which evicts the project's modules before importing: what is loaded from
    the project afterwards is what this application reached. A module imported inside a function body is
    never reached, which is the limitation Modal's mounted-source rule has too, and the reason
    'pxt service update --restart' exists.
    """
    loaded = {
        Path(file).resolve()
        for file in (getattr(module, '__file__', None) for module in list(sys.modules.values()))
        if file is not None
    }
    files = [path for path in loaded if path.is_relative_to(project_root) and path.is_file()]
    files += [project_root / name for name in LOCK_FILES if (project_root / name).is_file()]
    return _fingerprint(files, project_root, config)


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        # chunked reads: limit buffering
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(files: Iterable[Path], project_root: Path, config: DatabaseConfig | None) -> ProjectFingerprint:
    files = {path.relative_to(project_root).as_posix(): _content_hash(path) for path in files}
    declared_python = config.python_version if config is not None else None
    return ProjectFingerprint(
        files=files,
        # the version an image would use: the entry's, or the running interpreter's
        python_version=declared_python or f'{sys.version_info.major}.{sys.version_info.minor}',
        system_dependencies=(config.system_dependencies if config is not None else None) or [],
        pixeltable_version=pixeltable.__version__,
        uv_options=config.uv_options if config is not None else None,
        vars=(config.vars if config is not None else None) or {},
        secrets=(config.secrets if config is not None else None) or {},
    )
