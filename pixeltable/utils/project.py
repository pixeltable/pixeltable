"""The files a project consists of, and the fingerprint describing them."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pydantic
from pathspec import PathSpec

import pixeltable
from pixeltable import exceptions as excs
from pixeltable.config import DatabaseConfig

# a project declares its packages in one of these, so the selection always includes them
_LOCK_FILES = ('uv.lock', 'poetry.lock', 'requirements.txt')


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


def selected_files(project_root: Path, config: DatabaseConfig | None) -> list[Path]:
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

    files |= {project_root / name for name in _LOCK_FILES if (project_root / name).is_file()}
    return sorted(files)


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

    # bindings, never resolved values: a secret names the source of its value
    vars: dict[str, str]
    secrets: dict[str, str]

    def rebuild_needed(self, deployed: ProjectFingerprint) -> bool:
        """Whether an image built from deployed would differ from one built now."""
        return self._build_inputs() != deployed._build_inputs()

    def restart_needed(self, deployed: ProjectFingerprint) -> bool:
        """Whether a process started from deployed would differ from one started now."""
        return self != deployed

    def changes(self, deployed: ProjectFingerprint) -> list[str]:
        """What differs from deployed, one printable line each."""
        lines = [
            f'{path} changed'
            for path in sorted(set(self.files) & set(deployed.files))
            if self.files[path] != deployed.files[path]
        ]
        lines += [f'{path} added' for path in sorted(set(self.files) - set(deployed.files))]
        lines += [f'{path} removed' for path in sorted(set(deployed.files) - set(self.files))]
        for field in ('python_version', 'pixeltable_version'):
            was, now = getattr(deployed, field), getattr(self, field)
            if was != now:
                lines.append(f'{field} {was} -> {now}')
        if self.system_dependencies != deployed.system_dependencies:
            lines.append('system_dependencies changed')
        lines += [f'var {name} changed' for name in _changed_keys(self.vars, deployed.vars)]
        lines += [f'secret {name} changed' for name in _changed_keys(self.secrets, deployed.secrets)]
        return lines

    def _build_inputs(self) -> tuple:
        return (self.files, self.python_version, self.system_dependencies, self.pixeltable_version)


def _changed_keys(now: dict[str, str], was: dict[str, str]) -> list[str]:
    return sorted({name for name in set(now) | set(was) if now.get(name) != was.get(name)})


def project_fingerprint(project_root: Path, config: DatabaseConfig | None) -> ProjectFingerprint:
    """Fingerprint the project at project_root, given the specific config."""
    files = {
        path.relative_to(project_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in selected_files(project_root, config)
    }
    declared_python = config.python_version if config is not None else None
    return ProjectFingerprint(
        files=files,
        # the version an image would use: the entry's, or the running interpreter's
        python_version=declared_python or f'{sys.version_info.major}.{sys.version_info.minor}',
        system_dependencies=(config.system_dependencies if config is not None else None) or [],
        pixeltable_version=pixeltable.__version__,
        vars=(config.vars if config is not None else None) or {},
        secrets=(config.secrets if config is not None else None) or {},
    )
