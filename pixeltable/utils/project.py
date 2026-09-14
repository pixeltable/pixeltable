"""The files a project consists of, and the fingerprint describing them."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import os
import re
import sys
import sysconfig
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO, Literal

import pydantic
import toml
from pathspec import PathSpec
from tqdm import tqdm

import pixeltable
from pixeltable import exceptions as excs
from pixeltable.config import PROJECT_CONFIG_FILES, PYPROJECT_FILE, DatabaseConfig
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')

# how an image build installs the project's packages
DepsType = Literal['uv', 'pip', 'none']

# a project declares its packages in one of these, each installed by the tool it names
LOCK_FILES: dict[str, DepsType] = {'uv.lock': 'uv', 'requirements.txt': 'pip'}

IMAGE_INPUT_FILES: tuple[str, ...] = (*LOCK_FILES, PYPROJECT_FILE)


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


def _is_venv(dir_path: Path) -> bool:
    """Whether dir_path is a Python virtual environment."""
    return (dir_path / 'pyvenv.cfg').is_file() or (dir_path / 'conda-meta').is_dir()


def _collect_unignored_files(project_dir: Path) -> set[Path]:
    """All files under project_dir that git would not ignore, minus any virtual environment.

    Honors the .gitignore at every level of the tree, not just project_dir's: tools such as ruff, mypy and
    pytest keep their caches out of git by writing a `.gitignore` containing `*` into the cache directory
    itself, so a root-only scan bundles those caches even though `git status` reports a clean tree.

    An ignored directory is not descended into, matching git's rule that a nested negation cannot
    re-include a file whose parent directory is excluded. Directory symlinks are not followed (git stores
    them as symlinks rather than recursing).

    .git is skipped here, as git itself does, but only by default: an `include` pattern of `.git/**` still
    reaches it, which a project that derives its version from VCS metadata needs. A virtual environment is
    skipped whether or not a .gitignore covers it, since the pod installs the packages from the lockfile.
    __pycache__ is ignored: we don't want to ship bytecode
    """
    files: set[Path] = set()

    def visit(dir_path: Path, specs: list[tuple[Path, PathSpec]]) -> None:
        spec = _gitignore_spec(dir_path)
        if spec is not None:
            specs = [*specs, (dir_path, spec)]
        for entry in dir_path.iterdir():
            if entry.name in ('.git', '__pycache__'):
                continue
            is_dir = entry.is_dir() and not entry.is_symlink()
            if _is_gitignored(entry, is_dir, specs):
                continue
            if is_dir and _is_venv(entry):
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

    # we always include the lockfile and project config files, both are needed by the pod
    selected = (*LOCK_FILES, *PROJECT_CONFIG_FILES)
    files |= {project_root / name for name in selected if (project_root / name).is_file()}
    return sorted(files)


class _HashingReader:
    """Hashes every byte it yields."""

    def __init__(self, f: BinaryIO) -> None:
        self._f = f
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        self._digest.update(data)
        return data

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


@dataclasses.dataclass
class PackagedArchive:
    """An archive, and the content hash of every file written into it."""

    path: Path

    # path relative to the project root -> sha256 of the bytes written into the archive
    files: dict[str, str]


@dataclasses.dataclass
class PackagedContext:
    """An image context, and the content hash of every file written into it."""

    path: Path

    # path relative to the project root -> sha256 of the bytes written into the context
    files: dict[str, str]


def _member_hash(content_hash: str, *, symlink: bool, executable: bool) -> str:
    """One hash over what an unpacked project holds at a path: its bytes, its kind and its execute bit.

    Ownership and timestamps are left out: two packagings of one project differ in them, and tarfile's
    'data' extraction filter discards them. That filter preserves the execute bit, so this hash covers it.
    """
    return _digest({'content': content_hash, 'symlink': symlink, 'executable': executable})


def _path_hash(path: Path) -> str:
    """Read the file at path and hash it the way _add_hashed() hashes an archive member."""
    if path.is_symlink():
        # a symlink holds a path, so the path identifies it; reading through it would hash the target
        return _member_hash(_digest(os.readlink(path)), symlink=True, executable=False)
    return _member_hash(_content_hash(path), symlink=False, executable=bool(path.stat().st_mode & 0o111))


def _add_hashed(tf: tarfile.TarFile, path: Path, arcname: str) -> str:
    """Write path into tf and return the hash of the member written."""
    info = tf.gettarinfo(path, arcname=arcname)
    if info.issym():
        tf.addfile(info)
        return _member_hash(_digest(info.linkname), symlink=True, executable=False)
    if info.islnk():
        # gettarinfo() writes a second path to one inode as a hard link, which extracts as a regular
        # file holding the same bytes; the member itself holds no content
        tf.addfile(info)
        return _member_hash(_content_hash(path), symlink=False, executable=bool(info.mode & 0o111))
    with path.open('rb') as raw:
        reader = _HashingReader(raw)
        tf.addfile(info, reader)
    return _member_hash(reader.hexdigest(), symlink=False, executable=bool(info.mode & 0o111))


def create_project_archive(
    project_dir: Path | None = None, db_config: DatabaseConfig | None = None, show_progress: bool = False
) -> Path:
    """Produce an archive (tar file) of the project files, as selected by db_config."""
    return package_project_archive(project_dir, db_config, show_progress).path


def package_project_archive(
    project_dir: Path | None = None, db_config: DatabaseConfig | None = None, show_progress: bool = False
) -> PackagedArchive:
    """Produce an archive of the project files, as selected by db_config, and say what went into it.

    Includes every git-recognized file below the project root, plus the lockfile. The returned hashes are
    taken from the bytes written, so they describe the archive rather than a later reading of the project.
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
            'No dependency lockfile (uv.lock, requirements.txt) was found in '
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
        hashes: dict[str, str] = {}
        for f in files:
            relpath = f.relative_to(project_dir).as_posix()
            abbrev_path = relpath if len(relpath) <= max_pathlen else '…' + relpath[-(max_pathlen - 1) :]
            # refresh=False: the postfix is drawn by the following update(), which respects tqdm's redraw interval
            bar.set_postfix_str(abbrev_path, refresh=False)
            hashes[relpath] = _add_hashed(tf, f, f'project/{relpath}')
            bar.update(1)
        bar.set_postfix_str('', refresh=False)

    _logger.info(f'Project archive created: {archive_path}')
    return PackagedArchive(path=archive_path, files=hashes)


# what pip installs from a file rather than an index, named without a directory
_ARCHIVE_SUFFIXES = ('.whl', '.zip', '.tar.gz', '.tar.bz2', '.tar.xz', '.tgz')

# a PEP 508 name, optionally with extras: the text before a direct reference's '@'
_REQUIREMENT_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?$')


def _direct_reference(requirement: str) -> str | None:
    """The target of a 'name @ target' requirement, or None if requirement names a package from an index.

    PEP 508 makes the whitespace around the '@' optional, and a filename may also contain an '@', so the
    text before the first '@' decides: only a package name makes what follows a target.
    """
    name, sep, target = requirement.partition('@')
    if sep == '' or _REQUIREMENT_NAME.match(name.strip()) is None:
        return None
    return target.strip()


def _declared_dependencies(parsed: dict[str, Any]) -> list[str]:
    """Every dependency in pyproject.toml, across the tables a build tool installs from."""
    project = parsed.get('project', {})
    uv = parsed.get('tool', {}).get('uv', {})
    groups: list[list[Any]] = [
        project.get('dependencies', []),
        # a build backend is installed before the project, from requirements written the same way
        parsed.get('build-system', {}).get('requires', []),
        uv.get('constraint-dependencies', []),
        uv.get('override-dependencies', []),
    ]
    groups += list(project.get('optional-dependencies', {}).values())
    groups += list(parsed.get('dependency-groups', {}).values())
    # a dependency group also takes {'include-group': ...}, which names another group rather than a package
    return [entry for group in groups for entry in group if isinstance(entry, str)]


def _local_index_locations(parsed: dict[str, Any]) -> list[str]:
    """The package locations in pyproject.toml that lie on this machine."""
    found = parsed.get('tool', {}).get('uv', {}).get('find-links', [])
    return [entry for entry in found if isinstance(entry, str) and (entry.startswith('file:') or '://' not in entry)]


def _local_lock_sources(parsed: dict[str, Any], project_dir: Path) -> list[str]:
    """The packages uv.lock installs from a path rather than an index, other than the project itself.

    uv records the project's own package as a source too, at the project root; the archive carries
    that one.

    TODO: carry a path, directory or editable source into the image context and into
    installed_from_project, as _local_requirement_files() does for requirements.txt. Until then the
    context holds no such dependency, and image_digest() does not move when one changes.
    """
    local: list[str] = []
    for package in parsed.get('package', []):
        source = package.get('source', {}) if isinstance(package, dict) else {}
        if not isinstance(source, dict):
            continue
        for key in ('path', 'directory', 'editable', 'virtual'):
            target = source.get(key)
            if not isinstance(target, str):
                continue
            if (project_dir / target).resolve() == project_dir:
                continue
            local.append(f'{package.get("name", "?")} ({key} = {target!r})')
    return local


def _requirement_lines(text: str) -> list[str]:
    lines: list[str] = []
    pending = ''
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith('\\'):
            pending += stripped[:-1]
            continue
        lines.append(pending + stripped)
        pending = ''
    if pending != '':
        lines.append(pending)
    return lines


def _find_links_target(line: str) -> str | None:
    """Where a --find-links option points, or None if line sets another option."""
    for name in ('--find-links', '-f'):
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest == '':
            return ''
        if rest[0] in '= ':
            return rest[1:].strip()
        # optparse accepts '-fVALUE'; the '--' guard keeps '--find-links' and other long options out
        if name == '-f' and not line.startswith('--'):
            return rest.strip()
    return None


def _local_requirement_files(project_dir: Path, requirements: Path) -> list[Path]:
    """The files requirements.txt installs from a path in the project, rather than from an index or a url."""
    files: list[Path] = []
    for raw in _requirement_lines(requirements.read_text(encoding='utf-8')):
        line = raw.split('#', 1)[0].strip()
        if line == '':
            continue
        if line.startswith(('-r', '--requirement', '-c', '--constraint')):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} reads another file ({line}), which is not supported; '
                'write one file naming every dependency',
            )
        if line.startswith(('-e', '--editable')):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} declares {line!r}, an editable install, which a hosted image does not '
                'support; publish the package to an index and depend on the published version',
            )
        find_links = _find_links_target(line)
        if find_links is not None and (find_links.startswith('file:') or '://' not in find_links):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} looks for packages in {find_links}, a location on this machine; '
                'instead, publish the packages to an index and depend on the published versions, so that '
                'they can get picked up by the hosted image build',
            )
        if line.startswith('-'):
            continue
        # an environment marker decides whether pip installs the line
        line = line.split(';', 1)[0].strip()
        if line == '':
            continue

        target = _direct_reference(line)
        if target is None:
            target = line
            # pip reads a bare name as a package, not a path, unless it carries an archive suffix
            if '/' not in target and not target.startswith('.') and not target.endswith(_ARCHIVE_SUFFIXES):
                continue
        if target.startswith('file:'):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} installs {target}, a file: url naming this machine; write the path '
                'relative to the project root instead',
            )
        if '://' in target:
            continue
        if Path(target).is_absolute():
            # the context holds the file under a path relative to the project, so pip in the build
            # container would look for this one where nothing is
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} installs {target}, an absolute path naming this machine; write it '
                'relative to the project root instead',
            )

        root = project_dir.resolve()
        path = (root / target).resolve()
        if path != Path(os.path.normpath(root / target)):
            # resolving follows the symlink, so the context holds the target's name; pip reads the spelling
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} installs {target} through a symlink; write the path of the file '
                'itself, since pip reads the path as spelled',
            )
        if path.is_dir():
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} installs {target}, a source directory, which a hosted image build '
                'cannot compile; publish the package to an index and depend on the published version',
            )
        if not path.is_relative_to(project_dir):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} installs {target}, which is outside the project; an image build '
                'sends the project alone, so a file above it cannot be installed',
            )
        if not path.is_file():
            # this path doesn't exist
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{requirements.name} installs {target}, which cannot be resolved relative to the project root',
            )
        files.append(path)
    return files


def create_image_context(project_dir: Path | None = None) -> Path:
    """Return the path to a tarfile containing the manifests needed for an image build."""
    return package_image_context(project_dir).path


def package_image_context(project_dir: Path | None = None) -> PackagedContext:
    """Package the manifests an image build needs, and say what went into it.

    The returned hashes are taken from the bytes written, so they describe the context rather than a
    later reading of the project.
    """
    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = project_dir.resolve()
    files = [project_dir / name for name in IMAGE_INPUT_FILES if (project_dir / name).is_file()]
    installed_from_project: list[Path] = []
    # validate the input files
    for f in files:
        if f.name == PYPROJECT_FILE:
            try:
                parsed = toml.load(f)
            except toml.TomlDecodeError as exc:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION, f'{f.name} is not valid TOML: {exc}'
                ) from exc
            sources = parsed.get('tool', {}).get('uv', {}).get('sources', {})
            for name, source in sources.items():
                # uv takes a table, or a list of them where it picks one by marker or extra
                entries = source if isinstance(source, list) else [source]
                if not any(isinstance(e, dict) and ('path' in e or 'workspace' in e) for e in entries):
                    continue
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'dependency {name!r} is declared as a local source in {f.name}, which cannot be '
                    'installed in a hosted image; publish it to an index and depend on the published version',
                )
            for requirement in _declared_dependencies(parsed):
                target = _direct_reference(requirement)
                # the image build reaches a dependency over a url, or installs it from an index
                if target is None or ('://' in target and not target.startswith('file:')):
                    continue
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'{f.name} declares {requirement!r}, which names a source on this machine; instead, '
                    'publish the package to an index and depend on the published version, so that it can '
                    'get picked up by the hosted image build',
                )
            for location in _local_index_locations(parsed):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'{f.name} looks for packages in {location}, a location on this machine; instead, '
                    'publish the packages to an index and depend on the published versions, so that they '
                    'can get picked up by the hosted image build',
                )
            continue
        if f.name == 'uv.lock':
            try:
                parsed = toml.load(f)
            except toml.TomlDecodeError as exc:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION, f'{f.name} is not valid TOML: {exc}'
                ) from exc
            local = _local_lock_sources(parsed, project_dir)
            if len(local) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'{f.name} installs {"; ".join(local)} from a path rather than an index; instead, '
                    'publish the package to an index and depend on the published version, so that it can '
                    'get picked up by the hosted image build',
                )
            continue
        if f.name == 'requirements.txt':
            # pip runs in the context, so a requirement naming a path needs that file alongside the manifests
            installed_from_project.extend(_local_requirement_files(project_dir, f))

    files.extend(installed_from_project)

    fd, name = tempfile.mkstemp(suffix='.tar', prefix='pxt_image_')
    os.close(fd)
    context_path = Path(name)
    hashes: dict[str, str] = {}
    with tarfile.open(context_path, 'w') as tf:
        for f in files:
            relpath = f.relative_to(project_dir).as_posix()
            hashes[relpath] = _add_hashed(tf, f, relpath)
    _logger.info(f'Image context created: {context_path}')
    return PackagedContext(path=context_path, files=hashes)


def archive_object_name(org_id: str, archive_digest: str) -> str:
    """The object name of the project archive with this digest."""
    return f'archives/{org_id}/{archive_digest}.tar.bz2'


def image_object_name(org_id: str, image_digest: str) -> str:
    """The object name of the image context with this digest."""
    return f'images/{org_id}/{image_digest}/context.tar'


class ProjectPart(enum.StrEnum):
    """The parts that make up a project's fingerprint."""

    IMAGE = 'image'

    ARCHIVE = 'archive'

    BINDINGS = 'bindings'


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

    # path -> sha256 for the files in requirements.txt that install from a path in the project itself; separate
    # from files, which are affected by DatabaseConfig.exclude
    installed_from_project: dict[str, str] = {}

    # bindings, never resolved values: a var names the source of its value
    vars: dict[str, str]

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
        if self.vars != other.vars:
            parts.add(ProjectPart.BINDINGS)
        return parts

    def image_digest(self) -> str:
        """The image identity."""
        return _digest(self._image_inputs())

    def archive_digest(self) -> str:
        """The archive identity."""
        return _digest(self.files)

    def changes(self, other: ProjectFingerprint, parts: set[ProjectPart] | None = None) -> list[str]:
        """What differs from other in the given parts, one printable line each; defaults to every part."""
        if parts is None:
            parts = set(ProjectPart)
        lines: list[str] = []
        if ProjectPart.ARCHIVE in parts:
            lines += self._added_or_changed(other)
            lines += [f'{path} removed' for path in sorted(set(other.files) - set(self.files))]
        if ProjectPart.IMAGE in parts:
            if ProjectPart.ARCHIVE not in parts:
                # make sure to include the manifests
                lines += _changed_paths(self.image_files(), other.image_files())
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
            self.image_files(),
            self.python_version,
            self.system_dependencies,
            self.pixeltable_version,
            self.uv_options,
        )

    def image_files(self) -> dict[str, str]:
        """The manifests an image build reads, plus the project files they install from."""
        manifests = {path: content_hash for path, content_hash in self.files.items() if path in IMAGE_INPUT_FILES}
        return {**manifests, **self.installed_from_project}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _changed_paths(now: dict[str, str], was: dict[str, str]) -> list[str]:
    """One line per path both hold with different contents."""
    return [f'{path} changed' for path in sorted(set(now) & set(was)) if now[path] != was[path]]


def _changed_keys(now: dict[str, str], was: dict[str, str]) -> list[str]:
    return sorted({name for name in set(now) | set(was) if now.get(name) != was.get(name)})


def unpacked_digest(project_dir: Path) -> str:
    """The archive digest of every file under project_dir, as ProjectFingerprint.archive_digest() computes it."""
    files = {
        p.relative_to(project_dir).as_posix(): _path_hash(p)
        # a symlink counts even where its target is missing: is_file() follows the link, and dropping it
        # would make a correctly unpacked project look like a different one
        for p in project_dir.rglob('*')
        if p.is_symlink() or p.is_file()
    }
    return _digest(files)


def project_fingerprint(project_root: Path, config: DatabaseConfig | None) -> ProjectFingerprint:
    """Fingerprint every file an image built from project_root would hold.

    This decides whether an image is out of date, since an image holds the whole project.
    """
    return _fingerprint(_archive_files(project_root, config), project_root, config)


# Where this interpreter keeps the standard library and installed packages.
_ENV_DIRS = tuple(
    Path(sysconfig.get_paths()[name]).resolve()
    for name in ('stdlib', 'purelib', 'platlib')
    if name in sysconfig.get_paths()
)


def in_environment(path: Path) -> bool:
    """Returns True if path is a file of this interpreter's standard library or installed packages."""
    return any(path.is_relative_to(env_dir) for env_dir in _ENV_DIRS)


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        # chunked reads: limit buffering
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(files: Iterable[Path], project_root: Path, config: DatabaseConfig | None) -> ProjectFingerprint:
    requirements = project_root / 'requirements.txt'
    local_requirements = _local_requirement_files(project_root, requirements) if requirements.is_file() else []
    from_project = {p.relative_to(project_root).as_posix(): _path_hash(p) for p in local_requirements}
    files = {path.relative_to(project_root).as_posix(): _path_hash(path) for path in files}
    declared_python = config.python_version if config is not None else None
    return ProjectFingerprint(
        files=files,
        installed_from_project=from_project,
        # the version an image would use: the entry's, or the running interpreter's
        python_version=declared_python or f'{sys.version_info.major}.{sys.version_info.minor}',
        system_dependencies=(config.system_dependencies if config is not None else None) or [],
        pixeltable_version=pixeltable.__version__,
        uv_options=config.uv_options if config is not None else None,
        vars=(config.vars if config is not None else None) or {},
    )
