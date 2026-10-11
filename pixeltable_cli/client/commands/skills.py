"""`pxt skills install` - copy the Pixeltable skill for coding agents into the current directory."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import pathlib
import shutil
import sys
import tarfile
import uuid
import zlib
from typing import IO, NoReturn

from ..parser import Parser
from ..utils import EXIT_ERROR, EXIT_REFUSED, confirm_or_exit

ARCHIVE_URL = 'https://codeload.github.com/pixeltable/pixeltable-skill/tar.gz/refs/heads/main'

DOCS_URL = 'https://docs.pixeltable.com/platform/cli#coding-agents'

# the skill's directory in that repository
_SKILL_PARTS = ('skills', 'pixeltable-skill')

# the skill's name in its frontmatter; the Agent Skills format expects its directory to have the same name
SKILL_NAME = 'pixeltable'

# the project directories coding agents read skills from, and the agents that read each; Cursor also loads
# .claude/skills for compatibility, so it finds the skill in both
SKILL_DIRS = (
    ('.claude/skills', 'Claude Code, Cursor'),
    ('.agents/skills', 'Codex, Cursor, Gemini CLI, GitHub Copilot, OpenCode'),
)

_DOWNLOAD_TIMEOUT_SECS = 30

# far above the skill's size (about 150 KB packed), so a damaged or hostile archive cannot exhaust memory
_MAX_ARCHIVE_BYTES = 10 * 2**20
_MAX_UNPACKED_BYTES = 50 * 2**20

EPILOG = f"""\
Examples:
  pxt skills install                # at the root of the repository you start your coding agent in
  pxt skills install -f             # replace a copy that differs, e.g. to update it
  pxt skills install --json         # the same, machine-readable

What it writes:
  The Pixeltable skill, SKILL.md and its references, from github.com/pixeltable/pixeltable-skill,
  into the two project directories coding agents read skills from:

    .claude/skills/pixeltable       Claude Code, Cursor
    .agents/skills/pixeltable       Codex, Cursor, Gemini CLI, GitHub Copilot, OpenCode

  A copy that already matches is left as it is; one that differs is replaced only after a yes at the
  prompt, or with -f. How agents load it, and how to update or remove it: {DOCS_URL}

Exit codes:
  0  the skill is installed
  1  error: the skill could not be downloaded, read or written
  3  refused: a different copy is there, and -f was not given"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt skills', description='install the Pixeltable skill for coding agents', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True, metavar='install')

    p = sub.add_parser('install', help='copy the skill into this directory for coding agents', epilog=EPILOG)
    p.add_argument('-f', '--force', action='store_true', help='replace a copy that differs without asking')
    p.add_argument('--json', action='store_true', dest='as_json', help='Emit JSON output')

    args = parser.parse_args(argv)
    if args.action == 'install':
        _install(args)


def _install(args: argparse.Namespace) -> None:
    files, commit, version = _download()
    root = pathlib.Path.cwd()
    targets = [root / skills_dir / SKILL_NAME for skills_dir, _ in SKILL_DIRS]
    # before either copy is written: a linked directory on the way would put the skill outside root
    for target in targets:
        _check_parents(root, target)
    try:
        differing = [t for t in targets if _exists(t) and _read_tree(t) != files]
    except OSError as e:
        _fail(f'could not read the copy already there: {e}')
    if len(differing) > 0:
        names = ' and '.join(t.relative_to(root).as_posix() for t in differing)
        confirm_or_exit(f'replace the copy of the skill in {names}?', args.force, refused_exit_code=EXIT_REFUSED)

    states: list[str] = []
    for target in targets:
        if _exists(target) and target not in differing:
            states.append('already current')
            continue
        state = 'replaced' if _exists(target) else 'written'
        _write_tree(target, files)
        states.append(state)

    paths = [t.relative_to(root).as_posix() for t in targets]
    if args.as_json:
        print(
            json.dumps(
                {
                    'root': str(root),
                    'source': ARCHIVE_URL,
                    'version': version,
                    'commit': commit,
                    'targets': [{'path': path, 'state': state} for path, state in zip(paths, states)],
                },
                indent=2,
            )
        )
        return
    source = f'version {version}' if version is not None else 'unknown version'
    if commit is not None:
        source += f', commit {commit[:9]}'
    print(f'Pixeltable skill ({source}) in {root}:')
    for path, state, (_, readers) in zip(paths, states, SKILL_DIRS):
        print(f'  {path}  {state:<15}  {readers}')
    if any(state != 'already current' for state in states):
        print(f'How agents load it: {DOCS_URL}')


def _check_parents(root: pathlib.Path, target: pathlib.Path) -> None:
    """Fail if a directory between root and target is a link: writing through it would leave root."""
    for parent in reversed(target.relative_to(root).parents[:-1]):
        if (root / parent).is_symlink():
            _fail(
                f'{parent.as_posix()} is a link, so the skill would be written outside {root}. Make it a '
                'directory, or copy the skill there yourself.'
            )


def _download() -> tuple[dict[str, bytes], str | None, str | None]:
    """The skill's files by path within the skill, the commit they come from, and the skill's version."""
    import requests

    try:
        archive = _fetch()
    except requests.RequestException as e:
        cause = f'HTTP {e.response.status_code}' if e.response is not None else type(e).__name__
        _fail(
            f'could not download the skill from {ARCHIVE_URL} ({cause}). Behind a proxy, set HTTPS_PROXY; '
            'then run it again.'
        )
    if archive is None:
        _fail(f'the archive from {ARCHIVE_URL} is larger than {_MAX_ARCHIVE_BYTES // 2**20} MB')
    try:
        return unpack(archive)
    except (tarfile.TarError, OSError, EOFError, ValueError, zlib.error) as e:
        _fail(f'the archive from {ARCHIVE_URL} does not hold the skill: {e}')


def _fetch() -> bytes | None:
    """The archive's bytes, or None if it is larger than _MAX_ARCHIVE_BYTES."""
    import requests

    with requests.get(ARCHIVE_URL, timeout=_DOWNLOAD_TIMEOUT_SECS, stream=True) as resp:
        resp.raise_for_status()
        chunks: list[bytes] = []
        size = 0
        for chunk in resp.iter_content(chunk_size=2**16):
            size += len(chunk)
            if size > _MAX_ARCHIVE_BYTES:
                return None
            chunks.append(chunk)
    return b''.join(chunks)


def unpack(archive: bytes) -> tuple[dict[str, bytes], str | None, str | None]:
    """Read the skill out of a tar.gz of the skill repository.

    Only regular files under the skill's directory are taken, so a link or a path that leaves the directory
    cannot place anything outside the target. The commit comes from the pax header that GitHub's archives
    carry, and the version from the plugin manifest at the repository root.
    """
    # reading to the end checks the gzip CRC; tarfile alone stops reading early and can return altered files
    with gzip.GzipFile(fileobj=io.BytesIO(archive)) as gz:
        tar_bytes = gz.read(_MAX_UNPACKED_BYTES + 1)
    if len(tar_bytes) > _MAX_UNPACKED_BYTES:
        raise ValueError(f'it unpacks to more than {_MAX_UNPACKED_BYTES // 2**20} MB')
    files: dict[str, bytes] = {}
    version: str | None = None
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # every member sits under one top-level directory, e.g. 'pixeltable-skill-main/'
            parts = pathlib.PurePosixPath(member.name).parts[1:]
            if parts == ('plugin.json',):
                version = _manifest_version(tar.extractfile(member))
                continue
            if parts[: len(_SKILL_PARTS)] != _SKILL_PARTS:
                continue
            skill_parts = parts[len(_SKILL_PARTS) :]
            # '..' would climb out of the target; ':' and '\' would name a drive or a separator on Windows
            if len(skill_parts) == 0 or any(p in ('.', '..') or ':' in p or '\\' in p for p in skill_parts):
                continue
            fp = tar.extractfile(member)
            assert fp is not None
            files['/'.join(skill_parts)] = fp.read()
        commit = tar.pax_headers.get('comment')
    if 'SKILL.md' not in files:
        raise ValueError(f'no {"/".join(_SKILL_PARTS)}/SKILL.md')
    return files, commit, version


def _manifest_version(fp: IO[bytes] | None) -> str | None:
    if fp is None:
        return None
    try:
        version = json.load(fp).get('version')
    except (ValueError, AttributeError):
        return None
    return version if isinstance(version, str) else None


def _exists(path: pathlib.Path) -> bool:
    return path.exists() or path.is_symlink()


def _read_tree(root: pathlib.Path) -> dict[str, bytes] | None:
    """The files under root by relative path.

    Returns None unless root is a directory holding only directories and regular files: an installed copy
    never holds a link, so one anywhere under root makes the copy differ, and replacing it removes the link.
    """
    if root.is_symlink() or not root.is_dir():
        return None
    files: dict[str, bytes] = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            return None
        if p.is_file():
            files[p.relative_to(root).as_posix()] = p.read_bytes()
    return files


def _write_tree(target: pathlib.Path, files: dict[str, bytes]) -> None:
    """Make target hold exactly files.

    The new copy is written beside target and renamed into place, so a failure leaves the old copy, not half
    of each.
    """
    suffix = uuid.uuid4().hex[:8]
    staging = target.with_name(f'.{target.name}.new-{suffix}')
    old = target.with_name(f'.{target.name}.old-{suffix}')
    try:
        _write_files(staging, files)
        _swap(staging, target, old)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        _fail(f'could not write {target}: {e}')
    if not _exists(old):
        return
    # the new copy is in place; a previous one left beside it would be read as a second copy of the skill
    try:
        if old.is_symlink() or old.is_file():
            old.unlink()
        else:
            shutil.rmtree(old)
    except OSError as e:
        print(f'pxt skills install: could not remove the previous copy at {old}: {e}. Delete it.', file=sys.stderr)


def _write_files(root: pathlib.Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _swap(new: pathlib.Path, target: pathlib.Path, old: pathlib.Path) -> None:
    """Rename new into target's place, moving any previous copy to old; on failure, put that copy back."""
    if _exists(target):
        target.rename(old)
    try:
        new.rename(target)
    except OSError:
        if _exists(old):
            old.rename(target)
        raise


def _fail(message: str) -> NoReturn:
    print(f'pxt skills install: {message}', file=sys.stderr)
    sys.exit(EXIT_ERROR)
