"""`pxt skills install` - copy the Pixeltable skill for coding agents into the current directory."""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import shutil
import sys
import tarfile
import uuid
from typing import IO, NoReturn

from ..parser import Parser
from ..utils import EXIT_ERROR, EXIT_REFUSED, confirm_or_exit

ARCHIVE_URL = 'https://codeload.github.com/pixeltable/pixeltable-skill/tar.gz/refs/heads/main'

# the skill's directory in that repository
_SKILL_PARTS = ('skills', 'pixeltable-skill')

# the skill's name in its frontmatter; the Agent Skills format expects its directory to have the same name
SKILL_NAME = 'pixeltable'

# the project directories coding agents read skills from, and the agents that read each
SKILL_DIRS = (
    ('.claude/skills', 'Claude Code'),
    ('.agents/skills', 'Codex, Cursor, Gemini CLI, GitHub Copilot, OpenCode'),
)

_DOWNLOAD_TIMEOUT_SECS = 30

EPILOG = """\
Examples:
  pxt skills install                # in the directory you start your coding agent from
  pxt skills install -f             # replace a copy that differs, e.g. to update it
  pxt skills install --json         # the same, machine-readable

What it writes:
  The Pixeltable skill, SKILL.md and its references, from github.com/pixeltable/pixeltable-skill,
  into the two project directories coding agents read skills from:

    .claude/skills/pixeltable       Claude Code
    .agents/skills/pixeltable       Codex, Cursor, Gemini CLI, GitHub Copilot, OpenCode

  Start a new agent session to load it. A copy that already matches is left as it is; one that differs
  is replaced only after a yes at the prompt, or with -f.

Exit codes:
  0  the skill is installed
  1  error: the skill could not be downloaded or written
  3  refused: a different copy is there, and -f was not given"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt skills', description='install the Pixeltable skill for coding agents', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

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
    differing = [t for t in targets if _exists(t) and _read_tree(t) != files]
    if len(differing) > 0:
        names = ' and '.join(str(t.relative_to(root)) for t in differing)
        confirm_or_exit(f'replace the copy of the skill in {names}?', args.force, refused_exit_code=EXIT_REFUSED)

    written: list[pathlib.Path] = []
    for target in targets:
        if _exists(target) and target not in differing:
            continue
        _write_tree(target, files)
        written.append(target)

    if args.as_json:
        print(
            json.dumps(
                {
                    'source': ARCHIVE_URL,
                    'version': version,
                    'commit': commit,
                    'installed': [str(t.relative_to(root)) for t in targets],
                    'written': [str(t.relative_to(root)) for t in written],
                },
                indent=2,
            )
        )
        return
    source = f'version {version}' if version is not None else 'unknown version'
    if commit is not None:
        source += f', commit {commit[:9]}'
    print(f'Pixeltable skill ({source}):')
    for target, (_, readers) in zip(targets, SKILL_DIRS):
        state = 'written' if target in written else 'already current'
        print(f'  {target.relative_to(root)}  {state:<15}  {readers}')
    if len(written) > 0:
        print('Start a new agent session to load it.')


def _download() -> tuple[dict[str, bytes], str | None, str | None]:
    """The skill's files by path within the skill, the commit they come from, and the skill's version."""
    import requests

    try:
        resp = requests.get(ARCHIVE_URL, timeout=_DOWNLOAD_TIMEOUT_SECS)
        resp.raise_for_status()
    except requests.RequestException as e:
        _fail(f'could not download the skill from {ARCHIVE_URL}: {e}')
    try:
        return unpack(resp.content)
    except (tarfile.TarError, OSError, ValueError) as e:
        _fail(f'the archive from {ARCHIVE_URL} does not hold the skill: {e}')


def unpack(archive: bytes) -> tuple[dict[str, bytes], str | None, str | None]:
    """Read the skill out of a tar.gz of the skill repository.

    Only regular files under the skill's directory are taken, so a link or a path that leaves the directory
    cannot place anything outside the target. The commit comes from the pax header that GitHub's archives
    carry, and the version from the plugin manifest at the repository root.
    """
    files: dict[str, bytes] = {}
    version: str | None = None
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
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
    """The files under root by relative path, or None if root is not a directory."""
    if root.is_symlink() or not root.is_dir():
        return None
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob('*'))
        if p.is_file() and not p.is_symlink()
    }


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
    if old.is_symlink() or old.is_file():
        old.unlink()
    elif old.is_dir():
        shutil.rmtree(old, ignore_errors=True)


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
