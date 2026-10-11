"""`pxt skills install` - copy the Pixeltable skill for coding agents into the current directory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib
import shutil
import sys
import tarfile
import uuid
import zlib
from typing import IO, TYPE_CHECKING, NoReturn

from ..parser import Parser
from ..utils import EXIT_ERROR, EXIT_REFUSED, confirm_or_exit

if TYPE_CHECKING:
    import requests

ARCHIVE_URL = 'https://codeload.github.com/pixeltable/pixeltable-skill/tar.gz/refs/heads/main'

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

# written beside the skill's files: the commit installed and a hash of each file, so a later run can tell an
# unedited older copy, which it updates, from one edited since, which it replaces only when told to
RECORD_FILE = '.pxt-install.json'

_DOWNLOAD_TIMEOUT_SECS = 30

EPILOG = """\
Examples:
  pxt skills install                # in the directory you start your coding agent from
  pxt skills install -f             # also replace a copy edited since it was installed
  pxt skills install --json         # the same, machine-readable

What it writes:
  The Pixeltable skill, SKILL.md and its references, from github.com/pixeltable/pixeltable-skill,
  into the two project directories coding agents read skills from:

    .claude/skills/pixeltable       Claude Code, Cursor
    .agents/skills/pixeltable       Codex, Cursor, Gemini CLI, GitHub Copilot, OpenCode

  Each copy holds .pxt-install.json, the commit it came from. Running the command again updates a copy
  nobody has edited and leaves a current one as it is. A copy edited since it was installed, or one
  this command did not write, is replaced only after a yes at the prompt, or with -f.

  In Claude Code, /reload-skills loads it in a running session and /skills lists it; other agents
  load it in a new session. Commit both directories to share the skill; delete them to remove it.

Exit codes:
  0  the skill is installed
  1  error: the skill could not be downloaded or written
  3  refused: a copy was edited or written by hand, and -f was not given"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt skills', description='install the Pixeltable skill for coding agents', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True, metavar='install')

    p = sub.add_parser('install', help='copy the skill into this directory for coding agents', epilog=EPILOG)
    p.add_argument('-f', '--force', action='store_true', help='also replace a copy edited since it was installed')
    p.add_argument('--json', action='store_true', dest='as_json', help='Emit JSON output')

    args = parser.parse_args(argv)
    if args.action == 'install':
        _install(args)


def _install(args: argparse.Namespace) -> None:
    files, commit, version = _download()
    root = pathlib.Path.cwd()
    targets = [root / skills_dir / SKILL_NAME for skills_dir, _ in SKILL_DIRS]
    states = {t: _state(t, files) for t in targets}

    guarded = [t for t in targets if states[t][0] in ('edited', 'not ours')]
    if len(guarded) > 0:
        reasons = {'edited': 'edited since it was installed', 'not ours': 'not written by pxt skills install'}
        names = '; '.join(f'{t.relative_to(root).as_posix()} ({reasons[states[t][0]]})' for t in guarded)
        confirm_or_exit(f'replace the copy of the skill in {names}?', args.force, refused_exit_code=EXIT_REFUSED)

    record = json.dumps(
        {'source': ARCHIVE_URL, 'commit': commit, 'version': version, 'files': _hashes(files)}, indent=2
    ).encode()
    results: list[dict[str, str | None]] = []
    for target in targets:
        state, previous_commit = states[target]
        if state != 'current':
            _write_tree(target, {**files, RECORD_FILE: record})
        result = {'missing': 'written', 'outdated': 'updated', 'current': 'already current'}.get(state, 'replaced')
        results.append(
            {'path': target.relative_to(root).as_posix(), 'state': result, 'previous_commit': previous_commit}
        )
    _report(root, commit, version, results, as_json=args.as_json)


def _state(target: pathlib.Path, files: dict[str, bytes]) -> tuple[str, str | None]:
    """How the copy at target compares with files, and the commit it was installed from, if recorded.

    'missing', 'current' (the same files), 'outdated' (unedited since an earlier install), 'edited' (changed
    since its install), or 'not ours' (no install record, or not a plain directory).
    """
    if not _exists(target):
        return 'missing', None
    tree = _read_tree(target)
    if tree is None:
        return 'not ours', None
    raw_record = tree.pop(RECORD_FILE, None)
    if tree == files:
        return 'current', None
    try:
        record = json.loads(raw_record) if raw_record is not None else None
    except ValueError:
        record = None
    if not isinstance(record, dict) or not isinstance(record.get('files'), dict):
        return 'not ours', None
    previous_commit = record.get('commit') if isinstance(record.get('commit'), str) else None
    if _hashes(tree) == record['files']:
        return 'outdated', previous_commit
    return 'edited', previous_commit


def _report(
    root: pathlib.Path, commit: str | None, version: str | None, results: list[dict[str, str | None]], *, as_json: bool
) -> None:
    if as_json:
        print(
            json.dumps(
                {'root': str(root), 'source': ARCHIVE_URL, 'version': version, 'commit': commit, 'targets': results},
                indent=2,
            )
        )
        return
    source = f'version {version}' if version is not None else 'unknown version'
    if commit is not None:
        source += f', commit {commit[:9]}'
    print(f'Pixeltable skill ({source}) in {root}:')
    for result, (_, readers) in zip(results, SKILL_DIRS):
        state, previous_commit = result['state'], result['previous_commit']
        if previous_commit is not None and state in ('updated', 'replaced'):
            state = f'{state} from {previous_commit[:9]}'
        print(f'  {result["path"]}  {state:<24}  {readers}')
    if any(r['state'] != 'already current' for r in results):
        print(
            'In Claude Code, /reload-skills loads it in a running session and /skills lists it; other agents load '
            'it in a new session.\nCommit both directories to share the skill with collaborators.'
        )
    repo_root = _repository_root(root)
    if repo_root is not None and repo_root != root:
        print(
            f'This is below the repository root, {repo_root}. Claude Code started there loads this copy only once '
            f'it reads a file under {root.relative_to(repo_root).as_posix()}/; run pxt skills install at the root to '
            'load it at startup.'
        )


def _download() -> tuple[dict[str, bytes], str | None, str | None]:
    """The skill's files by path within the skill, the commit they come from, and the skill's version."""
    import requests

    try:
        resp = requests.get(ARCHIVE_URL, timeout=_DOWNLOAD_TIMEOUT_SECS)
        resp.raise_for_status()
    except requests.RequestException as e:
        _fail(f'could not download the skill from {ARCHIVE_URL}: {_download_failure(e)}')
    try:
        return unpack(resp.content)
    except (tarfile.TarError, OSError, EOFError, ValueError, zlib.error) as e:
        _fail(f'the archive from {ARCHIVE_URL} does not hold the skill: {e}')


def _download_failure(e: requests.RequestException) -> str:
    """The cause of a failed download, and what to do about it, in one line."""
    import requests

    if isinstance(e, requests.HTTPError) and e.response is not None:
        return f'GitHub answered HTTP {e.response.status_code}. Run it again later.'
    if isinstance(e, requests.exceptions.ProxyError):
        return 'the proxy did not accept the connection. Check the address in HTTPS_PROXY and run it again.'
    if isinstance(e, requests.exceptions.SSLError):
        cause = "the server's TLS certificate could not be verified"
    elif isinstance(e, requests.Timeout):
        cause = f'no answer within {_DOWNLOAD_TIMEOUT_SECS} s'
    else:
        cause = 'no connection'
    return f'{cause}. Check the network; behind a proxy, set HTTPS_PROXY and run it again.'


def unpack(archive: bytes) -> tuple[dict[str, bytes], str | None, str | None]:
    """Read the skill out of a tar.gz of the skill repository.

    Only regular files under the skill's directory are taken, so a link or a path that leaves the directory
    cannot place anything outside the target. The commit comes from the pax header that GitHub's archives
    carry, and the version from the plugin manifest at the repository root.
    """
    # decompressing to the end checks the gzip CRC; tarfile alone stops reading early and can return altered files
    tar_bytes = gzip.decompress(archive)
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
            # the install record is ours to write
            if skill_parts == (RECORD_FILE,):
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


def _hashes(files: dict[str, bytes]) -> dict[str, str]:
    return {rel: hashlib.sha256(content).hexdigest() for rel, content in sorted(files.items())}


def _repository_root(start: pathlib.Path) -> pathlib.Path | None:
    """The nearest directory at or above start that holds a .git entry, or None."""
    for d in (start, *start.parents):
        if (d / '.git').exists():
            return d
    return None


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
