"""Tests for 'pxt lint', with a script in place of a coding agent as the reviewer."""

import contextlib
import json
import os
import pathlib
import shlex
import shutil
import stat
import sys

import psutil
import pytest

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import lint

from .conftest import PxtRunner

pytestmark = pytest.mark.db_roots('local', reason='reviews files on the filesystem, independent of any catalog')

_APP = 'import pixeltable as pxt\n\nTableModel = pxt.model_base()\n'
_REPORT = 'import app\n\nrows = app.Signups.collect()\nseen = set()\n'
_CLAUDE = 'claude -p --tools Read,Grep,Glob --allowedTools Read,Grep,Glob --strict-mcp-config --setting-sources user'


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A project with an application file, a script in a subdirectory, and skipped directories."""
    for name in [*utils.AGENT_MARKERS, 'CURSOR_EXTENSION_HOST_ROLE', 'PXT_LINT_REVIEWER']:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / 'pixeltable.toml').write_text('[[pixeltable.database]]\n', encoding='utf-8')
    (tmp_path / 'app.py').write_text(_APP, encoding='utf-8')
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports' / 'report.py').write_text(_REPORT, encoding='utf-8')
    for skipped in ('.venv', '__pycache__', 'node_modules'):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / 'skipped.py').write_text('x = 1\n', encoding='utf-8')
    return tmp_path


def reviewer(tmp_path: pathlib.Path, body: str) -> str:
    """Return the command of a reviewer script; the script saves its prompt in prompt.txt, then runs body. Like an
    agent's CLI, it reads and writes UTF-8."""
    script = tmp_path / f'reviewer_{len(list(tmp_path.glob("reviewer_*")))}.py'
    script.write_text(
        'import pathlib, sys, time\n'
        "sys.stdin.reconfigure(encoding='utf-8')\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        f"pathlib.Path({str(tmp_path / 'prompt.txt')!r}).write_text(sys.stdin.read(), encoding='utf-8')\n" + body,
        encoding='utf-8',
    )
    return f'{shlex.quote(sys.executable)} {shlex.quote(str(script))}'


def answer(*findings: dict[str, object]) -> str:
    """Return a reviewer body that prints findings as JSON, wrapped in prose with braces and a code fence."""
    return (
        'print("I checked {app.py} first.")\nprint("```json")\n'
        f'print({json.dumps({"findings": list(findings)}, ensure_ascii=False)!r})\nprint("```")\n'
    )


def search_path(*dirs: pathlib.Path) -> str:
    """Return a PATH with dirs, the directory of pxt, and the system directories."""
    pxt = shutil.which('pxt')
    assert pxt is not None
    system_dirs = (
        [os.path.join(os.environ['SYSTEMROOT'], 'System32')] if sys.platform == 'win32' else ['/usr/bin', '/bin']
    )
    return os.pathsep.join([*map(str, dirs), str(pathlib.Path(pxt).parent), *system_dirs])


def fake_claude(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return a directory with a claude that answers every review with no findings; on Windows, a claude.cmd, like an
    npm install of Claude Code."""
    bin_dir = tmp_path_factory.mktemp('bin')
    if sys.platform == 'win32':
        script = bin_dir / 'fake_claude.py'
        script.write_text(
            f'import sys\nsys.stdin.buffer.read()\nprint({json.dumps({"findings": []})!r})\n', encoding='utf-8'
        )
        (bin_dir / 'claude.cmd').write_text(f'@"{sys.executable}" "{script}" %*\n', encoding='utf-8')
        return bin_dir
    (bin_dir / 'claude').write_text(
        f"#!/bin/sh\ncat > /dev/null\necho '{json.dumps({'findings': []})}'\n", encoding='utf-8'
    )
    (bin_dir / 'claude').chmod(stat.S_IRWXU)
    return bin_dir


_FINDING = {'rule': 'R05', 'file': 'reports/report.py', 'line': 3, 'why': 'loops over rows', 'fix': 'use group_by()'}


class TestLint:
    def test_checklist(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """Without a reviewer, the command prints the project's Python files and the rules, and exits 0; named paths
        restrict the files; PXT_LINT_REVIEWER=none also prints the checklist."""
        rules = cli('guide', 'review').stdout
        out = cli('lint', cwd=project).stdout
        assert 'Files:\n  app.py\n  reports/report.py\n\n' in out
        assert 'skipped.py' not in out
        assert out.endswith(rules)

        assert cli('lint', 'reports', '--json', cwd=project).json == {
            'reviewer': None,
            'findings': None,
            'files': ['reports/report.py'],
            'rules': rules,
        }

        out = cli('lint', 'app.py', cwd=project, env_overrides={'PXT_LINT_REVIEWER': 'none'}).stdout
        assert 'Files:\n  app.py\n\n' in out

    def test_review(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A reviewer gets the rules and the numbered lines of each file, and the command prints its findings with the
        rule's title and guide, and exits 2. A non-integer line becomes null, a finding without a rule, file, why or
        fix string is skipped with a notice, and an answer that repeats the prompt's example first counts only its
        last findings object. With no findings, the command exits 0."""
        r = cli(
            'lint', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': reviewer(project, answer(_FINDING))}
        )
        assert r.returncode == 2
        prompt = (project / 'prompt.txt').read_text(encoding='utf-8')
        assert cli('guide', 'review').stdout in prompt
        assert (
            '<file path="reports/report.py">\n    1  import app\n    2  \n    3  rows = app.Signups.collect()' in prompt
        )
        assert r.stdout == (
            'reports/report.py:3: R05 A value from other rows computed in a Python loop\n'
            '    Why: loops over rows\n'
            '    Fix: use group_by()\n'
            '    Guide: pxt guide queries\n'
            '1 finding: fix each, or say why it does not apply.\n'
        )

        bad_lines = reviewer(project, answer(_FINDING, {**_FINDING, 'line': 'n/a'}))
        r = cli('lint', '--json', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': bad_lines})
        assert r.returncode == 2
        assert r.json == {'findings': [_FINDING, {**_FINDING, 'line': None}]}

        malformed = reviewer(project, answer({'rule': 'R05'}, _FINDING, {**_FINDING, 'fix': None}))
        r = cli('lint', '--json', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': malformed})
        assert (r.returncode, r.json) == (2, {'findings': [_FINDING]})
        assert r.stderr == "pxt lint: skipped 2 malformed findings in the reviewer's answer.\n"

        example = {'rule': 'R05', 'file': 'report.py', 'line': 14, 'why': '...', 'fix': '...'}
        echoing = reviewer(project, f'print({json.dumps({"findings": [example]})!r})\n' + answer(_FINDING))
        r = cli('lint', '--json', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': echoing})
        assert r.json == {'findings': [_FINDING]}

        r = cli('lint', cwd=project, env_overrides={'PXT_LINT_REVIEWER': reviewer(project, answer())})
        assert r.stdout == 'No findings.\n'

    def test_non_ascii(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """Characters outside the locale's code page reach the reviewer in the source and come back in its --json
        findings, and the text output prints findings that contain them."""
        (project / 'labels.py').write_text('NEXT = "\u2192"\n', encoding='utf-8')
        # UTF-8 encodes U+201D with the byte 0x9D, which cp1252 cannot decode
        finding = {**_FINDING, 'why': 'loops over rows \u201cone by one\u201d \u2192 slow'}
        command = reviewer(project, answer(finding))
        r = cli('lint', '--json', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': command})
        assert (r.returncode, r.json) == (2, {'findings': [finding]})
        assert '<file path="labels.py">\n    1  NEXT = "\u2192"\n' in (project / 'prompt.txt').read_text(
            encoding='utf-8'
        )

        r = cli('lint', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': command})
        assert r.returncode == 2
        assert r.stdout.endswith('1 finding: fix each, or say why it does not apply.\n')

    def test_project_config_reviewer(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A reviewer in the project configuration never runs: only PXT_LINT_REVIEWER sets one."""
        with open(project / 'pixeltable.toml', 'a', encoding='utf-8') as fp:
            fp.write(f'\n[lint]\nreviewer = {json.dumps(reviewer(project, answer(_FINDING)))}\n')
        assert cli('lint', cwd=project).stdout.startswith('No reviewer ran')
        assert not (project / 'prompt.txt').exists()

    def test_reviewer_failures(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A reviewer that fails, answers without the JSON, is missing, cannot be parsed or cannot run is an error,
        exit 1; a blank PXT_LINT_REVIEWER counts as unset."""
        not_executable = project / 'reviewer'
        not_executable.write_text('#!/bin/sh\n', encoding='utf-8')
        cases = [
            (reviewer(project, 'sys.stderr.write("no credits")\nsys.exit(3)'), 'the reviewer exited 3: no credits'),
            (
                reviewer(project, 'print("Looks fine to me.")'),
                'has no valid {"findings": [...]} object: Looks fine to me.',
            ),
            (
                reviewer(project, 'print(\'{"findings": [{"rule": "R05"}]}\')'),
                'has no valid {"findings": [...]} object',
            ),
            ('no-such-reviewer --flag', 'the reviewer command was not found'),
            ('claude "-p', 'the reviewer command cannot be parsed (No closing quotation)'),
        ]
        # Windows has no execute permission
        if sys.platform != 'win32':
            cases.append((str(not_executable), 'the reviewer command cannot run (Permission denied)'))
        for command, message in cases:
            r = cli('lint', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': command})
            assert r.returncode == 1
            assert message in r.stderr
        assert cli('lint', cwd=project, env_overrides={'PXT_LINT_REVIEWER': '  '}).stdout.startswith('No reviewer ran')

    def test_reviewer_timeout(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reviewer that does not answer within the timeout is an error, exit 1, and its child processes are
        stopped."""
        # in-process: only a patched REVIEWER_TIMEOUT_SECS shortens the wait
        monkeypatch.chdir(project)
        monkeypatch.setattr(lint, 'REVIEWER_TIMEOUT_SECS', 3)
        child_pid = project / 'child.pid'
        body = (
            'import subprocess\n'
            'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
            f'pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n'
            'time.sleep(60)\n'
        )
        monkeypatch.setenv('PXT_LINT_REVIEWER', reviewer(project, body))
        with pytest.raises(SystemExit) as exc:
            lint.run([])
        assert exc.value.code == 1
        assert 'did not answer within 3 s' in capsys.readouterr().err
        with contextlib.suppress(psutil.NoSuchProcess):
            psutil.Process(int(child_pid.read_text())).wait(timeout=10)

    def test_agent_reviewer(
        self, cli: PxtRunner, project: pathlib.Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Without PXT_LINT_REVIEWER, a run under Claude Code reviews with claude and says so on stderr. Outside a
        coding agent, under one without a preset, or under one whose CLI is not on PATH, it prints the checklist and
        gives the cause on stderr. No run writes a file, and PXT_LINT_REVIEWER=none turns the review off."""
        files_before = {path.name: path.read_text(encoding='utf-8') for path in project.iterdir() if path.is_file()}
        causes: list[tuple[dict[str, str | None], str]] = [
            ({}, 'no coding agent runs pxt lint'),
            ({'ANTIGRAVITY_AGENT': '1'}, 'pxt lint has no reviewer command for antigravity'),
            ({'CODEX_SANDBOX': '1'}, 'codex is not on PATH'),
            ({'CLAUDECODE': '1'}, 'claude is not on PATH'),
        ]
        for agent_env, cause in causes:
            r = cli('lint', cwd=project, env_overrides={'PATH': search_path(), **agent_env})
            assert r.stdout.startswith('No reviewer ran')
            assert r.stderr == f'pxt lint: no reviewer: PXT_LINT_REVIEWER is not set, and {cause}.\n'

        with_claude: dict[str, str | None] = {'CLAUDECODE': '1', 'PATH': search_path(fake_claude(tmp_path_factory))}
        notice = (
            f'pxt lint: reviewing with {_CLAUDE}. A review takes several minutes: give pxt lint a timeout of 10 '
            'minutes. Each review runs that model on your account; PXT_LINT_REVIEWER=none turns it off.\n'
        )
        for _ in range(2):
            r = cli('lint', cwd=project, env_overrides=with_claude)
            assert (r.stdout, r.stderr) == ('No findings.\n', notice)
        assert {path.name: path.read_text(encoding='utf-8') for path in project.iterdir() if path.is_file()} == (
            files_before
        )

        r = cli('lint', cwd=project, env_overrides={**with_claude, 'PXT_LINT_REVIEWER': 'none'})
        assert r.stdout.startswith('No reviewer ran')

    def test_usage_errors(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A missing path, no Python files, a project configuration that does not parse, more source than a review
        takes, or a file that cannot be read is an error, exit 1; the checklist takes any amount of source."""
        r = cli('lint', 'missing.py', cwd=project, check=False)
        assert r.returncode == 1
        assert 'missing.py does not exist' in r.stderr

        (project / 'docs').mkdir()
        r = cli('lint', 'docs', cwd=project, check=False)
        assert r.returncode == 1
        assert 'no Python files to review' in r.stderr

        (project / 'broken').mkdir()
        (project / 'broken' / 'pyproject.toml').write_text('[tool.pixeltable\n', encoding='utf-8')
        r = cli('lint', '--json', cwd=project / 'broken', check=False)
        assert r.returncode == 1
        assert (r.json['error_code'], r.json['retryable']) == ('INVALID_CONFIGURATION', False)
        assert r.json['message'].startswith(f'{project / "broken" / "pyproject.toml"} cannot be parsed: ')

        (project / 'big.py').write_text('x = 1\n' * 70_000, encoding='utf-8')
        assert '  big.py\n' in cli('lint', cwd=project).stdout
        r = cli('lint', cwd=project, check=False, env_overrides={'PXT_LINT_REVIEWER': reviewer(project, answer())})
        assert r.returncode == 1
        assert 'more than 400000 characters; review fewer at a time, such as one directory' in r.stderr

        # Windows needs a privilege to create a symlink
        if sys.platform != 'win32':
            (project / 'moved').mkdir()
            (project / 'moved' / 'report.py').symlink_to(project / 'deleted.py')
            r = cli(
                'lint',
                'moved',
                cwd=project,
                check=False,
                env_overrides={'PXT_LINT_REVIEWER': reviewer(project, answer())},
            )
            assert r.returncode == 1
            assert 'cannot read deleted.py: No such file or directory' in r.stderr
