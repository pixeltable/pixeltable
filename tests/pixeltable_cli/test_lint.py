"""Tests for 'pxt lint', with a script in place of a coding agent as the reviewer."""

import json
import os
import pathlib
import shlex
import shutil
import stat
import subprocess
import sys

import pytest

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import lint

from .conftest import PxtRunner

pytestmark = pytest.mark.db_roots('local', reason='reviews files on the filesystem, independent of any catalog')

_APP = 'import pixeltable as pxt\n\nTableModel = pxt.model_base()\n'
_REPORT = 'import app\n\nrows = app.Signups.collect()\nseen = set()\n'
_CLAUDE = 'claude -p --tools Read,Grep,Glob --allowedTools Read,Grep,Glob'


def clear_lint_env(monkeypatch: pytest.MonkeyPatch, home: pathlib.Path) -> None:
    """Clear the environment variables that select a reviewer, and use home as PIXELTABLE_HOME."""
    for name in [*utils.AGENT_MARKERS, 'CURSOR_EXTENSION_HOST_ROLE', 'LINT_REVIEWER', 'PIXELTABLE_CONFIG']:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PIXELTABLE_HOME', str(home))


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A project with an application file, a script in a subdirectory, and skipped directories."""
    (tmp_path / 'home').mkdir()
    clear_lint_env(monkeypatch, tmp_path / 'home')
    (tmp_path / 'pixeltable.toml').write_text('[[pixeltable.database]]\n', encoding='utf-8')
    (tmp_path / 'app.py').write_text(_APP, encoding='utf-8')
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports' / 'report.py').write_text(_REPORT, encoding='utf-8')
    for skipped in ('.venv', '__pycache__', 'node_modules'):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / 'skipped.py').write_text('x = 1\n', encoding='utf-8')
    return tmp_path


def reviewer(tmp_path: pathlib.Path, body: str) -> str:
    """Return the command of a reviewer script; the script saves its prompt in prompt.txt, then runs body."""
    script = tmp_path / f'reviewer_{len(list(tmp_path.glob("reviewer_*")))}.py'
    script.write_text(
        'import pathlib, sys, time\n'
        f'pathlib.Path({str(tmp_path / "prompt.txt")!r}).write_text(sys.stdin.read())\n' + body,
        encoding='utf-8',
    )
    return f'{shlex.quote(sys.executable)} {shlex.quote(str(script))}'


def answer(*findings: dict[str, object]) -> str:
    """Return a reviewer body that prints findings as JSON, wrapped in prose with braces and a code fence."""
    return (
        'print("I checked {app.py} first.")\nprint("```json")\n'
        f'print({json.dumps({"findings": list(findings)})!r})\nprint("```")\n'
    )


def pixeltable_reviewer(project: pathlib.Path) -> str | None:
    """Return the [lint] reviewer that Pixeltable's own Config reads from project's configuration."""
    # Pixeltable refuses a configuration file with an option it does not know
    loaded = subprocess.run(
        [
            sys.executable,
            '-c',
            'import pathlib, sys; from pixeltable.config import Config; '
            'config = Config({}, project_root=pathlib.Path(sys.argv[1])); '
            'print(config.get_string_value("reviewer", section="lint"))',
            str(project),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    reviewer = loaded.stdout.splitlines()[-1]
    return None if reviewer == 'None' else reviewer


def search_path(*dirs: pathlib.Path) -> str:
    """Return a PATH with dirs, the directory of pxt, and the system directories."""
    pxt = shutil.which('pxt')
    assert pxt is not None
    return os.pathsep.join([*map(str, dirs), str(pathlib.Path(pxt).parent), '/usr/bin', '/bin'])


def fake_claude(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return a directory with a claude that answers every review with no findings."""
    bin_dir = tmp_path_factory.mktemp('bin')
    (bin_dir / 'claude').write_text(
        f"#!/bin/sh\ncat > /dev/null\necho '{json.dumps({'findings': []})}'\n", encoding='utf-8'
    )
    (bin_dir / 'claude').chmod(stat.S_IRWXU)
    return bin_dir


_FINDING = {'rule': 'R05', 'file': 'reports/report.py', 'line': 3, 'why': 'loops over rows', 'fix': 'use group_by()'}


class TestLint:
    def test_checklist(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """Without a reviewer, the command prints the project's Python files and the rules, and exits 0; named paths
        restrict the files, and LINT_REVIEWER=none prints the checklist even with a reviewer in the project
        configuration."""
        rules = cli('guide', 'review').stdout
        out = cli('lint', cwd=project).stdout
        assert 'Files:\n  app.py\n  reports/report.py\n\n' in out
        assert 'skipped.py' not in out
        assert out.endswith(rules)

        assert cli('lint', 'reports', '--json', cwd=project).json == {
            'reviewer': None,
            'files': ['reports/report.py'],
            'rules': rules,
        }

        with open(project / 'pixeltable.toml', 'a', encoding='utf-8') as fp:
            fp.write(f'\n[lint]\nreviewer = {json.dumps(reviewer(project, "sys.exit(9)"))}\n')
        out = cli('lint', 'app.py', cwd=project, env_overrides={'LINT_REVIEWER': 'none'}).stdout
        assert 'Files:\n  app.py\n\n' in out

    def test_review(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A reviewer gets the rules and the numbered lines of each file, and the command prints its findings with the
        rule's title and guide, and exits 2; with no findings, it exits 0."""
        r = cli('lint', cwd=project, check=False, env_overrides={'LINT_REVIEWER': reviewer(project, answer(_FINDING))})
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

        two_findings = reviewer(project, answer(_FINDING, {**_FINDING, 'line': 'n/a'}))
        r = cli('lint', '--json', cwd=project, check=False, env_overrides={'LINT_REVIEWER': two_findings})
        assert r.returncode == 2
        assert r.json == {'findings': [_FINDING, {**_FINDING, 'line': None}]}

        r = cli('lint', cwd=project, env_overrides={'LINT_REVIEWER': reviewer(project, answer())})
        assert r.stdout == 'No findings.\n'

    def test_reviewer_precedence(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """The environment variable comes first, then the project configuration and the home configuration."""
        in_home, in_project, from_env = (
            reviewer(project, answer({**_FINDING, 'rule': rule})) for rule in ('R04', 'R01', 'R02')
        )

        def reported_rule(env_overrides: dict[str, str | None] | None = None) -> str:
            r = cli('lint', '--json', cwd=project, check=False, env_overrides=env_overrides)
            return r.json['findings'][0]['rule']

        (project / 'home' / 'config.toml').write_text(f'[lint]\nreviewer = {json.dumps(in_home)}\n', encoding='utf-8')
        assert reported_rule() == 'R04'
        with open(project / 'pixeltable.toml', 'a', encoding='utf-8') as fp:
            fp.write(f'\n[lint]\nreviewer = {json.dumps(in_project)}\n')
        assert reported_rule() == 'R01'
        assert reported_rule({'LINT_REVIEWER': from_env}) == 'R02'

    def test_reviewer_failures(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A reviewer that fails, answers without the JSON or is missing is an error, exit 1."""
        cases = [
            (reviewer(project, 'sys.stderr.write("no credits")\nsys.exit(3)'), 'the reviewer exited 3: no credits'),
            (reviewer(project, 'print("Looks fine to me.")'), 'has no {"findings": [...]} object: Looks fine to me.'),
            (reviewer(project, 'print(\'{"findings": [{"rule": "R05"}]}\')'), 'has no {"findings": [...]} object'),
            ('no-such-reviewer --flag', 'the reviewer command was not found'),
        ]
        for command, message in cases:
            r = cli('lint', cwd=project, check=False, env_overrides={'LINT_REVIEWER': command})
            assert r.returncode == 1
            assert message in r.stderr

    def test_reviewer_timeout(
        self, project: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reviewer that does not answer within the timeout is an error, exit 1."""
        # in-process: only a patched REVIEWER_TIMEOUT_SECS shortens the 600 s wait
        monkeypatch.chdir(project)
        monkeypatch.setattr(lint, 'REVIEWER_TIMEOUT_SECS', 1)
        monkeypatch.setenv('LINT_REVIEWER', reviewer(project, 'time.sleep(10)'))
        with pytest.raises(SystemExit) as exc:
            lint.run([])
        assert exc.value.code == 1
        assert 'did not answer within 1 s' in capsys.readouterr().err

    def test_agent_reviewer(
        self, cli: PxtRunner, project: pathlib.Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Without a configured reviewer, a run under Claude Code reviews with claude and writes it to the project
        configuration; outside a coding agent, or under one whose CLI is not on PATH, it prints the checklist."""
        config_before = (project / 'pixeltable.toml').read_text(encoding='utf-8')
        for agent_env in ({}, {'CODEX_SANDBOX': '1'}, {'CLAUDECODE': '1'}):
            r = cli('lint', cwd=project, env_overrides={'PATH': search_path(), **agent_env})
            assert r.stdout.startswith('No reviewer ran')
        assert (project / 'pixeltable.toml').read_text(encoding='utf-8') == config_before

        with_claude: dict[str, str | None] = {'CLAUDECODE': '1', 'PATH': search_path(fake_claude(tmp_path_factory))}
        r = cli('lint', cwd=project, env_overrides=with_claude)
        assert r.stdout == 'No findings.\n'
        assert f'reviewing with {_CLAUDE} ([lint] reviewer in pixeltable.toml)' in r.stderr
        assert pixeltable_reviewer(project) == _CLAUDE

        # the second run uses the configured reviewer
        r = cli('lint', cwd=project, env_overrides=with_claude)
        assert (r.stdout, r.stderr) == ('No findings.\n', '')

        (project / 'pixeltable.toml').write_text(config_before + '\n[lint]\nreviewer = "none"\n', encoding='utf-8')
        assert cli('lint', cwd=project, env_overrides=with_claude).stdout.startswith('No reviewer ran')

    def test_agent_reviewer_config_files(
        self, cli: PxtRunner, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pyproject.toml with a [tool.pixeltable.pixeltable] table gets [tool.pixeltable.lint]. One with the settings
        of [pixeltable] directly under [tool.pixeltable], as 'pxt init' writes it, has no place for [lint], and neither
        has a directory outside a project: there the review runs without writing a configuration."""
        clear_lint_env(monkeypatch, tmp_path_factory.mktemp('home'))
        with_claude: dict[str, str | None] = {'CLAUDECODE': '1', 'PATH': search_path(fake_claude(tmp_path_factory))}
        this_run_only = (
            f'pxt lint: reviewing with {_CLAUDE} (this run only). Each review runs that model on your account; '
            '[lint] reviewer = "none" in the project configuration turns it off.\n'
        )

        nested = tmp_path_factory.mktemp('nested')
        (nested / 'pyproject.toml').write_text(
            '[project]\nname = "shop"\n\n[[tool.pixeltable.pixeltable.database]]\n', encoding='utf-8'
        )
        (nested / 'app.py').write_text(_APP, encoding='utf-8')
        assert '([lint] reviewer in pyproject.toml)' in cli('lint', cwd=nested, env_overrides=with_claude).stderr
        assert pixeltable_reviewer(nested) == _CLAUDE

        for flat_layout in ('[[tool.pixeltable.database]]\n', None):
            project = tmp_path_factory.mktemp('flat')
            if flat_layout is not None:
                (project / 'pyproject.toml').write_text(f'[project]\nname = "shop"\n\n{flat_layout}', encoding='utf-8')
            (project / 'script.py').write_text(_APP, encoding='utf-8')
            files_before = {path.name: path.read_text(encoding='utf-8') for path in project.iterdir()}
            r = cli('lint', cwd=project, env_overrides=with_claude)
            assert (r.stdout, r.stderr) == ('No findings.\n', this_run_only)
            assert {path.name: path.read_text(encoding='utf-8') for path in project.iterdir()} == files_before
            if flat_layout is not None:
                assert pixeltable_reviewer(project) is None

    def test_usage_errors(self, cli: PxtRunner, project: pathlib.Path) -> None:
        """A missing path, no Python files, or more source than a review takes is an error, exit 1."""
        r = cli('lint', 'missing.py', cwd=project, check=False)
        assert r.returncode == 1
        assert 'missing.py does not exist' in r.stderr

        (project / 'docs').mkdir()
        r = cli('lint', 'docs', cwd=project, check=False)
        assert r.returncode == 1
        assert 'no Python files to review' in r.stderr

        (project / 'big.py').write_text('x = 1\n' * 70_000, encoding='utf-8')
        r = cli('lint', cwd=project, check=False)
        assert r.returncode == 1
        assert 'more than 400000 characters; explicitly list the files to review' in r.stderr
