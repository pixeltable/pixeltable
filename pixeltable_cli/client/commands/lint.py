"""`pxt lint` - review a project's Python code against the rules of the agent guide 'review'.

TODO:
- Validate each reviewer preset except claude's with the signed-in CLI on a fixture project: it reads the whole
  prompt on stdin, prints only its answer on stdout, cannot edit files or run shell commands, denies a request for
  approval instead of waiting for an answer, and exits nonzero on failure.
- cursor: pass a prompt over 131 KB in a file; "$(cat)" exceeds Linux's size limit for one argument.
- cursor: the preset needs sh, which Windows does not have.
- codex: codex reads files with shell commands; confirm that its read-only sandbox blocks their writes.
- opencode: replace the plan agent with one that also denies shell commands, if plan allows them.
- antigravity: add a preset once agy reads its prompt on stdin and has a read-only mode.
- Run test_agent_reviewer for every agent in AGENT_MARKERS, with a fake CLI for each.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, NamedTuple, TypedDict

from pixeltable_cli.utils import find_project_root

from ..parser import Parser
from ..utils import EXIT_CHANGES_PENDING, EXIT_ERROR, coding_agent, fail, plural
from .guide import guide_paths, pixeltable_dir


class ReviewerPreset(NamedTuple):
    """A coding agent's command that reads the prompt on stdin and gets only read-only tools."""

    executable: str  # the agent's CLI; pxt lint uses the preset only if it is on PATH
    command: str


# keyed by the agent names in AGENT_MARKERS
REVIEWER_PRESETS = {
    'claude': ReviewerPreset('claude', 'claude -p --tools Read,Grep,Glob --allowedTools Read,Grep,Glob'),
    'codex': ReviewerPreset('codex', 'codex exec --sandbox read-only --skip-git-repo-check -'),
    # Cursor's agent takes the prompt only as an argument
    'cursor': ReviewerPreset('agent', 'sh -c \'exec agent -p --mode ask --trust --output-format text "$(cat)"\''),
    # gemini appends the -p text to the prompt on stdin
    'gemini': ReviewerPreset('gemini', "gemini --approval-mode plan --skip-trust -p 'Follow the instructions above.'"),
    # auggie takes stdin as context for the instruction in its argument
    'augment': ReviewerPreset('auggie', "auggie --print --quiet --ask 'Follow the instructions on stdin.'"),
    'opencode': ReviewerPreset('opencode', 'opencode run --agent plan'),
}
REVIEWER_ENV_VAR = 'LINT_REVIEWER'
REVIEWER_TIMEOUT_SECS = 600

SKIPPED_DIRS = {'__pycache__', 'node_modules', 'site-packages', 'venv', 'build', 'dist'}
# the prompt contains every file and has to fit the reviewer's context window (400,000 characters: ~100,000 tokens)
MAX_SOURCE_CHARS = 400_000

EPILOG = f"""\
Examples:
  pxt lint                           # every Python file of the project
  pxt lint report.py app.py          # these files only

Exit codes:
  0  no findings, or no reviewer: the command printed the rules and the files for you to review
  1  error: bad arguments, or the reviewer failed or did not answer as asked
  2  findings: fix each one, or say why it does not apply

Notes:
  'pxt guide review' prints the rules. A reviewer is a command that reads a prompt on stdin and answers on
  stdout, such as a coding agent's non-interactive mode. ${REVIEWER_ENV_VAR} sets it. Without it, pxt lint
  reviews with the CLI of the coding agent that runs it, if pxt lint has a command for that agent
  ({', '.join(REVIEWER_PRESETS)}). ${REVIEWER_ENV_VAR}=none turns the review off: the command then prints the
  rules and the files for the coding agent that runs it.
  A review sends the files to the reviewer's model, on the reviewer's account."""

PROMPT = """\
You review the Python code of a project that uses Pixeltable {version}. Report only places that match one of the
rules below: Python code that redoes Pixeltable's work, or that uses Pixeltable wrongly. Do not review style, naming
or general Python quality.

A place is a finding only if it matches a rule's "Look for" and none of its "Not a finding" cases. Base the fix on
the rule's "Instead", applied to this code. Before you name any other construct in a fix, confirm in the
guides to this release ({guides_dir}) or in the installed package ({package_dir}) that it works; if no fix
works in this release, say so in "fix". You may read files; change nothing.

<rules>
{rules}
</rules>

{files}

Answer with one JSON object and nothing else:
{{"findings": [{{"rule": "R05", "file": "report.py", "line": 14, "why": "...", "fix": "..."}}]}}
"why" says what this code does wrong, and "fix" what to change in it, in one or two sentences each. An empty list is
a valid answer."""

CHECKLIST = """\
No reviewer ran, so review these files yourself against the rules below. For each place that matches a rule, fix
it, or say in your report why it does not apply.

Files:
{files}

{rules}"""


class Finding(TypedDict):
    rule: str
    file: str
    line: int | None
    why: str
    fix: str


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt lint',
        description="review the project's Python code against the rules in 'pxt guide review'",
        epilog=EPILOG,
        usage_exit_code=EXIT_ERROR,
    )
    parser.add_argument(
        'paths', nargs='*', metavar='PATH', help='files or directories to review (default: the project)'
    )
    parser.add_argument('--json', action='store_true', dest='as_json', help='Emit JSON output')
    args = parser.parse_args(argv)

    root = find_project_root(pathlib.Path.cwd())
    if root is None:
        root = pathlib.Path.cwd()
    for path in args.paths:
        if not os.path.exists(path):
            parser.error(f'{path} does not exist')
    files = _python_files(root, [pathlib.Path(p) for p in args.paths])
    if len(files) == 0:
        fail('pxt lint', 'there are no Python files to review', 'NO_FILES', args.as_json)
    names = [_relative_to_root(path, root) for path in files]

    rules_path = guide_paths()['review']
    configured = os.environ.get(REVIEWER_ENV_VAR, '').strip()
    reviewer = configured if configured != '' else _agent_reviewer()
    if reviewer is None or reviewer == 'none':
        _print_checklist(names, rules_path, args.as_json)
        return

    sources = {name: path.read_text(encoding='utf-8', errors='replace') for name, path in zip(names, files)}
    if sum(len(source) for source in sources.values()) > MAX_SOURCE_CHARS:
        fail(
            'pxt lint',
            f'the files have more than {MAX_SOURCE_CHARS} characters; explicitly list the files to review',
            'TOO_LARGE',
            args.as_json,
        )
    findings = _review(reviewer, sources, rules_path, root, args.as_json)
    _print_findings(findings, rules_path, args.as_json)
    if len(findings) > 0:
        sys.exit(EXIT_CHANGES_PENDING)


def _python_files(root: pathlib.Path, paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Return the named files, and the .py files under the named directories, or under root if none is named."""
    files: list[pathlib.Path] = []
    for path in paths if len(paths) > 0 else [root]:
        if path.is_file():
            files.append(path)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith('.') and d not in SKIPPED_DIRS)
            files += [pathlib.Path(dirpath) / name for name in sorted(filenames) if name.endswith('.py')]
    return list(dict.fromkeys(files))


def _relative_to_root(path: pathlib.Path, root: pathlib.Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(root.resolve())) if resolved.is_relative_to(root.resolve()) else str(path)


def _agent_reviewer() -> str | None:
    """Return the preset command of the coding agent that runs pxt lint, if it has one and its CLI is on PATH."""
    agent = coding_agent()
    preset = REVIEWER_PRESETS.get(agent) if agent is not None else None
    if preset is None or shutil.which(preset.executable) is None:
        return None
    print(
        f'pxt lint: reviewing with {preset.command}. Each review runs that model on your account; '
        f'{REVIEWER_ENV_VAR}=none turns it off.',
        file=sys.stderr,
    )
    return preset.command


def _print_findings(findings: list[Finding], rules_path: pathlib.Path, as_json: bool) -> None:
    if as_json:
        print(json.dumps({'findings': findings}, indent=2))
        return
    rules = _rule_headings(rules_path)
    for f in findings:
        title, guide = rules.get(f['rule'], ('', ''))
        location = f['file'] if f.get('line') is None else f'{f["file"]}:{f["line"]}'
        print(f'{location}: {f["rule"]} {title}'.rstrip())
        print(f'    Why: {f["why"]}')
        print(f'    Fix: {f["fix"]}')
        if guide != '':
            print(f'    Guide: {guide}')
    print(
        'No findings.'
        if len(findings) == 0
        else f'{plural(len(findings), "finding")}: fix each, or say why it does not apply.'
    )


def _print_checklist(files: list[str], rules_path: pathlib.Path, as_json: bool) -> None:
    rules = rules_path.read_text(encoding='utf-8')
    if as_json:
        print(json.dumps({'reviewer': None, 'files': files, 'rules': rules}, indent=2))
        return
    print(CHECKLIST.format(files='\n'.join(f'  {f}' for f in files), rules=rules), end='')


def _review(
    reviewer: str, sources: dict[str, str], rules_path: pathlib.Path, root: pathlib.Path, as_json: bool
) -> list[Finding]:
    """Send the rules and the files to the reviewer command, and return its findings."""
    files = '\n\n'.join(
        f'<file path="{name}">\n'
        + '\n'.join(f'{i:>5}  {line}' for i, line in enumerate(source.splitlines(), start=1))
        + '\n</file>'
        for name, source in sources.items()
    )
    prompt = PROMPT.format(
        version=importlib.metadata.version('pixeltable'),
        guides_dir=rules_path.parent,
        package_dir=pixeltable_dir(),
        rules=rules_path.read_text(encoding='utf-8'),
        files=files,
    )
    try:
        argv = shlex.split(reviewer)
    except ValueError as e:
        fail('pxt lint', f'the reviewer command cannot be parsed ({e}): {reviewer}', 'REVIEWER_FAILED', as_json)
    try:
        result = subprocess.run(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=REVIEWER_TIMEOUT_SECS,
            cwd=root,
            # opencode takes the project root from PWD, which cwd= leaves unchanged
            env={**os.environ, 'PWD': str(root)},
            check=False,
        )
    except FileNotFoundError:
        fail('pxt lint', f'the reviewer command was not found: {reviewer}', 'REVIEWER_FAILED', as_json)
    except OSError as e:
        fail('pxt lint', f'the reviewer command cannot run ({e.strerror or e}): {reviewer}', 'REVIEWER_FAILED', as_json)
    except subprocess.TimeoutExpired:
        fail(
            'pxt lint',
            f'the reviewer did not answer within {REVIEWER_TIMEOUT_SECS} s: {reviewer}',
            'REVIEWER_FAILED',
            as_json,
        )
    if result.returncode != 0:
        tail = result.stderr.strip()[-500:]
        fail(
            'pxt lint',
            f'the reviewer exited {result.returncode}: {tail if tail != "" else reviewer}',
            'REVIEWER_FAILED',
            as_json,
        )
    findings = _parse_findings(result.stdout)
    if findings is None:
        fail(
            'pxt lint',
            f'the reviewer\'s answer has no {{"findings": [...]}} object: {result.stdout.strip()[:300]}',
            'REVIEWER_FAILED',
            as_json,
        )
    return findings


def _parse_findings(answer: str) -> list[Finding] | None:
    """Parse the findings out of the reviewer's answer; None if the answer has no valid {"findings": [...]} object."""
    # a model may wrap the object in prose or a code fence, and the prose may contain braces
    decoder = json.JSONDecoder()
    parsed: Any = None
    for start in (i for i, ch in enumerate(answer) if ch == '{'):
        try:
            parsed, _ = decoder.raw_decode(answer, start)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get('findings'), list):
            break
    if not isinstance(parsed, dict) or not isinstance(parsed.get('findings'), list):
        return None
    findings: list[Finding] = []
    for f in parsed['findings']:
        if not isinstance(f, dict) or not all(isinstance(f.get(key), str) for key in ('rule', 'file', 'why', 'fix')):
            return None
        line = f.get('line')
        # bool is a subclass of int, but a JSON true is not a line number
        if not isinstance(line, int) or isinstance(line, bool):
            line = None
        findings.append(Finding(rule=f['rule'], file=f['file'], line=line, why=f['why'], fix=f['fix']))
    return findings


def _rule_headings(rules_path: pathlib.Path) -> dict[str, tuple[str, str]]:
    """Return each rule's title and the 'pxt guide' commands of its Guide line, by rule id."""
    rules: dict[str, tuple[str, str]] = {}
    for section in re.split(r'^## ', rules_path.read_text(encoding='utf-8'), flags=re.MULTILINE)[1:]:
        rule_id, _, title = section.splitlines()[0].partition(' ')
        guide = re.search(r'^Guide: (.+)$', section, flags=re.MULTILINE)
        areas = [] if guide is None else [area.strip() for area in guide.group(1).split(',')]
        rules[rule_id] = (title, ', '.join(f'pxt guide {area}' for area in areas))
    return rules
