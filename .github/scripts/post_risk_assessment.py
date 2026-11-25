"""Post the risk assessment in Claude's final message as one PR comment, and update that comment on later runs.

The final message ends with a fenced json block, as `.claude/commands/pxt-risk.md` specifies. The comment carries a
hidden marker, so each run edits the same comment instead of adding one. When the assessment did not complete or its
JSON is not valid, the comment says so, so that a verdict for an older commit never looks current. A PR whose diff
is over GitHub's size limit, and an assessment that runs out of turns or time, are too large to assess, which needs
2 humans.

Reads REPO, PR_NUMBER, HEAD_SHA, DIFF_TOO_LARGE ('true' when GitHub refused the PR's diff), RUN_URL, OUTCOME (the
Claude step's outcome), EXECUTION_FILE (the claude-code-action execution file), ASSESS_MAX_TURNS and ASSESS_MINUTES
(the Claude step's limits), and ASSESS_STARTED (when the Claude step started, in seconds since the epoch) from the
environment; `gh` authenticates with GH_TOKEN.
"""

import json
import os
import re
import subprocess
import sys
import time
from typing import Any

from post_claude_review import final_message, review_json

MARKER = '<!-- pxt-risk-assessment -->'
COMMENT_AUTHOR = 'github-actions[bot]'
# GitHub rejects comment bodies over 65536 characters.
MAX_BODY_CHARS = 60000
EVIDENCE_PATH = re.compile(r'(?!\.{1,2}(/|#|$))[\w.\-]+(/(?!\.{1,2}(/|#|$))[\w.\-]+)*(#L\d+(-L\d+)?)?', re.ASCII)
WWW = re.compile(r'www\.', re.IGNORECASE)


def validate(data: Any) -> list[str]:
    """Returns the problems that make the assessment unusable, or an empty list."""
    if data is None:
        return ['the final message has no JSON block that parses']
    if not isinstance(data, dict):
        return ['the JSON block is not an object']
    problems = []
    humans = data.get('required_humans')
    if type(humans) is not int or humans not in (1, 2):
        problems.append(f'required_humans is {humans!r}, not 1 or 2')
    if not isinstance(data.get('summary'), str) or not data['summary'].strip():
        problems.append('summary is missing')
    reasons = data.get('reasons')
    if data.get('required_humans') == 2 and not (
        isinstance(reasons, list) and any(isinstance(r, dict) and r.get('why') for r in reasons)
    ):
        problems.append('a verdict of 2 humans has no reasons')
    if (
        data.get('required_humans') == 1
        and isinstance(reasons, list)
        and any(isinstance(r, dict) and r.get('why') for r in reasons)
    ):
        problems.append('a verdict of 1 human has reasons')
    for key in ('affected', 'reasons', 'linked_prs', 'register_gaps'):
        if key in data and not isinstance(data[key], list):
            problems.append(f'{key} is not a list')
    for key in ('linked_prs', 'register_gaps'):
        if isinstance(data.get(key), list) and not all(isinstance(v, str) for v in data[key]):
            problems.append(f'{key} has an entry that is not a string')
    for key, fields in (('affected', ('component', 'effect')), ('reasons', ('invariant', 'why', 'evidence'))):
        entries = data.get(key) if isinstance(data.get(key), list) else []
        if not all(isinstance(e, dict) for e in entries):
            problems.append(f'{key} has an entry that is not an object')
        elif any(e.get(f) is not None and not isinstance(e[f], str) for e in entries for f in fields):
            problems.append(f'{key} has an entry with a field that is not a string')
    for entry in data.get('affected') if isinstance(data.get('affected'), list) else []:
        invariants = (entry.get('invariants') or []) if isinstance(entry, dict) else []
        if not isinstance(invariants, list) or not all(isinstance(i, str) for i in invariants):
            problems.append('affected has an entry whose invariants is not a list of strings')
            break
    return problems


def clean(value: Any) -> str:
    """Agent text on one line, with no marker, raw HTML, @mention, markdown link or image, or autolinked URL."""
    text = ' '.join(str(value).replace('<!--', '').replace('-->', '').split())
    text = text.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')
    text = text.replace('<', '&lt;').replace('>', '&gt;').replace('@', '@\u200b')
    return WWW.sub(lambda m: m.group(0)[:3] + '\u200b.', text.replace('://', ':\u200b//'))


def evidence_link(url: Any, repo: str, head_sha: str) -> str:
    """A link to the assessed commit's code in this repository; any other URL is dropped."""
    prefix = f'https://github.com/{repo}/blob/{head_sha}/'
    if not head_sha or not isinstance(url, str) or not url.startswith(prefix):
        return ''
    return f' ([evidence]({url}))' if EVIDENCE_PATH.fullmatch(url[len(prefix) :]) else ''


def commit_link(repo: str, sha: str) -> str:
    return f'[`{sha[:10]}`](https://github.com/{repo}/commit/{sha})' if sha else 'an unknown commit'


def render(data: dict[str, Any], repo: str, head_sha: str, run_url: str) -> str:
    humans = data['required_humans']
    verdict = '2 humans required' if humans == 2 else '1 human is enough'
    lines = [MARKER, f'## Risk assessment: {verdict}', '']
    lines += [f'Assessed {commit_link(repo, head_sha)}.', '', clean(data['summary']), '']

    reasons = [r for r in data.get('reasons') or [] if isinstance(r, dict) and r.get('why')]
    if humans == 2:
        lines += ['### Why 2 humans', '']
        for r in reasons:
            invariant = f'**{clean(r["invariant"])}**: ' if r.get('invariant') else ''
            lines.append(f'- {invariant}{clean(r["why"])}{evidence_link(r.get("evidence"), repo, head_sha)}')
        lines.append('')

    affected = [a for a in data.get('affected') or [] if isinstance(a, dict) and a.get('component')]
    if affected:
        lines += ['### Affected components', '']
        for a in affected:
            invariants = [clean(i) for i in a.get('invariants') or []]
            ids = f' ({", ".join(invariants)})' if invariants else ''
            effect = f': {clean(a["effect"])}' if a.get('effect') else ''
            lines.append(f'- **{clean(a["component"])}**{ids}{effect}')
        lines.append('')

    linked = [clean(p) for p in data.get('linked_prs') or []]
    if linked:
        lines += [f'Linked PRs: {", ".join(linked)}', '']

    gaps = [clean(g) for g in data.get('register_gaps') or []]
    gap_lines = []
    if gaps:
        gap_lines += [f'<details><summary>Risks the register does not cover yet ({len(gaps)})</summary>', '']
        gap_lines += [f'- {g}' for g in gaps]
        gap_lines += ['', '</details>', '']

    body = '\n'.join([*lines, *gap_lines, footer(run_url)])
    if len(body) <= MAX_BODY_CHARS:
        return body
    omitted = f' The {len(gaps)} risks the register does not cover yet are omitted.' if gaps else ''
    tail = f'\n\n(Truncated.{omitted} The run log has the full assessment.)\n\n{footer(run_url)}'
    return '\n'.join(lines)[: MAX_BODY_CHARS - len(tail)] + tail


def render_too_large(run_url: str, reason: str) -> str:
    return '\n'.join(
        [
            MARKER,
            '## Risk assessment: 2 humans required',
            '',
            f'{reason} A change too large to assess needs 2 humans.',
            '',
            footer(run_url),
        ]
    )


def result_subtype(execution_file: str) -> str | None:
    try:
        with open(execution_file, encoding='utf-8') as f:
            messages = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(messages, list):
        return None
    results = [m for m in messages if isinstance(m, dict) and m.get('type') == 'result']
    return results[-1].get('subtype') if results else None


def exhausted_limit(outcome: str, execution_file: str) -> str | None:
    """The limit that ended the assessment, if it ran out of turns or time; None otherwise."""
    if execution_file and os.path.isfile(execution_file) and result_subtype(execution_file) == 'error_max_turns':
        turns = os.environ.get('ASSESS_MAX_TURNS', '')
        return f'its limit of {turns} turns' if turns else 'its turn limit'
    minutes, started = os.environ.get('ASSESS_MINUTES', ''), os.environ.get('ASSESS_STARTED', '')
    timed_out = (
        outcome != 'success'
        and minutes.isdigit()
        and started.isdigit()
        and time.time() - int(started) >= int(minutes) * 60 - 30
    )
    return f'its limit of {minutes} minutes' if timed_out else None


def render_failure(repo: str, head_sha: str, run_url: str, problems: list[str]) -> str:
    detail = '; '.join(clean(p) for p in problems)
    return '\n'.join(
        [
            MARKER,
            '## Risk assessment: not available',
            '',
            f'The assessment of {commit_link(repo, head_sha)} did not complete: {detail}. '
            'An earlier verdict no longer applies.',
            '',
            footer(run_url),
        ]
    )


def footer(run_url: str) -> str:
    run = f' [Run]({run_url}).' if run_url else ''
    return (
        '<sub>Advisory: this check does not block merging. A human author counts as one of the required humans. '
        f'The assessment uses `docs/design/critical-components.md`.{run}</sub>'
    )


def gh(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['gh', *args], input=stdin, text=True, capture_output=True, check=False)


def existing_comments(repo: str, pr: str) -> list[str] | None:
    """The ids of this check's comments on the PR; None if listing the comments failed."""
    query = f'.[] | select(.user.login == "{COMMENT_AUTHOR}" and (.body | contains("{MARKER}"))) | .id'
    result = gh('api', '--paginate', f'repos/{repo}/issues/{pr}/comments', '--jq', query)
    if result.returncode != 0:
        print(f'::warning::Listing the PR comments failed: {result.stderr.strip()}')
        return None
    return result.stdout.split()


def upsert(repo: str, pr: str, body: str) -> bool:
    """Updates every comment of this check on the PR, or posts one if there is none."""
    payload = json.dumps({'body': body})
    comment_ids = existing_comments(repo, pr)
    if comment_ids is None:
        return False
    if not comment_ids:
        results = [gh('api', '-X', 'POST', f'repos/{repo}/issues/{pr}/comments', '--input', '-', stdin=payload)]
    else:
        results = [
            gh('api', '-X', 'PATCH', f'repos/{repo}/issues/comments/{comment_id}', '--input', '-', stdin=payload)
            for comment_id in comment_ids
        ]
    failed = [r for r in results if r.returncode != 0]
    for r in failed:
        print(f'::warning::GitHub rejected the comment: {r.stderr.strip()}')
    if not failed:
        print(f'{"Updated" if comment_ids else "Posted"} the risk assessment comment.')
    return not failed


def main() -> int:
    repo, pr = os.environ['REPO'], os.environ['PR_NUMBER']
    head_sha, run_url = os.environ.get('HEAD_SHA', ''), os.environ.get('RUN_URL', '')
    execution_file = os.environ.get('EXECUTION_FILE', '')

    commit = commit_link(repo, head_sha)
    limit = exhausted_limit(os.environ.get('OUTCOME', ''), execution_file)
    reason = None
    if os.environ.get('DIFF_TOO_LARGE') == 'true':
        reason = f'The diff of {commit} is over the size limit of the GitHub API, so it was not assessed.'
    elif limit is not None:
        reason = f'The assessment of {commit} did not finish within {limit}.'
    if reason is not None:
        print('Verdict: 2 humans; the change is too large to assess.')
        return 0 if upsert(repo, pr, render_too_large(run_url, reason)) else 1

    data = None
    if os.environ.get('OUTCOME') != 'success':
        problems = [f'the Claude step ended with outcome {os.environ.get("OUTCOME") or "unknown"}']
    elif not execution_file or not os.path.isfile(execution_file):
        problems = ['the Claude step left no execution file']
    else:
        try:
            data = review_json(final_message(execution_file))
            problems = validate(data)
        except (OSError, ValueError, TypeError, AttributeError) as e:
            data, problems = None, [f'the execution file is malformed ({type(e).__name__}: {e})']

    body = ''
    if not problems and data is not None:
        try:
            body = render(data, repo, head_sha, run_url)
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            problems = [f'the JSON block has a malformed field ({type(e).__name__}: {e})']
    if problems or data is None:
        print(f'::warning::The risk assessment is not usable: {"; ".join(problems)}')
        body = render_failure(repo, head_sha, run_url, problems)
    else:
        print(f'Verdict: {data["required_humans"]} human(s).')
    return 0 if upsert(repo, pr, body) and not problems else 1


if __name__ == '__main__':
    sys.exit(main())
