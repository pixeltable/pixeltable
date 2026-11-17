"""Post the risk assessment in Claude's final message as one PR comment, and update that comment on later runs.

The final message ends with a fenced json block, as `.claude/commands/pxt-risk.md` specifies. The comment carries a
hidden marker, so each run edits the same comment instead of adding one. When the assessment did not complete or its
JSON is not valid, the comment says so, so that a verdict for an older commit never looks current.

Reads REPO, PR_NUMBER, HEAD_SHA, RUN_URL, OUTCOME (the Claude step's outcome), and EXECUTION_FILE (the
claude-code-action execution file) from the environment; `gh` authenticates with GH_TOKEN.
"""

import json
import os
import subprocess
import sys
from typing import Any

from post_claude_review import final_message, review_json

MARKER = '<!-- pxt-risk-assessment -->'
COMMENT_AUTHOR = 'github-actions[bot]'
# GitHub rejects comment bodies over 65536 characters.
MAX_BODY_CHARS = 60000


def validate(data: dict[str, Any] | None) -> list[str]:
    """Returns the problems that make the assessment unusable, or an empty list."""
    if data is None:
        return ['the final message has no JSON block that parses']
    problems = []
    if data.get('required_humans') not in (1, 2):
        problems.append(f'required_humans is {data.get("required_humans")!r}, not 1 or 2')
    if not isinstance(data.get('summary'), str) or not data['summary'].strip():
        problems.append('summary is missing')
    reasons = data.get('reasons')
    if data.get('required_humans') == 2 and not (
        isinstance(reasons, list) and any(isinstance(r, dict) and r.get('why') for r in reasons)
    ):
        problems.append('a verdict of 2 humans has no reasons')
    for key in ('affected', 'reasons', 'linked_prs', 'register_gaps'):
        if key in data and not isinstance(data[key], list):
            problems.append(f'{key} is not a list')
    return problems


def clean(value: Any) -> str:
    """Agent text on one line, without anything that could fake or break the marker."""
    return ' '.join(str(value).replace('<!--', '').replace('-->', '').split())


def commit_link(repo: str, sha: str) -> str:
    return f'[`{sha[:10]}`](https://github.com/{repo}/commit/{sha})' if sha else 'an unknown commit'


def render(data: dict[str, Any], repo: str, head_sha: str, run_url: str) -> str:
    humans = data['required_humans']
    sha = clean(data.get('head_sha') or head_sha)
    verdict = '2 humans required' if humans == 2 else '1 human is enough'
    lines = [MARKER, f'## Risk assessment: {verdict}', '']
    confidence = f' Confidence: {clean(data["confidence"])}.' if data.get('confidence') else ''
    lines += [f'Assessed {commit_link(repo, sha)}.{confidence}', '', clean(data['summary']), '']

    reasons = [r for r in data.get('reasons') or [] if isinstance(r, dict) and r.get('why')]
    if humans == 2:
        lines += ['### Why 2 humans', '']
        for r in reasons:
            invariant = f'**{clean(r["invariant"])}**: ' if r.get('invariant') else ''
            evidence = (
                f' ([evidence]({clean(r["evidence"])}))' if str(r.get('evidence', '')).startswith('https://') else ''
            )
            lines.append(f'- {invariant}{clean(r["why"])}{evidence}')
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
    if gaps:
        lines += [f'<details><summary>Risks the register does not cover yet ({len(gaps)})</summary>', '']
        lines += [f'- {g}' for g in gaps]
        lines += ['', '</details>', '']

    lines.append(footer(run_url))
    body = '\n'.join(lines)
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + '\n\n(truncated)\n\n' + footer(run_url)
    return body


def render_failure(repo: str, head_sha: str, run_url: str, problems: list[str]) -> str:
    return '\n'.join(
        [
            MARKER,
            '## Risk assessment: not available',
            '',
            f'The assessment of {commit_link(repo, head_sha)} did not complete: {"; ".join(problems)}. '
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


def existing_comment(repo: str, pr: str) -> str | None:
    """Returns the id of this check's comment on the PR, if there is one."""
    query = f'.[] | select(.user.login == "{COMMENT_AUTHOR}" and (.body | contains("{MARKER}"))) | .id'
    result = gh('api', '--paginate', f'repos/{repo}/issues/{pr}/comments', '--jq', query)
    if result.returncode != 0:
        print(f'::warning::Listing the PR comments failed: {result.stderr.strip()}')
        return None
    ids = result.stdout.split()
    return ids[0] if ids else None


def upsert(repo: str, pr: str, body: str) -> bool:
    payload = json.dumps({'body': body})
    comment_id = existing_comment(repo, pr)
    if comment_id is None:
        result = gh('api', '-X', 'POST', f'repos/{repo}/issues/{pr}/comments', '--input', '-', stdin=payload)
    else:
        result = gh('api', '-X', 'PATCH', f'repos/{repo}/issues/comments/{comment_id}', '--input', '-', stdin=payload)
    if result.returncode != 0:
        print(f'::warning::GitHub rejected the comment: {result.stderr.strip()}')
        return False
    print(f'{"Updated" if comment_id else "Posted"} the risk assessment comment.')
    return True


def main() -> int:
    repo, pr = os.environ['REPO'], os.environ['PR_NUMBER']
    head_sha, run_url = os.environ.get('HEAD_SHA', ''), os.environ.get('RUN_URL', '')
    execution_file = os.environ.get('EXECUTION_FILE', '')

    data = None
    if os.environ.get('OUTCOME') != 'success':
        problems = [f'the Claude step ended with outcome {os.environ.get("OUTCOME") or "unknown"}']
    elif not execution_file or not os.path.isfile(execution_file):
        problems = ['the Claude step left no execution file']
    else:
        data = review_json(final_message(execution_file))
        problems = validate(data)

    if problems or data is None:
        print(f'::warning::The risk assessment is not usable: {"; ".join(problems)}')
        body = render_failure(repo, head_sha, run_url, problems)
    else:
        body = render(data, repo, head_sha, run_url)
        print(f'Verdict: {data["required_humans"]} human(s).')
    return 0 if upsert(repo, pr, body) and not problems else 1


if __name__ == '__main__':
    sys.exit(main())
