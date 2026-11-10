"""Read Claude's CI-fix result, record the PR, and post `agent review` when a new fix PR opens.

The final message ends with a fenced json block, as `.claude/commands/pxt-ci-fix.md` specifies. Reads
EXECUTION_FILE, REPO, RUN_ID, WORKFLOW, FAILED_JOBS, and BRANCH from the environment; `gh` authenticates with
GH_TOKEN. Writes pr to GITHUB_OUTPUT when a fix PR should be reviewed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from post_claude_review import final_message, review_json

ACTIONS_WITH_PR = frozenset({'opened', 'updated'})


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def write_output(pr: str) -> None:
    path = os.environ.get('GITHUB_OUTPUT')
    if not path:
        return
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'pr={pr}\n')


def fallback_pr(repo: str, branch: str) -> int | None:
    if not branch:
        return None
    raw = gh('pr', 'list', '--repo', repo, '--head', branch, '--json', 'number', '--jq', '.[0].number // empty').strip()
    return int(raw) if raw.isdigit() else None


def ensure_marker(repo: str, pr: int, marker: str) -> None:
    body = gh('pr', 'view', str(pr), '--repo', repo, '--json', 'body', '--jq', '.body // ""')
    if '<!-- claude-ci-fix' in body:
        return
    separator = '' if body.endswith('\n') else '\n'
    subprocess.run(
        ['gh', 'pr', 'edit', str(pr), '--repo', repo, '--body-file', '-'],
        input=f'{body}{separator}\n{marker}\n',
        text=True,
        check=True,
        capture_output=True,
    )


def has_agent_review_comment(repo: str, pr: int) -> bool:
    comments = json.loads(gh('api', f'repos/{repo}/issues/{pr}/comments'))
    return any(str(c.get('body') or '').strip().lower().startswith('agent review') for c in comments)


def post_agent_review(repo: str, pr: int) -> None:
    gh('pr', 'comment', str(pr), '--repo', repo, '--body', 'agent review')


def marker_comment(run_id: str, workflow: str, failed_jobs: list[str]) -> str:
    payload = {'workflow': workflow, 'run': int(run_id) if run_id.isdigit() else run_id, 'jobs': failed_jobs}
    return f'<!-- claude-ci-fix {json.dumps(payload, separators=(",", ":"))} -->'


def pr_number(data: dict[str, Any] | None, repo: str, branch: str) -> int | None:
    if data is not None:
        action = data.get('action')
        pr = data.get('pr')
        if action in ACTIONS_WITH_PR and isinstance(pr, int):
            return pr
        if action not in ACTIONS_WITH_PR:
            return None
    return fallback_pr(repo, branch)


def main() -> int:
    repo = os.environ['REPO']
    branch = os.environ.get('BRANCH', '')
    data = review_json(final_message(os.environ['EXECUTION_FILE']))
    if data is None:
        print("::warning::Claude's final message has no CI-fix JSON that parses.")
    else:
        print(f'CI-fix action={data.get("action")!r}: {data.get("reason", "no reason given")}')

    pr = pr_number(data, repo, branch)
    if pr is None:
        print('No fix PR to review.')
        return 0 if data is not None else 1

    jobs_raw = os.environ.get('FAILED_JOBS', '[]')
    try:
        failed_jobs = json.loads(jobs_raw)
    except json.JSONDecodeError:
        failed_jobs = []
    if not isinstance(failed_jobs, list):
        failed_jobs = []
    ensure_marker(repo, pr, marker_comment(os.environ.get('RUN_ID', ''), os.environ.get('WORKFLOW', ''), failed_jobs))
    action = (data or {}).get('action')
    if action != 'updated' and not has_agent_review_comment(repo, pr):
        post_agent_review(repo, pr)
        print(f'Posted `agent review` on PR #{pr}.')
    print(f'Reviewing PR #{pr}.')
    write_output(str(pr))
    return 0


if __name__ == '__main__':
    sys.exit(main())
