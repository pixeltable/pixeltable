"""Decide whether a completed scheduled CI run should start a Claude CI-fix job.

Handles the "workflow_run" event of `tests`, `stress-tests`, and `nightly`. A fix job runs only for a scheduled
run on the default branch whose conclusion is failure or timed_out, and only when that run is still the latest
scheduled run of that workflow. A closed unmerged PR that already names this run is left alone: a person took over
or dismissed it. Reads GITHUB_EVENT_PATH, GITHUB_OUTPUT, and REPO from the environment; `gh` authenticates with
GH_TOKEN. Writes run, run_url, workflow, tested_ref, failed_jobs, existing_pr, branch, and closed_prs, or nothing
when no fix job runs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Nightly checks out the `release` tag. A fix merged to main does not clear nightly until the next release.
TESTED_REF = {'nightly': 'release', 'tests': 'main', 'stress-tests': 'main'}
WATCHED = frozenset(TESTED_REF)
MARKER = '<!-- claude-ci-fix'
MARKER_JSON = re.compile(r'<!--\s*claude-ci-fix\s+(\{.*?\})\s*-->', re.DOTALL)
BRANCH_PREFIX = 'claude/ci-fix-'
FAIL_CONCLUSIONS = frozenset({'failure', 'timed_out'})
PR_JSON = 'number,title,state,headRefName,body,author,mergedAt,url,closedAt'


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def marker_payload(body: str) -> dict[str, Any] | None:
    match = MARKER_JSON.search(body)
    if match is None:
        return None
    try:
        data: dict[str, Any] = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def failed_job_names(repo: str, run_id: int) -> list[str]:
    raw = gh(
        'api',
        '--paginate',
        f'repos/{repo}/actions/runs/{run_id}/jobs',
        '--jq',
        '.jobs[] | select(.conclusion == "failure" or .conclusion == "timed_out") | .name',
    )
    return [name for name in raw.splitlines() if name]


def latest_scheduled_run_id(repo: str, workflow_file: str) -> int | None:
    raw = gh(
        'api',
        f'repos/{repo}/actions/workflows/{workflow_file}/runs?event=schedule&per_page=10',
        '--jq',
        '.workflow_runs[0].id',
    ).strip()
    return int(raw) if raw.isdigit() else None


def list_prs(repo: str, *gh_args: str) -> list[dict[str, Any]]:
    raw = gh('pr', 'list', '--repo', repo, '--json', PR_JSON, *gh_args)
    data: list[dict[str, Any]] = json.loads(raw)
    return data


def is_ci_fix_pr(pr: dict[str, Any], workflow: str) -> bool:
    head = pr.get('headRefName') or ''
    body = pr.get('body') or ''
    if head.startswith(f'{BRANCH_PREFIX}{workflow}-'):
        return True
    if MARKER not in body:
        return False
    payload = marker_payload(body)
    return payload is None or payload.get('workflow') == workflow


def closed_unmerged(pr: dict[str, Any]) -> bool:
    return pr.get('state') == 'CLOSED' and not pr.get('mergedAt')


def write_outputs(values: dict[str, str]) -> None:
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        for key, value in values.items():
            f.write(f'{key}={value}\n')


def _env_json(name: str, default: Any) -> Any:
    raw = os.environ.get(name, '')
    if not raw:
        return default
    return json.loads(raw)


def write_context_file() -> int:
    """Write ci-fix-context.json from the plan-ci-fix job outputs in the environment."""
    existing = os.environ.get('EXISTING_PR') or ''
    context = {
        'run': os.environ['RUN'],
        'run_url': os.environ['RUN_URL'],
        'workflow': os.environ['WORKFLOW_NAME'],
        'tested_ref': os.environ['TESTED_REF'],
        'failed_jobs': _env_json('FAILED_JOBS', []),
        'existing_pr': int(existing) if existing.isdigit() else None,
        'branch': os.environ['BRANCH'],
        'closed_prs': _env_json('CLOSED_PRS', []),
        'repository': os.environ['REPO'],
    }
    Path('ci-fix-context.json').write_text(json.dumps(context, indent=2) + '\n', encoding='utf-8')
    print('Wrote ci-fix-context.json')
    return 0


def main() -> int:
    repo = os.environ['REPO']
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as f:
        event = json.load(f)
    run = event.get('workflow_run') or {}
    workflow = run.get('name') or ''
    run_id = run.get('id')
    default_branch = (event.get('repository') or {}).get('default_branch') or 'main'
    if workflow not in WATCHED:
        print(f'Workflow {workflow!r} is not a scheduled CI workflow this job watches.')
        return 0
    if run.get('event') != 'schedule':
        print(f'Run {run_id} was triggered by {run.get("event")!r}, not schedule.')
        return 0
    if run.get('conclusion') not in FAIL_CONCLUSIONS:
        print(f'Run {run_id} concluded {run.get("conclusion")!r}, so no CI-fix runs.')
        return 0
    head_branch = run.get('head_branch') or ''
    if head_branch != default_branch:
        print(f'Run {run_id} is on {head_branch!r}, not {default_branch!r}.')
        return 0
    if head_branch.startswith(BRANCH_PREFIX):
        print(f'Run {run_id} is on a CI-fix branch, so no nested fix runs.')
        return 0
    workflow_file = (run.get('path') or '').rsplit('/', 1)[-1]
    if not workflow_file:
        print(f'Run {run_id} has no workflow path.')
        return 0
    latest = latest_scheduled_run_id(repo, workflow_file)
    if latest is not None and latest != run_id:
        print(f'Run {run_id} is not the latest scheduled {workflow} run ({latest} is), so no CI-fix runs.')
        return 0

    failed_jobs = failed_job_names(repo, run_id)
    run_url = run.get('html_url') or f'https://github.com/{repo}/actions/runs/{run_id}'
    mentioned = list_prs(repo, '--state', 'all', '--limit', '50', '--search', f'{run_id} in:body')
    marked = list_prs(repo, '--state', 'all', '--limit', '50', '--search', 'claude-ci-fix in:body')
    seen: set[int] = set()
    related: list[dict[str, Any]] = []
    for pr in mentioned + marked:
        number = pr.get('number')
        if not isinstance(number, int) or number in seen:
            continue
        seen.add(number)
        if is_ci_fix_pr(pr, workflow) or (pr.get('body') or '').find(str(run_id)) >= 0:
            related.append(pr)

    for pr in related:
        body = pr.get('body') or ''
        if str(run_id) not in body:
            continue
        if closed_unmerged(pr):
            print(f'PR #{pr["number"]} already names run {run_id} and was closed unmerged, so no CI-fix recreates it.')
            return 0
        if pr.get('state') == 'OPEN' and not is_ci_fix_pr(pr, workflow):
            print(f'PR #{pr["number"]} already names run {run_id}, so no CI-fix opens another.')
            return 0

    open_prs = [pr for pr in related if pr.get('state') == 'OPEN' and is_ci_fix_pr(pr, workflow)]
    existing = open_prs[0] if open_prs else None
    # One open CI-fix PR per workflow: later failures update it.
    if existing is None:
        for pr in list_prs(repo, '--state', 'open', '--limit', '100'):
            if is_ci_fix_pr(pr, workflow):
                existing = pr
                break

    closed = [
        {
            'number': pr['number'],
            'title': pr.get('title') or '',
            'url': pr.get('url') or '',
            'head': pr.get('headRefName') or '',
            'jobs': (marker_payload(pr.get('body') or '') or {}).get('jobs'),
        }
        for pr in related
        if closed_unmerged(pr)
    ]
    branch = (
        existing.get('headRefName')
        if existing and existing.get('headRefName')
        else f'{BRANCH_PREFIX}{workflow}-{run_id}'
    )
    print(
        f'Investigating {workflow} run {run_id} ({len(failed_jobs)} failed jobs); '
        f'existing PR: {existing["number"] if existing else "(none)"}'
    )
    write_outputs(
        {
            'run': str(run_id),
            'run_url': run_url,
            'workflow': workflow,
            'tested_ref': TESTED_REF[workflow],
            'failed_jobs': json.dumps(failed_jobs, separators=(',', ':')),
            'existing_pr': str(existing['number']) if existing else '',
            'branch': branch,
            'closed_prs': json.dumps(closed, separators=(',', ':')),
        }
    )
    return 0


if __name__ == '__main__':
    sys.exit(write_context_file() if sys.argv[1:] == ['context'] else main())
