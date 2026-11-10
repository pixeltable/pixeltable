"""Decide whether a completed scheduled CI run should start a Claude CI-fix job.

Handles the "workflow_run" event of `tests`, `stress-tests`, and `nightly`. A fix job runs only for a scheduled
run on the default branch whose conclusion is failure or timed_out, and only when that run is still the latest
scheduled run of that workflow. It does not run when a PR already names this run (a person took over, or closed
it) or when this workflow already has an open CI-fix PR. The fix branch is computed here from the run id and is
never read from a PR. Reads GITHUB_EVENT_PATH, GITHUB_OUTPUT, and REPO from the environment; `gh` authenticates
with GH_TOKEN. Writes workflow, branch, and context (the JSON the fix job reads), or nothing when no fix job runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

# Nightly checks out the `release` tag. A fix merged to main does not clear nightly until the next release.
TESTED_REF = {'nightly': 'release', 'tests': 'main', 'stress-tests': 'main'}
BRANCH_PREFIX = 'claude/ci-fix-'
FAIL_CONCLUSIONS = frozenset({'failure', 'timed_out'})
PR_JSON = 'number,title,state,headRefName,body,mergedAt,url,isCrossRepository'


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


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
    """A PR in this repository on a branch this job names. Pushing such a branch needs write access."""
    return pr.get('isCrossRepository') is False and (pr.get('headRefName') or '').startswith(
        f'{BRANCH_PREFIX}{workflow}-'
    )


def closed_unmerged(pr: dict[str, Any]) -> bool:
    return pr.get('state') == 'CLOSED' and not pr.get('mergedAt')


def write_outputs(values: dict[str, str]) -> None:
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        for key, value in values.items():
            f.write(f'{key}={value}\n')


def main() -> int:
    repo = os.environ['REPO']
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as f:
        event = json.load(f)
    run = event.get('workflow_run') or {}
    workflow = run.get('name') or ''
    run_id = run.get('id')
    default_branch = (event.get('repository') or {}).get('default_branch') or 'main'
    if workflow not in TESTED_REF:
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
    workflow_file = (run.get('path') or '').rsplit('/', 1)[-1]
    if not workflow_file:
        print(f'Run {run_id} has no workflow path.')
        return 0
    latest = latest_scheduled_run_id(repo, workflow_file)
    if latest is not None and latest != run_id:
        print(f'Run {run_id} is not the latest scheduled {workflow} run ({latest} is), so no CI-fix runs.')
        return 0

    # A PR of any author that names this run and did not merge: a person took it over, or closed it, or it is open.
    for pr in list_prs(repo, '--state', 'all', '--limit', '50', '--search', f'{run_id} in:body'):
        if str(run_id) in (pr.get('body') or '') and not pr.get('mergedAt'):
            print(f'PR #{pr["number"]} already names run {run_id}, so no CI-fix runs.')
            return 0

    fix_prs = [pr for pr in list_prs(repo, '--state', 'all', '--limit', '200') if is_ci_fix_pr(pr, workflow)]
    for pr in fix_prs:
        if pr.get('state') == 'OPEN':
            print(f'PR #{pr["number"]} is already an open {workflow} CI-fix PR, so no CI-fix runs.')
            return 0

    failed_jobs = failed_job_names(repo, run_id)
    branch = f'{BRANCH_PREFIX}{workflow}-{run_id}'
    context = {
        'run': run_id,
        'run_url': run.get('html_url') or f'https://github.com/{repo}/actions/runs/{run_id}',
        'workflow': workflow,
        'tested_ref': TESTED_REF[workflow],
        'failed_jobs': failed_jobs,
        'branch': branch,
        'closed_prs': [
            {'number': pr['number'], 'title': pr.get('title') or '', 'url': pr.get('url') or ''}
            for pr in fix_prs
            if closed_unmerged(pr)
        ],
        'repository': repo,
    }
    print(f'Investigating {workflow} run {run_id} ({len(failed_jobs)} failed jobs) on {branch}.')
    write_outputs({'workflow': workflow, 'branch': branch, 'context': json.dumps(context, separators=(',', ':'))})
    return 0


if __name__ == '__main__':
    sys.exit(main())
