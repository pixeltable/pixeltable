"""Decide whether Claude reviews a pull request, and at which tier.

Handles an "issue_comment" event (someone asked for a review) and a "workflow_run" event (a PR's tests started).
Reads GITHUB_EVENT_NAME, GITHUB_EVENT_PATH, GITHUB_OUTPUT, REPO, and ACTOR from the environment; `gh` authenticates with
GH_TOKEN. Writes run, pr, tier, and requested to GITHUB_OUTPUT.
"""

import json
import os
import re
import subprocess
import sys
from typing import Any

# Keep in sync with the files that step 2 of .claude/commands/pxt-review.md skips.
SKIPPED = re.compile(
    r'(^|/)(uv\.lock|poetry\.lock|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|Cargo\.lock|go\.sum)$'
    r'|\.min\.(js|css)$|(^|/)__snapshots__/|\.snap$'
)
SENSITIVE = re.compile(r'auth|billing|payment|stripe|secret|permission|migration|/metadata/', re.IGNORECASE)
SMALL_PR_LINES = 200


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def pr_number(event_name: str, event: dict[str, Any], repo: str) -> str | None:
    if event_name == 'issue_comment':
        return str(event['issue']['number'])
    run = event['workflow_run']
    if run['pull_requests']:
        return str(run['pull_requests'][0]['number'])
    # GitHub leaves pull_requests empty for a PR from a fork, so find the open PR by its head commit.
    prs = json.loads(
        gh('pr', 'list', '--repo', repo, '--state', 'open', '--limit', '500', '--json', 'number,headRefOid')
    )
    return next((str(p['number']) for p in prs if p['headRefOid'] == run['head_sha']), None)


def has_write_access(repo: str, login: str) -> bool:
    try:
        permission = gh('api', f'repos/{repo}/collaborators/{login}/permission', '--jq', '.permission').strip()
    except subprocess.CalledProcessError:
        return False  # bots and unknown users
    return permission in ('admin', 'write')


def plan(event_name: str, event: dict[str, Any], repo: str, actor: str) -> dict[str, str]:
    requested = event_name == 'issue_comment'
    pr = pr_number(event_name, event, repo)
    if pr is None:
        return {'run': 'false', 'pr': '', 'tier': '', 'requested': str(requested).lower()}
    info = json.loads(gh('pr', 'view', pr, '--repo', repo, '--json', 'author,isDraft,files,reviews,comments'))

    # A request needs a commenter with write access; an automatic review needs an author with write access, a PR that
    # is ready, and no earlier Claude review.
    run = has_write_access(repo, actor if requested else info['author']['login'])
    if not requested:
        bodies = [r['body'] for r in info['reviews']] + [c['body'] for c in info['comments']]
        run = run and not info['isDraft'] and not any(b.startswith('## Claude review') for b in bodies)

    reviewed = [f for f in info['files'] if not SKIPPED.search(f['path'])]
    lines = sum(f['additions'] + f['deletions'] for f in reviewed)
    sensitive = any(SENSITIVE.search(f['path']) for f in reviewed)
    tier = 'small' if lines < SMALL_PR_LINES and not sensitive else 'large'
    print(f'PR #{pr}: run={run}, {lines} changed lines, sensitive={sensitive}, tier={tier}')
    return {'run': str(run).lower(), 'pr': pr, 'tier': tier, 'requested': str(requested).lower()}


def main() -> int:
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as f:
        event = json.load(f)
    outputs = plan(os.environ['GITHUB_EVENT_NAME'], event, os.environ['REPO'], os.environ['ACTOR'])
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        f.writelines(f'{key}={value}\n' for key, value in outputs.items())
    return 0


if __name__ == '__main__':
    sys.exit(main())
