"""Decide which pull requests Claude reviews, and at which tier.

Handles an "issue_comment" event (someone asked for a review), a "workflow_run" event (a PR's tests started), and a
"schedule" event (a sweep for PRs marked ready for review, which start no tests run). Reads GITHUB_EVENT_NAME,
GITHUB_EVENT_PATH, GITHUB_OUTPUT, REPO, and ACTOR from the environment; `gh` authenticates with GH_TOKEN. Writes
reviews, a JSON list of {"pr", "tier", "requested"}, to GITHUB_OUTPUT.
"""

import datetime
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
# The sweep runs hourly and looks back three hours: a late or skipped run does not lose a PR, and a PR whose review
# fails or stops before posting is tried at most three times.
READY_WINDOW = datetime.timedelta(hours=3)
# Errors that make the sweep skip a PR, or the whole run, instead of failing; the next hourly run tries again.
SWEEP_ERRORS = (subprocess.CalledProcessError, KeyError, TypeError, ValueError)

OPEN_PRS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef { name }
    pullRequests(states: OPEN, first: 100, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        isDraft
        baseRefName
        headRefOid
        timelineItems(itemTypes: [READY_FOR_REVIEW_EVENT], last: 1) {
          nodes { ... on ReadyForReviewEvent { createdAt } }
        }
      }
    }
  }
}
"""


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def open_prs(repo: str) -> tuple[str, list[dict[str, Any]]]:
    """Returns (default branch name, every open PR)."""
    owner, name = repo.split('/')
    prs: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        args = ['api', 'graphql', '-f', f'query={OPEN_PRS_QUERY}', '-F', f'owner={owner}', '-F', f'name={name}']
        if cursor is not None:
            args += ['-f', f'cursor={cursor}']
        repository = json.loads(gh(*args))['data']['repository']
        page = repository['pullRequests']
        prs += page['nodes']
        if not page['pageInfo']['hasNextPage']:
            return repository['defaultBranchRef']['name'], prs
        cursor = page['pageInfo']['endCursor']


def pr_from_tests_run(event: dict[str, Any], repo: str) -> str | None:
    run = event['workflow_run']
    if run['pull_requests']:
        return str(run['pull_requests'][0]['number'])
    # GitHub leaves pull_requests empty for a PR from a fork, so find the open PR by its head commit.
    _, prs = open_prs(repo)
    return next((str(p['number']) for p in prs if p['headRefOid'] == run['head_sha']), None)


def prs_marked_ready(repo: str, now: datetime.datetime) -> list[str]:
    default_branch, prs = open_prs(repo)
    result = []
    for pr in prs:
        events = pr['timelineItems']['nodes']
        # `tests` runs only for PRs into the default branch, so the sweep skips PRs into other branches.
        if pr['isDraft'] or pr['baseRefName'] != default_branch or not events:
            continue
        if now - datetime.datetime.fromisoformat(events[0]['createdAt']) <= READY_WINDOW:
            result.append(str(pr['number']))
    return result


def has_write_access(repo: str, login: str) -> bool:
    try:
        permission = gh('api', f'repos/{repo}/collaborators/{login}/permission', '--jq', '.permission').strip()
    except subprocess.CalledProcessError:
        return False  # bots and unknown users
    return permission in ('admin', 'write')


def decide(repo: str, pr: str, requested: bool, actor: str) -> dict[str, Any] | None:
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
    return {'pr': pr, 'tier': tier, 'requested': requested} if run else None


def error_detail(e: Exception) -> str:
    return e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else repr(e)


def sweep(repo: str, actor: str, now: datetime.datetime) -> list[dict[str, Any] | None]:
    try:
        candidates = prs_marked_ready(repo, now)
    except SWEEP_ERRORS as e:
        print(f'::warning::The sweep could not list open PRs: {error_detail(e)}')
        return []
    decisions = []
    for pr in candidates:
        try:
            decisions.append(decide(repo, pr, False, actor))
        except SWEEP_ERRORS as e:
            print(f'::warning::Skipped PR #{pr}: {error_detail(e)}')
    return decisions


def plan(event_name: str, event: dict[str, Any], repo: str, actor: str, now: datetime.datetime) -> list[dict[str, Any]]:
    # Errors outside the sweep fail the run, so a failed request or tests-run review shows as a failed run.
    if event_name == 'issue_comment':
        decisions = [decide(repo, str(event['issue']['number']), True, actor)]
    elif event_name == 'workflow_run':
        pr = pr_from_tests_run(event, repo)
        decisions = [decide(repo, pr, False, actor)] if pr is not None else []
    else:
        decisions = sweep(repo, actor, now)
    return [d for d in decisions if d is not None]


def main() -> int:
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as f:
        event = json.load(f)
    now = datetime.datetime.now(datetime.timezone.utc)
    reviews = plan(os.environ['GITHUB_EVENT_NAME'], event, os.environ['REPO'], os.environ['ACTOR'], now)
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        f.write(f'reviews={json.dumps(reviews)}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
