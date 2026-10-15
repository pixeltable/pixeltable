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
# The sweep looks back further than its own interval, so a skipped or late run does not lose a PR.
READY_WINDOW = datetime.timedelta(hours=24)

READY_PRS_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        isDraft
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


def pr_from_tests_run(event: dict[str, Any], repo: str) -> str | None:
    run = event['workflow_run']
    if run['pull_requests']:
        return str(run['pull_requests'][0]['number'])
    # GitHub leaves pull_requests empty for a PR from a fork, so find the open PR by its head commit.
    prs = json.loads(
        gh('pr', 'list', '--repo', repo, '--state', 'open', '--limit', '500', '--json', 'number,headRefOid')
    )
    return next((str(p['number']) for p in prs if p['headRefOid'] == run['head_sha']), None)


def prs_marked_ready(repo: str, now: datetime.datetime) -> list[str]:
    owner, name = repo.split('/')
    data = json.loads(
        gh('api', 'graphql', '-f', f'query={READY_PRS_QUERY}', '-F', f'owner={owner}', '-F', f'name={name}')
    )
    result = []
    for pr in data['data']['repository']['pullRequests']['nodes']:
        events = pr['timelineItems']['nodes']
        if pr['isDraft'] or not events:
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


def plan(event_name: str, event: dict[str, Any], repo: str, actor: str, now: datetime.datetime) -> list[dict[str, Any]]:
    if event_name == 'issue_comment':
        candidates, requested = [str(event['issue']['number'])], True
    elif event_name == 'workflow_run':
        pr = pr_from_tests_run(event, repo)
        candidates, requested = ([pr] if pr is not None else []), False
    else:
        candidates, requested = prs_marked_ready(repo, now), False
    decisions = (decide(repo, pr, requested, actor) for pr in candidates)
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
