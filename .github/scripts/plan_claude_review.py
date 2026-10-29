"""Decide whether a PR comment asks Claude for a review, and extract the focus it names.

Handles the "issue_comment" event. A comment asks for a review when it is "claude review", "agent review", or "agents
review", in any case, alone or followed by a focus. The focus follows a colon, a comma, a dash between spaces, or a line
break, or starts with the word "focus": "claude review focus on function signatures" or "claude review: error paths".
Only a commenter with write access can ask, only on an open PR, and only while no review of that PR is queued or
running. Reads GITHUB_EVENT_PATH, GITHUB_OUTPUT, GITHUB_RUN_ID, GITHUB_WORKFLOW_REF, REPO, and ACTOR from the
environment; `gh` authenticates with GH_TOKEN. Writes pr and focus to GITHUB_OUTPUT, or nothing when no review runs.
"""

import json
import os
import re
import subprocess
import sys

# Without a separator or the word "focus", a comment such as "Claude review missed the null case" is a remark, not a
# request. The separator is dropped; punctuation that starts the focus, as in ": .github/workflows", stays.
REQUEST = re.compile(
    r'(?:claude|agents?)\s+review(?:[\s.!]*$|\s*[:,]\s*|\s+-+\s+|[ \t]*\r?\n\s*|\s+(?=focus\b))(.*)',
    re.IGNORECASE | re.DOTALL,
)
# The focus becomes the last argument of the review command, on one line.
MAX_FOCUS_CHARS = 500


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def has_write_access(repo: str, login: str) -> bool:
    try:
        permission = gh('api', f'repos/{repo}/collaborators/{login}/permission', '--jq', '.permission').strip()
    except subprocess.CalledProcessError:
        return False  # bots and unknown users
    return permission in ('admin', 'write')


def review_running(repo: str, pr: int) -> bool:
    """Returns True when another run of this workflow has a review job for pr that has not completed."""
    workflow = os.environ['GITHUB_WORKFLOW_REF'].split('@')[0].rsplit('/', 1)[1]
    run_ids = gh(
        'api',
        f'repos/{repo}/actions/workflows/{workflow}/runs?per_page=50',
        '--jq',
        f'.workflow_runs[] | select(.status != "completed" and .id != {os.environ["GITHUB_RUN_ID"]}) | .id',
    ).split()
    # The workflow names each review job after its PR.
    job = f'.jobs[] | select(.name == "review #{pr}" and .status != "completed") | .id'
    return any(gh('api', f'repos/{repo}/actions/runs/{run_id}/jobs', '--jq', job).strip() for run_id in run_ids)


def focus_of(body: str) -> str | None:
    """Returns the focus that a review request names, '' when it names none, or None when body is no request."""
    match = REQUEST.match(body.strip())
    if match is None:
        return None
    focus = ' '.join(match.group(1).split())[:MAX_FOCUS_CHARS]
    # A focus without a letter or digit, as in "claude review.", names nothing.
    return focus if re.search(r'\w', focus) else ''


def main() -> int:
    repo, actor = os.environ['REPO'], os.environ['ACTOR']
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as f:
        event = json.load(f)
    focus = focus_of(event['comment']['body'])
    if focus is None:
        print('The comment asks for no review.')
        return 0
    pr = event['issue']['number']
    if event['issue']['state'] != 'open':
        print(f'PR #{pr} is closed, so no review runs.')
        return 0
    # claude-code-action fails the job when the actor lacks write access, so skip the review here instead.
    if not has_write_access(repo, actor):
        print(f'{actor} lacks write access to {repo}, so no review runs.')
        return 0
    # "agent review" also asks Grok and Muse; a request while a review runs repeats nothing and cancels nothing.
    if review_running(repo, pr):
        print(f'A review of PR #{pr} is already queued or running, so this request starts none.')
        return 0
    print(f'Reviewing PR #{pr} with focus: {focus or "(none)"}')
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        f.write(f'pr={pr}\nfocus={focus}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
