"""Decide whether a PR comment asks Claude for a review, and extract the focus it names.

Handles the "issue_comment" event. A comment asks for a review when it starts with "claude review", "agent review", or
"agents review", in any case, and the rest of the comment names the focus, such as "focus on function signatures". Only
a commenter with write access can ask. Reads GITHUB_EVENT_PATH, GITHUB_OUTPUT, REPO, and ACTOR from the environment;
`gh` authenticates with GH_TOKEN. Writes pr and focus to GITHUB_OUTPUT, or nothing when no review runs.
"""

import json
import os
import re
import subprocess
import sys

REQUEST = re.compile(r'(?:claude|agents?)\s+review\b[\s:,.-]*(.*)', re.IGNORECASE | re.DOTALL)
# The focus becomes the last argument of the review command, on one line.
MAX_FOCUS_CHARS = 500


def has_write_access(repo: str, login: str) -> bool:
    try:
        permission = subprocess.run(
            ['gh', 'api', f'repos/{repo}/collaborators/{login}/permission', '--jq', '.permission'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return False  # bots and unknown users
    return permission in ('admin', 'write')


def focus_of(body: str) -> str | None:
    """Returns the focus that a review request names, '' when it names none, or None when body is no request."""
    match = REQUEST.match(body.strip())
    if match is None:
        return None
    return ' '.join(match.group(1).split())[:MAX_FOCUS_CHARS]


def main() -> int:
    repo, actor = os.environ['REPO'], os.environ['ACTOR']
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as f:
        event = json.load(f)
    focus = focus_of(event['comment']['body'])
    if focus is None:
        print('The comment asks for no review.')
        return 0
    # claude-code-action fails the job when the actor lacks write access, so skip the review here instead.
    if not has_write_access(repo, actor):
        print(f'{actor} lacks write access to {repo}, so no review runs.')
        return 0
    pr = event['issue']['number']
    print(f'Reviewing PR #{pr} with focus: {focus or "(none)"}')
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        f.write(f'pr={pr}\nfocus={focus}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
