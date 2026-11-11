"""Find the fix PR Claude opened, comment `agent review` on it, and name it for the review job.

The final message ends with a fenced json block, as `.claude/commands/pxt-ci-fix.md` specifies. Reads
EXECUTION_FILE, REPO, and BRANCH from the environment; `gh` authenticates with GH_TOKEN. The PR comes from BRANCH,
which plan_claude_ci_fix.py computed, not from the number in Claude's message. Writes pr to GITHUB_OUTPUT when a
fix PR opened.
"""

from __future__ import annotations

import os
import subprocess
import sys

from post_claude_review import final_message, review_json


def gh(*args: str) -> str:
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def pr_on_branch(repo: str, branch: str) -> int | None:
    raw = gh(
        'pr',
        'list',
        '--repo',
        repo,
        '--head',
        branch,
        '--json',
        'number,isCrossRepository',
        '--jq',
        '[.[] | select(.isCrossRepository == false)][0].number // empty',
    ).strip()
    return int(raw) if raw.isdigit() else None


def main() -> int:
    repo, branch = os.environ['REPO'], os.environ['BRANCH']
    data = review_json(final_message(os.environ['EXECUTION_FILE']))
    if data is None:
        print("::warning::Claude's final message has no CI-fix JSON that parses.")
        return 1
    print(f'CI-fix action={data.get("action")!r}: {data.get("reason", "no reason given")}')
    if data.get('action') != 'opened':
        print('No fix PR to review.')
        return 0
    pr = pr_on_branch(repo, branch)
    if pr is None:
        print(f'::warning::Claude reported a fix PR, but {branch} has no open PR in {repo}.')
        return 1
    gh('pr', 'comment', str(pr), '--repo', repo, '--body', 'agent review')
    print(f'Posted `agent review` on PR #{pr}.')
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        f.write(f'pr={pr}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
