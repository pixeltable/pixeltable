"""Post the review in Claude's final message as one GitHub pull request review.

The final message ends with a fenced json block, as `.claude/commands/pxt-review.md` specifies. Reads EXECUTION_FILE
(the claude-code-action execution file), REPO, and PR_NUMBER from the environment; `gh` authenticates with GH_TOKEN.
"""

import json
import os
import subprocess
import sys
from typing import Any


def final_message(execution_file: str) -> str:
    with open(execution_file, encoding='utf-8') as f:
        messages = json.load(f)
    results = [m for m in messages if m.get('type') == 'result']
    return (results[-1].get('result') or '') if results else ''


def review_json(message: str) -> dict[str, Any] | None:
    # Match fences by whole lines. A JSON string cannot hold a raw newline, so a fence quoted in the body or a comment
    # never fills a line of its own.
    lines = message.splitlines()
    closes = [i for i, line in enumerate(lines) if line.strip() == '```']
    if not closes:
        return None
    opens = [i for i, line in enumerate(lines[: closes[-1]]) if line.strip() == '```json']
    if not opens:
        return None
    try:
        data: dict[str, Any] = json.loads('\n'.join(lines[opens[-1] + 1 : closes[-1]]))
    except json.JSONDecodeError as e:
        print(f'The review JSON does not parse: {e}')
        return None
    return data


def post(repo: str, pr: str, review: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['gh', 'api', '-X', 'POST', f'repos/{repo}/pulls/{pr}/reviews', '--input', '-'],
        input=json.dumps(review),
        text=True,
        capture_output=True,
        check=False,
    )


def inline_comment(comment: dict[str, Any]) -> dict[str, Any]:
    result = {'path': comment['path'], 'line': comment['line'], 'side': 'RIGHT', 'body': comment['body']}
    start_line = comment.get('start_line')
    if start_line is not None and start_line < comment['line']:
        result |= {'start_line': start_line, 'start_side': 'RIGHT'}
    return result


def main() -> int:
    repo, pr = os.environ['REPO'], os.environ['PR_NUMBER']
    data = review_json(final_message(os.environ['EXECUTION_FILE']))
    if data is None:
        print("::warning::Claude's final message has no review JSON that parses, so nothing was posted.")
        return 1
    if not data.get('post'):
        print(f'Nothing to post: {data.get("reason", "no reason given")}')
        return 0

    comments = [inline_comment(c) for c in data.get('comments', [])]
    review = {'commit_id': data['commit_id'], 'body': data['body'], 'event': 'COMMENT', 'comments': comments}
    result = post(repo, pr, review)
    if result.returncode == 0:
        print(f'Posted one review with {len(comments)} inline comments.')
        return 0

    if not comments:
        print(f'::warning::GitHub rejected the review: {result.stderr.strip()}')
        return 1

    # GitHub rejects the whole review when one comment's line is outside the diff. Move every comment into the body,
    # and leave out commit_id, in case GitHub rejected that instead.
    print(f'::warning::GitHub rejected the inline comments: {result.stderr.strip()}')
    moved = '\n\n'.join(f'**{c["path"]}:{c["line"]}**\n\n{c["body"]}' for c in comments)
    review = {'body': f'{data["body"]}\n\n### Comments not attached to lines\n\n{moved}', 'event': 'COMMENT'}
    result = post(repo, pr, review)
    if result.returncode == 0:
        print(f'Posted one review with {len(comments)} comments in its body.')
        return 0
    print(f'::warning::GitHub rejected the review: {result.stderr.strip()}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
