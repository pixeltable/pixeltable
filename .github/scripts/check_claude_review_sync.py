"""Compare the shared review implementation with another local checkout, without changing either."""

import argparse
from pathlib import Path

SHARED_FILES = [
    '.claude/commands/pxt-address.md',
    '.claude/commands/pxt-review.md',
    '.claude/skills/address-the-review/SKILL.md',
    '.claude/skills/address-the-review/references/critic.md',
    '.github/workflows/claude-review.yml',
    *[
        f'.github/scripts/{name}'
        for name in [
            'claude_review_contract.py',
            'prepare_claude_review.py',
            'post_claude_review.py',
            'plan_claude_review.py',
            'test_plan_claude_review.py',
            'test_claude_review_contract.py',
            'test_prepare_claude_review.py',
            'test_post_claude_review.py',
            'README.claude-review.md',
            'check_claude_review_sync.py',
        ]
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('other', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    different = [
        path
        for path in SHARED_FILES
        if not (args.other / path).is_file() or (root / path).read_bytes() != (args.other / path).read_bytes()
    ]
    if different:
        raise SystemExit('Shared review files differ: ' + ', '.join(different))
    print(f'{len(SHARED_FILES)} shared review files match.')


if __name__ == '__main__':
    main()
