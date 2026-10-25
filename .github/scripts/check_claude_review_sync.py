"""Compare the shared review implementation with another local checkout, without changing either."""

from __future__ import annotations

import argparse
import subprocess
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


def has_ref(repo: Path, ref: str) -> bool:
    result = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', '--verify', '--quiet', f'{ref}^{{commit}}'],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def merged_tree(repo: Path, base: str) -> str:
    result = subprocess.run(
        ['git', '-C', str(repo), 'merge-tree', '--write-tree', '-z', base, 'HEAD'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise SystemExit(f'Merge preview failed in {repo}: {(result.stderr or result.stdout).strip()}')
    parts = result.stdout.split('\0')
    tree = parts[0].strip()
    if result.returncode == 1:
        end = parts.index('', 1) if '' in parts[1:] else len(parts)
        conflicts = {item.split('\t', 1)[1] for item in parts[1:end] if '\t' in item}
        idx = end + 1
        while idx < len(parts) and parts[idx].isdigit():
            count = int(parts[idx])
            paths = parts[idx + 1 : idx + 1 + count]
            kind = parts[idx + 1 + count] if idx + 1 + count < len(parts) else ''
            if kind.startswith('CONFLICT'):
                conflicts.update(paths)
            idx += count + 3
        shared_conflicts = [path for path in SHARED_FILES if path in conflicts]
        if shared_conflicts:
            raise SystemExit(f'Shared review files have merge conflicts in {repo}: ' + ', '.join(shared_conflicts))
    return tree


def blob_bytes(repo: Path, tree: str, path: str) -> bytes | None:
    exists = subprocess.run(
        ['git', '-C', str(repo), 'cat-file', '-e', f'{tree}:{path}'], capture_output=True, check=False
    )
    if exists.returncode != 0:
        return None
    return subprocess.check_output(['git', '-C', str(repo), 'cat-file', 'blob', f'{tree}:{path}'])


def check_sync(root: Path, other: Path, base: str | None = None) -> None:
    different = [
        path
        for path in SHARED_FILES
        if not (root / path).is_file()
        or not (other / path).is_file()
        or (root / path).read_bytes() != (other / path).read_bytes()
    ]
    if different:
        raise SystemExit('Shared review files differ: ' + ', '.join(different))
    ref = base or 'origin/main'
    if base is not None:
        for repo in [root, other]:
            if not has_ref(repo, ref):
                raise SystemExit(f'Base ref {ref} was not found in {repo}')
    if has_ref(root, ref) and has_ref(other, ref):
        root_tree = merged_tree(root, ref)
        other_tree = merged_tree(other, ref)
        merged_different = [
            path
            for path in SHARED_FILES
            if (left := blob_bytes(root, root_tree, path)) is None or left != blob_bytes(other, other_tree, path)
        ]
        if merged_different:
            raise SystemExit('Shared review merge preview differs: ' + ', '.join(merged_different))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('other', type=Path)
    parser.add_argument('--base', default=None, help='Base ref for merge preview (default: origin/main when present)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    check_sync(root, args.other, args.base)
    print(f'{len(SHARED_FILES)} shared review files match.')


if __name__ == '__main__':
    main()
