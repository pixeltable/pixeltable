#!/usr/bin/env python3
"""Fail if any test file copied into compat/tests/ differs from upstream tests/.

Checks, for every file the selection requires:
- it exists in compat/tests/ and is byte-identical to its upstream counterpart;
- the manifest written by sync_tests.py lists it with the upstream file's sha256 (catches upstream changes that were
  not synced);
and that compat/tests/ holds no other .py files besides the harness's conftest.py and utils.py, so that a test cannot
be added or patched outside the copy.

Usage: python check_unmodified.py   (exit status 0 = all unmodified)
"""

from __future__ import annotations

import hashlib
import sys

from pxt_compat.selection import (
    COMPAT_DIR,
    COMPAT_TESTS,
    DATA_LINKS,
    HARNESS_FILES,
    MANIFEST,
    UPSTREAM_TESTS,
    required_files,
)


def main() -> int:
    problems: list[str] = []
    required = required_files()
    manifest: dict[str, str] = {}
    if MANIFEST.is_file():
        for line in MANIFEST.read_text(encoding='utf-8').splitlines():
            if line.strip():
                digest, rel = line.split()
                manifest[rel] = digest
    else:
        problems.append(f'missing manifest {MANIFEST}; run sync_tests.py')

    for rel in required:
        upstream = UPSTREAM_TESTS / rel
        copy = COMPAT_TESTS / rel
        if not copy.is_file():
            problems.append(f'{rel}: not copied (run sync_tests.py)')
            continue
        upstream_bytes = upstream.read_bytes()
        if copy.read_bytes() != upstream_bytes:
            problems.append(f'{rel}: differs from upstream tests/{rel}')
        digest = hashlib.sha256(upstream_bytes).hexdigest()
        if manifest.get(rel) != digest:
            problems.append(f'{rel}: manifest is out of date (run sync_tests.py)')

    allowed = set(required) | HARNESS_FILES
    for path in sorted(COMPAT_TESTS.rglob('*.py')):
        rel = path.relative_to(COMPAT_TESTS).as_posix()
        if rel.startswith('data/'):
            continue
        if rel not in allowed:
            problems.append(f'{rel}: unexpected file in compat/tests (not an upstream copy nor a harness file)')

    for link, target in DATA_LINKS:
        link_path = COMPAT_DIR / link
        if not link_path.is_symlink():
            problems.append(f'{link}: expected a symlink to {target} (run sync_tests.py)')

    if problems:
        print('\n'.join(problems))
        print(f'FAILED: {len(problems)} problem(s)')
        return 1
    print(f'OK: {len(required)} files identical to upstream')
    return 0


if __name__ == '__main__':
    sys.exit(main())
