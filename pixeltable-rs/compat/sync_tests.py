#!/usr/bin/env python3
"""Copy the selected upstream test files, and the tests/ modules they import, verbatim into compat/tests/.

conftest.py and utils.py are never copied: the harness replaces them. Test data is not copied either; it is reached
through symlinks into the upstream checkout (see selection.DATA_LINKS).

Usage: python sync_tests.py [--prune]
  --prune  also delete files in compat/tests/ that an earlier sync copied but the selection no longer needs
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

from pxt_compat.selection import (
    COMPAT_DIR,
    COMPAT_TESTS,
    DATA_LINKS,
    MANIFEST,
    UPSTREAM_TESTS,
    manifest_files,
    required_files,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prune', action='store_true', help='delete previously synced files no longer selected')
    args = parser.parse_args()

    files = required_files()
    COMPAT_TESTS.mkdir(exist_ok=True)
    lines: list[str] = []
    changed = 0
    for rel in files:
        src = UPSTREAM_TESTS / rel
        dst = COMPAT_TESTS / rel
        data = src.read_bytes()
        if not dst.is_file() or dst.read_bytes() != data:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            changed += 1
            print(f'copied   {rel}')
        lines.append(f'{sha256(data)}  {rel}')

    stale = sorted(set(manifest_files()) - set(files))
    for rel in stale:
        if args.prune:
            (COMPAT_TESTS / rel).unlink(missing_ok=True)
            print(f'pruned   {rel}')
        else:
            print(f'stale    {rel} (no longer selected; rerun with --prune to delete)')

    MANIFEST.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    for link, target in DATA_LINKS:
        link_path = COMPAT_DIR / link
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink() and os.readlink(link_path) == target:
            continue
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_dir() and not link_path.is_symlink():
                print(f'error: {link_path} is a directory, expected a symlink to {target}', file=sys.stderr)
                return 1
            link_path.unlink()
        link_path.symlink_to(target, target_is_directory=True)
        print(f'linked   {link} -> {target}')
        if not link_path.resolve().is_dir():
            print(f'warning: {link} points to a missing directory', file=sys.stderr)

    print(f'{len(files)} files in manifest, {changed} copied, {len(stale)} stale')
    return 0


if __name__ == '__main__':
    sys.exit(main())
