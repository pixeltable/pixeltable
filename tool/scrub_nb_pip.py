#!/usr/bin/env python3
"""Drop `%pip install` lines from notebooks and retag pip-only cells.

Used by `scripts/prepare-nb-tests.sh --no-pip`. Parses JSON so quoted package specifiers
cannot break the notebook (PXT-1767).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PIP_PREFIX = '%pip install'
PIP_ONLY_TAG = 'execute-only-with-pip'
SKIP_TAG = 'skip-execution'


def _source_lines(source: str | list[Any]) -> list[str]:
    text = source if isinstance(source, str) else ''.join(str(part) for part in source)
    return text.splitlines(keepends=True)


def _drop_pip_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if not line.lstrip().startswith(PIP_PREFIX)]


def _replace_pip_tag(cell: dict[str, Any]) -> None:
    metadata = cell.get('metadata')
    if not isinstance(metadata, dict):
        return
    tags = metadata.get('tags')
    if not isinstance(tags, list):
        return
    metadata['tags'] = [SKIP_TAG if tag == PIP_ONLY_TAG else tag for tag in tags]


def scrub_notebook(nb: dict[str, Any]) -> dict[str, Any]:
    """Remove pip-install lines from code cells and retag pip-only cells. Mutates `nb`."""
    cells = nb.get('cells', [])
    if not isinstance(cells, list):
        return nb
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, (str, list)):
                cell['source'] = _drop_pip_lines(_source_lines(source))
        _replace_pip_tag(cell)
    return nb


def scrub_file(src: Path, dest: Path) -> None:
    nb: dict[str, Any] = json.loads(src.read_text(encoding='utf-8'))
    scrub_notebook(nb)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> int:
    if len(sys.argv) != 3:
        print('Usage: scrub_nb_pip.py INPUT OUTPUT', file=sys.stderr)
        return 1
    scrub_file(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
