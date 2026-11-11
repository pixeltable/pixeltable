"""Regression tests for JSON-aware %pip scrubbing (PXT-1767)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tool.scrub_nb_pip import scrub_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_RELEASE = REPO_ROOT / 'docs' / 'release'
SCRUB_NB_PIP = REPO_ROOT / 'tool' / 'scrub_nb_pip.py'

# Mirrors docs/release/howto/cookbooks/video/video-generate-ai.ipynb after #1880.
QUOTED_PIP_LINE = '%pip install -qU pixeltable "google-genai>=2.22"'


def _notebook(*cells: dict[str, Any]) -> dict[str, Any]:
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': list(cells)}


def _code_cell(source: list[str], tags: list[str] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {} if tags is None else {'tags': tags}
    return {'cell_type': 'code', 'execution_count': None, 'metadata': metadata, 'outputs': [], 'source': source}


class TestScrubNbPip:
    """prepare-nb-tests.sh --no-pip must leave valid JSON when pip lines quote specifiers."""

    def test_quoted_specifier_roundtrip(self, tmp_path: Path) -> None:
        src = tmp_path / 'quoted.ipynb'
        dest = tmp_path / 'scrubbed.ipynb'
        src.write_text(
            json.dumps(
                _notebook(
                    _code_cell([QUOTED_PIP_LINE + '\n', 'import pixeltable as pxt\n'], tags=['execute-only-with-pip'])
                ),
                indent=1,
            )
            + '\n',
            encoding='utf-8',
        )
        raw = src.read_text(encoding='utf-8')
        assert r'\"google-genai>=2.22\"' in raw

        subprocess.run([sys.executable, str(SCRUB_NB_PIP), str(src), str(dest)], check=True)

        parsed = json.loads(dest.read_text(encoding='utf-8'))
        text = dest.read_text(encoding='utf-8')
        assert '%pip install' not in text
        assert 'skip-execution' in text
        cell = parsed['cells'][0]
        assert cell['source'] == ['import pixeltable as pxt\n']
        assert cell['metadata']['tags'] == ['skip-execution']

    def test_docs_release_notebooks_remain_valid_json(self, tmp_path: Path) -> None:
        notebooks = [p for p in DOCS_RELEASE.rglob('*.ipynb') if '.ipynb_checkpoints' not in p.parts]
        assert notebooks, f'no notebooks under {DOCS_RELEASE}'
        for src in notebooks:
            dest = tmp_path / src.relative_to(DOCS_RELEASE)
            scrub_file(src, dest)
            text = dest.read_text(encoding='utf-8')
            json.loads(text)
            assert '%pip install' not in text, src
