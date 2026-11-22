"""Tests for 'pxt guide'."""

import pathlib

import pytest

import pixeltable

from .conftest import PxtRunner

pytestmark = pytest.mark.db_roots('local', reason='prints files of the installed package, independent of any catalog')

_GUIDES = sorted(
    path
    for path in (pathlib.Path(pixeltable.__file__).parent / 'agent_guides').glob('*.md')
    if path.name != 'README.md'
)


class TestGuide:
    def test_guide(self, cli: PxtRunner) -> None:
        """Without an area, the command lists each guide with the line after its title; with one, it prints that
        guide, in text or as JSON; an unknown area is a usage error that lists the areas."""
        summaries = dict(line.split(maxsplit=1) for line in cli('guide').stdout.splitlines())
        assert list(summaries) == [path.stem for path in _GUIDES]
        assert summaries['tables'] == "Declaring tables, writing rows, and changing a table's schema."
        assert cli('guide', '--json').json == summaries

        tables = (_GUIDES[0].parent / 'tables.md').read_text(encoding='utf-8')
        assert cli('guide', 'tables').stdout == tables
        assert cli('guide', 'tables', '--json').json == {'area': 'tables', 'text': tables}

        r = cli('guide', 'tabels', check=False)
        assert r.returncode == 2
        assert "invalid choice: 'tabels'" in r.stderr
        assert 'tables' in r.stderr
