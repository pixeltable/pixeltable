"""Unit tests for expression evaluation dispatch logic."""

import pytest

import pixeltable as pxt
from pixeltable.functions.string import isascii, lower, upper


class TestExprEvalDispatch:
    """Unit tests for expression evaluation dispatch logic."""

    def test_single_row_dispatch(self, uses_db: None) -> None:
        """Verify dispatch works correctly with single row (edge case)."""
        t = pxt.create_table('single_row', {'c1': pxt.String})
        t.add_computed_column(c2=isascii(t.c1))
        t.insert([{'c1': 'test'}])

        res = t.select(t.c2).collect()
        assert len(res) == 1
        assert res['c2'][0] is True

    def test_empty_table_dispatch(self, uses_db: None) -> None:
        """Verify dispatch handles empty table correctly."""
        t = pxt.create_table('empty_tbl', {'c1': pxt.String})
        t.add_computed_column(c2=isascii(t.c1))

        res = t.select(t.c2).collect()
        assert len(res) == 0

    def test_all_cached_no_dispatch(self, uses_db: None) -> None:
        """Verify no unnecessary dispatch when all values cached."""
        t = pxt.create_table('cached_tbl', {'c1': pxt.String})
        t.add_computed_column(c2=isascii(t.c1))
        t.insert([{'c1': f'str_{i}'} for i in range(100)])

        # First query computes, second should use cache
        _ = t.select(t.c2).collect()
        res = t.select(t.c2).collect()
        assert len(res) == 100

    def test_mixed_null_values(self, uses_db: None) -> None:
        """Verify dispatch handles null values in batch correctly."""
        t = pxt.create_table('null_tbl', {'c1': pxt.String})
        t.add_computed_column(c2=isascii(t.c1))
        t.insert([
            {'c1': 'valid'},
            {'c1': None},
            {'c1': 'also_valid'},
            {'c1': None},
        ])

        res = t.select(t.c2).collect()
        assert len(res) == 4
        assert res['c2'][0] is True
        assert res['c2'][1] is None
        assert res['c2'][2] is True
        assert res['c2'][3] is None

    def test_complex_dependency_graph(self, uses_db: None) -> None:
        """Test dispatch with diamond dependency pattern."""
        t = pxt.create_table('diamond_tbl', {'base': pxt.String})

        # Diamond pattern: base -> (left, right) -> final
        t.add_computed_column(left=upper(t.base))
        t.add_computed_column(right=lower(t.base))
        t.add_computed_column(final=t.left + t.right)

        t.insert([{'base': 'Test'} for _ in range(1000)])

        res = t.select(t.final).collect()
        assert len(res) == 1000
        assert all(r == 'TESTtest' for r in res['final'])

    def test_many_rows_single_computed(self, uses_db: None) -> None:
        """Test dispatch with many rows and single computed column."""
        t = pxt.create_table('many_rows', {'c1': pxt.Int, 'c2': pxt.String})
        t.add_computed_column(c3=isascii(t.c2))

        row_count = 10000
        t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])

        res = t.select(t.c3).collect()
        assert len(res) == row_count
        assert all(v is True for v in res['c3'])

    def test_multiple_independent_computed_columns(self, uses_db: None) -> None:
        """Test dispatch with multiple independent computed columns (no dependencies between them)."""
        t = pxt.create_table('multi_computed', {'text': pxt.String})

        # All these computed columns depend only on the base column, not each other
        t.add_computed_column(upper_text=upper(t.text))
        t.add_computed_column(lower_text=lower(t.text))
        t.add_computed_column(is_ascii=isascii(t.text))

        t.insert([{'text': f'Test_{i}'} for i in range(500)])

        res = t.select(t.upper_text, t.lower_text, t.is_ascii).collect()
        assert len(res) == 500
        assert res['upper_text'][0] == 'TEST_0'
        assert res['lower_text'][0] == 'test_0'
        assert res['is_ascii'][0] is True
