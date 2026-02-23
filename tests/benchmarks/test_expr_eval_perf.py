from typing import Any

import pytest

import pixeltable as pxt
from pixeltable.functions.string import isalpha, isascii, lower, upper


class TestExprEvalPerformance:
    """Benchmarks for expression evaluation dispatch optimization."""

    @pytest.mark.benchmark(group='wide_table')
    def test_wide_table_evaluation(self, uses_db: None, benchmark: Any) -> None:
        """Test performance with many computed columns (benefits from vectorized dispatch)."""
        t = pxt.create_table('wide_tbl', {'c1': pxt.Int, 'c2': pxt.String})

        # Add 20 computed columns to stress the dispatch logic
        for i in range(20):
            t.add_computed_column(**{f'computed_{i}': t.c2 + f'_{i}'})

        row_count = 10000
        t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])

        def query_wide_table() -> None:
            res = t.head(row_count)
            assert len(res) == row_count

        benchmark(query_wide_table)

    @pytest.mark.benchmark(group='dependency_chain')
    def test_dependency_chain_evaluation(self, uses_db: None, benchmark: Any) -> None:
        """Test performance with chained dependencies (tests ready-slot computation)."""
        t = pxt.create_table('chain_tbl', {'text': pxt.String})

        # Create a chain of dependencies: each column depends on the previous
        t.add_computed_column(step1=upper(t.text))
        t.add_computed_column(step2=lower(t.step1))
        t.add_computed_column(step3=upper(t.step2))
        t.add_computed_column(step4=lower(t.step3))
        t.add_computed_column(step5=upper(t.step4))

        row_count = 50000
        t.insert([{'text': f'Test_String_{i}'} for i in range(row_count)])

        def query_chain() -> None:
            res = t.select(t.step5).collect()
            assert len(res) == row_count

        benchmark(query_chain)

    @pytest.mark.benchmark(group='insert_computed')
    def test_insert_with_computed_columns(self, uses_db: None, benchmark: Any) -> None:
        """Test insert performance when computed columns need evaluation."""

        def do_insert() -> None:
            t = pxt.create_table('insert_tbl', {'c1': pxt.Int, 'c2': pxt.String})
            t.add_computed_column(c3=isascii(t.c2))
            t.add_computed_column(c4=isalpha(t.c2))
            t.add_computed_column(c5=upper(t.c2))

            row_count = 50000
            t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])
            pxt.drop_table('insert_tbl')

        benchmark(do_insert)

    @pytest.mark.benchmark(group='batch_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_batch_scaling(self, uses_db: None, benchmark: Any, row_count: int) -> None:
        """Test how performance scales with row count (vectorization benefit)."""
        t = pxt.create_table(f'scale_tbl_{row_count}', {'c1': pxt.Int, 'c2': pxt.String})
        t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])

        def select_with_functions() -> None:
            res = t.select(t.c1, isascii(t.c2), isalpha(t.c2)).collect()
            assert len(res) == row_count

        benchmark(select_with_functions)
