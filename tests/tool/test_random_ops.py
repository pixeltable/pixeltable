import logging
from typing import Iterator

import pytest

import pixeltable as pxt
from tool.random_ops import OpStatus, RandomTableOps, RandomTableOpsConfig, is_expected

from ..utils import pxt_raises

pytestmark = pytest.mark.db_roots('local', reason='random_ops addresses tables by plain names')


@pytest.fixture
def ops(uses_db: None) -> Iterator[RandomTableOps]:
    """A RandomTableOps over the single base table tbl_0; restores the logging setup that its constructor changes."""
    loggers = [logging.getLogger(name) for name in ('pixeltable', 'pixeltable.random_ops', 'sqlalchemy.engine')]
    saved = [(logger, logger.level, list(logger.handlers), logger.propagate) for logger in loggers]
    yield RandomTableOps(0, False, [], [], RandomTableOpsConfig(num_base_tables=1))
    for logger, level, handlers, propagate in saved:
        for handler in logger.handlers[:]:
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(level)
        logger.propagate = propagate


class TestRandomOps:
    def test_drop_column_errors(self, uses_db: None) -> None:
        """A drop that falls through to a base table column is expected; a drop blocked by dependents is not."""
        t = pxt.create_table('tbl', {'c1': pxt.Int})
        v = pxt.create_view('view', t)
        v.add_column(c3=pxt.Int | None)
        t.add_column(c3=pxt.Int | None)
        v.drop_column('c3')  # another worker drops the view's column first
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Cannot drop base table column 'c3'") as info:
            v.drop_column('c3', if_not_exists='ignore')
        assert is_expected(info.value)[0]

        t.add_computed_column(c4=t.c3 + 1)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='following columns depend on it') as info:
            t.drop_column('c3')
        assert not is_expected(info.value)[0]

    def test_drops_succeed(self, ops: RandomTableOps) -> None:
        t = ops.get_random_tbl(allow_view=False)
        assert t is not None
        pxt.create_view('view_0', t)

        result = ops.drop_view()
        assert result.status == OpStatus.SUCCESS, result
        assert result.msg.startswith("Dropped view 'view_0'"), result

        result = ops.drop_table()
        assert result.status == OpStatus.SUCCESS, result
        assert result.msg.startswith("Dropped table 'tbl_0'"), result
