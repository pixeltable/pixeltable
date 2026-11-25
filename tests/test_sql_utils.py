import logging

import pytest
import sqlalchemy as sql

from pixeltable.env import Env
from pixeltable.utils.sql import log_explain


@pytest.mark.db_roots('local', reason='runs EXPLAIN on the local Postgres engine')
def test_log_explain(uses_db: None, caplog: pytest.LogCaptureFixture) -> None:
    # the generic dialect renders CAST(... AS DOUBLE), which Postgres rejects; the engine dialect renders % as %%
    x = sql.cast(sql.literal(7.5), sql.Double)
    stmt = sql.select((x / 2.0).label('d'), (sql.literal(7) % 3).label('m'), sql.literal('a%b').like('a%').label('l'))
    logger = logging.getLogger('test_log_explain')
    with Env.get().engine.begin() as conn, caplog.at_level(logging.DEBUG, logger=logger.name):
        log_explain(logger, stmt, conn)
        # a failed EXPLAIN would abort the transaction for every later statement
        assert conn.execute(sql.select(sql.literal(1))).scalar() == 1
    assert 'EXPLAIN failed' not in caplog.text
    assert 'SqlScanNode explain' in caplog.text
