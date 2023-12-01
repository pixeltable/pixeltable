import logging

import sqlalchemy as sql


def log_stmt(logger: logging.Logger, stmt) -> None:
    logger.debug(f'executing {str(stmt.compile(dialect=sql.dialects.postgresql.dialect()))}')

def log_explain(logger: logging.Logger, stmt: sql.sql.ClauseElement, conn: sql.engine.Connection) -> None:
    try:
        # don't set dialect=Env.get().engine.dialect: x % y turns into x %% y, which results in a syntax error
        stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
        explain_result = conn.execute(sql.text(f'EXPLAIN {stmt_str}'))
        explain_str = '\n'.join([str(row) for row in explain_result])
        logger.debug(f'SqlScanNode explain:\n{explain_str}')
    except Exception as e:
        logger.warning(f'EXPLAIN failed')
