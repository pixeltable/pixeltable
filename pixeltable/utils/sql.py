import logging

import sqlalchemy as sql
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import URL


def log_stmt(logger: logging.Logger, stmt: sql.sql.ClauseElement) -> None:
    logger.debug(f'executing {stmt.compile(dialect=postgresql.dialect())}')


def log_explain(logger: logging.Logger, stmt: sql.sql.ClauseElement, conn: sql.engine.Connection) -> None:
    try:
        # don't set dialect=Env.get().engine.dialect: x % y turns into x %% y, which results in a syntax error
        stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
        explain_result = conn.execute(sql.text(f'EXPLAIN {stmt_str}'))
        explain_str = '\n'.join(str(row) for row in explain_result)
        logger.debug(f'SqlScanNode explain:\n{explain_str}')
    except Exception:
        logger.warning('EXPLAIN failed')


def add_option_to_db_url(url: str | URL, option: str) -> URL:
    """Add a connection option to a database URL.

    Args:
        url: Database URL as string or SQLAlchemy URL object
        option: Option to add (e.g., '-c search_path=test_schema,public' or '-c timezone=UTC')

    Returns:
        Modified URL object with the option added to the query parameters
    """
    db_url = sql.make_url(url) if isinstance(url, str) else url

    # Get existing options and parse them
    # Query parameters can be strings or tuples (if multiple values exist)
    existing_options_raw = db_url.query.get('options', '') if db_url.query else ''
    option_parts = (
        list(existing_options_raw) if isinstance(existing_options_raw, tuple) else existing_options_raw.split()
    )
    option_parts.append(option)
    options_str = ' '.join(option_parts)

    # Create new URL with updated options
    return db_url.set(query={**(dict(db_url.query) if db_url.query else {}), 'options': options_str})
