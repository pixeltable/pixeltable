import logging

import sqlalchemy as sql


def log_stmt(logger: logging.Logger, stmt) -> str:
    logger.debug(f'executing {str(stmt.compile(dialect=sql.dialects.postgresql.dialect()))}')