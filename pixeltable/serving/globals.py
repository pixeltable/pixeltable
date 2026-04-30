"""Shared serving-layer types: SqlExport spec + per-route SqlExporter."""

from __future__ import annotations

import logging
from typing import Literal

import pydantic
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
import pixeltable.utils.sql as sql_utils

_logger = logging.getLogger('pixeltable')


class SqlExport(pydantic.BaseModel):
    """
    Specification of an external RDBMS target for SQL export.

    Attributes:
        db_connect: SQLAlchemy connection string for the target database (e.g.
            `'postgresql+psycopg://user:pw@host/db'`, `'sqlite:///path/to.db'`).
        table: Name of the target table. It must already exist; resolution fails
            if the table is missing.
        db_schema: Optional database schema qualifier (e.g. `'analytics'`); leave `None` to
            use the connection's default schema.
        method: How to write each row into the target table.

            - `'insert'`: append the row via `INSERT ... VALUES`.
            - `'update'`: update the row by primary-key match
              (`UPDATE ... SET ... WHERE pk=...`). Requires that the target table has a
              primary key whose metadata is exposed by the dialect. The exported columns
              must include all primary-key columns of the target plus at least one non-PK
              column to set. This is a strict update, **not** an upsert: if the WHERE
              clause matches zero rows, the export fails. Useful when the source is
              append-only but the target is a deduplicated current-state view.
            - `'merge'`: upsert via the target table's primary key.
              **Currently not supported.**
    """

    model_config = pydantic.ConfigDict(extra='forbid')

    db_connect: str
    table: str
    db_schema: str | None = None
    method: Literal['insert', 'update', 'merge'] = 'insert'


class SqlExporter:
    """Per-route SQL export state.

    Validates the spec and resolves the target table at construction time (called once per route at
    registration); used per request via export_row().
    """

    engine: sql.Engine
    table: sql.Table
    method: Literal['insert', 'update']

    # cached SQLAlchemy construct: the engine's compiled-SQL cache hits on every reuse, so per-request
    # cost is bind + send, no recompile.
    _stmt: sql.sql.expression.Executable

    # primary-key column names for method='update'; used to translate row dict to WHERE bindparams
    _pk_names: list[str]

    def __init__(
        self, spec: SqlExport, *, engine: sql.Engine, output_schema: dict[str, ts.ColumnType], error_prefix: str
    ) -> None:
        if spec.method == 'merge':
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f"{error_prefix}: 'merge' is not yet supported"
            )
        # response-body form: media columns are URL strings on the wire
        src_schema = {
            col_name: ts.StringType(nullable=col_type.nullable) if col_type.is_media_type() else col_type
            for col_name, col_type in output_schema.items()
        }
        self.engine = engine
        self.table = sql_utils.resolve_table(
            engine=engine,
            table_name=spec.table,
            schema_name=spec.db_schema,
            source_schema=src_schema,
            if_exists='insert',
            if_not_exists='error',
            error_prefix=error_prefix,
        )
        # 'merge' rejected above; remaining values are 'insert' | 'update'
        self.method = spec.method

        if spec.method == 'update':
            # require a primary key on the target; the response columns must include every PK column
            # so we can build the WHERE clause, plus at least one non-PK column to set
            pk_names = [col.name for col in self.table.primary_key.columns]
            if not pk_names:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'{error_prefix}: target table {self.table.name!r} has no primary key; '
                    f"required for method='update' (note: requires the target dialect's metadata "
                    f'to expose the primary key)',
                )
            missing_pk = [n for n in pk_names if n not in output_schema]
            if missing_pk:
                raise excs.RequestError(
                    excs.ErrorCode.MISSING_REQUIRED,
                    f"{error_prefix}: response columns must include the target table's primary-key "
                    f'columns (missing: {missing_pk}). Add them to `outputs`.',
                )
            non_pk = [n for n in output_schema if n not in pk_names]
            if not non_pk:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f"{error_prefix}: method='update' requires at least one non-primary-key column "
                    f'in the response columns (got only {pk_names!r})',
                )
            # SQLAlchemy reserves bare column names for SET-clause auto-binds, so the WHERE bindparam
            # uses a 'b_' prefix; we translate the row dict at execute time
            where = sql.and_(*(self.table.c[n] == sql.bindparam(f'b_{n}') for n in pk_names))
            self._stmt = self.table.update().where(where).values({n: sql.bindparam(n) for n in non_pk})
            self._pk_names = pk_names
        else:
            self._stmt = self.table.insert()
            self._pk_names = []

    def export_row(self, row: pydantic.BaseModel) -> None:
        """Insert or update one response-body row, per spec.method.

        Raises HTTPException(500) on SQL failure.
        For method='update', also raises HTTPException(500) if the WHERE clause matched zero rows
        """
        from fastapi import HTTPException

        params = row.model_dump(mode='python')
        if self.method == 'update':
            # add prefixed copies of PK values for the WHERE bindparams
            params |= {f'b_{n}': params[n] for n in self._pk_names}
        try:
            with self.engine.connect() as conn:
                result = conn.execute(self._stmt, [params])
                if self.method == 'update' and result.rowcount != 1:
                    raise HTTPException(
                        status_code=500, detail=f'export_sql update affected {result.rowcount} rows; expected 1'
                    )
                conn.commit()
        except HTTPException:
            raise
        except Exception as e:
            # log full diagnostics server-side; client gets a generic message so we don't leak SQL
            # text, parameter values, or driver internals carried by DBAPI/SQLAlchemy exceptions
            _logger.exception('export_sql write to %s failed', self.table.name)
            raise HTTPException(status_code=500, detail='export_sql write failed') from e
