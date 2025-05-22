from __future__ import annotations

import dataclasses
from typing import Optional

import sqlalchemy as sql

from pixeltable import type_system as ts
from pixeltable.exprs.expr import DataRow, Expr
from pixeltable.exprs.literal import Literal
from pixeltable.exprs.row_builder import RowBuilder
from pixeltable.exprs.sql_element_cache import SqlElementCache


@dataclasses.dataclass
class SampleClause:
    """Defines a sampling clause for a table."""

    version: Optional[int]
    n: Optional[int]
    n_per_stratum: Optional[int]
    fraction: Optional[float]
    seed: Optional[int]
    stratify_exprs: Optional[list[Expr]]

    DEFAULT_SEED = 0
    CURRENT_VERSION = 1

    def __post_init__(self) -> None:
        """If no version was provided, provide the default version"""
        if self.version is None:
            self.version = self.CURRENT_VERSION
        if self.seed is None:
            self.seed = self.DEFAULT_SEED

    @property
    def is_stratified(self) -> bool:
        """Check if the sampling is stratified"""
        return self.stratify_exprs is not None and len(self.stratify_exprs) > 0

    @property
    def is_repeatable(self) -> bool:
        """Return true if the same rows will continue to be sampled if source rows are added or deleted."""
        return not self.is_stratified and self.fraction is not None

    def display_str(self, inline: bool = False) -> str:
        return str(self)

    def as_dict(self) -> dict:
        """Return a dictionary representation of the object"""
        d = dataclasses.asdict(self)
        d['_classname'] = self.__class__.__name__
        if self.is_stratified:
            d['stratify_exprs'] = [e.as_dict() for e in self.stratify_exprs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SampleClause:
        """Create a SampleClause from a dictionary representation"""
        d_cleaned = {key: value for key, value in d.items() if key != '_classname'}
        s = cls(**d_cleaned)
        if s.is_stratified:
            s.stratify_exprs = [Expr.from_dict(e) for e in d_cleaned.get('stratify_exprs', [])]
        return s

    def __repr__(self) -> str:
        s = ','.join(e.display_str(inline=True) for e in self.stratify_exprs)
        return (
            f'sample_{self.version}(n={self.n}, n_per_stratum={self.n_per_stratum}, '
            + f'fraction={self.fraction}, seed={self.seed}, [{s}])'
        )

    @classmethod
    def fraction_to_md5_hex(cls, fraction: float) -> str:
        # Maximum count for the upper 32 bits of MD5: 2^32
        max_md5_value = (2**32) - 1

        # Calculate the fraction of this value
        threshold_int = max_md5_value * int(1_000_000_000 * fraction) // 1_000_000_000

        # Convert to hexadecimal string with padding
        return format(threshold_int, '08x') + 'ffffffffffffffffffffffff'

    @classmethod
    def key_sql_expr(
        cls, sql_elements: SqlElementCache, seed: int, sql_cols: list[sql.KeyedColumnElement]
    ) -> sql.ColumnElement:
        """Construct expression which is the ordering key for rows to be sampled
        General SQL form is:
        - MD5('<seed::text>' [ + '___' + <rowid_col_val>::text]+
        """
        seed_text = "'" + str(seed) + "'"
        sql_expr: sql.ColumnElement = sql.cast(sql.literal_column(seed_text), sql.Text)
        for e in sql_cols:
            sql_expr = sql_expr + sql.literal_column("'___'") + sql.cast(e, sql.Text)
        sql_expr = sql.func.md5(sql_expr)
        return sql_expr


class SampleKey(Expr):
    """
    An `Expr` that computes a custom expression from the RowidRef's required to uniquely access a table
    """

    def __init__(self, seed_expr: Expr, rowids: list[Expr]):
        super().__init__(ts.StringType(nullable=True))
        self.components = [seed_expr, *rowids]
        self.id: Optional[int] = self._create_id()

    @property
    def _seed(self) -> Optional[int]:
        assert isinstance(self._seed_expr.val, (int, type(None)))
        return self._seed_expr.val

    @property
    def _seed_expr(self) -> Expr:
        return self.components[0]

    def _equals(self, other: SampleKey) -> bool:
        return True

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise NotImplementedError

    def as_literal(self) -> Optional[Literal]:
        return None

    def _as_dict(self) -> dict:
        return {**super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> SampleKey:
        assert len(components) >= 2
        return cls(components[0], components[1:])

    def __repr__(self) -> str:
        return ''

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        """Construct expression which is the ordering key for rows to be sampled
        General SQL form is:
        - MD5('<seed::text>' [ + '___' + <rowid_col_val>::text]+
        """
        rowid_sql_expr = [e.sql_expr(sql_elements) for e in self.components[1:]]
        return SampleClause.key_sql_expr(sql_elements, self._seed, rowid_sql_expr)
