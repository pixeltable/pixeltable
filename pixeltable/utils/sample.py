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
    fract: Optional[float]
    seed: Optional[int]
    stratify_list: Optional[list[Expr]]

    CURRENT_VERSION = 1

    def __post_init__(self) -> None:
        """If no version was provided, provide the default version"""
        if self._version is None:
            self.version = self.CURRENT_VERSION

    @property
    def _version(self) -> int:
        return self.version

    @property
    def _n(self) -> Optional[int]:
        return self.n

    @property
    def _n_per_stratum(self) -> Optional[int]:
        return self.n_per_stratum

    @property
    def _fraction(self) -> Optional[float]:
        return self.fract

    @property
    def _seed(self) -> Optional[int]:
        return self.seed

    @property
    def _stratify_list(self) -> list[Expr]:
        return self.stratify_list

    @property
    def is_stratified(self) -> bool:
        """Check if the sampling is stratified"""
        return self._stratify_list is not None and len(self._stratify_list) > 0

    def display_str(self, inline: bool = False) -> str:
        return str(self)

    def as_dict(self) -> dict:
        """Return a dictionary representation of the object"""
        d = dataclasses.asdict(self)
        d['_classname'] = self.__class__.__name__
        if self.is_stratified:
            d['stratify_list'] = [e.as_dict() for e in self._stratify_list]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SampleClause:
        """Create a SampleClause from a dictionary representation"""
        d_cleaned = {key: value for key, value in d.items() if key != '_classname'}
        s = cls(**d_cleaned)
        if s.is_stratified:
            s.stratify_list = [Expr.from_dict(e) for e in d_cleaned.get('stratify_list', [])]
        return s

    def __repr__(self) -> str:
        s = ','.join(e.display_str(inline=True) for e in self._stratify_list)
        return (
            f'sample_{self._version}(n={self._n}, n_per_stratum={self._n_per_stratum}, '
            + f'fraction={self._fraction}, seed={self._seed}, [{s}])'
        )

    @classmethod
    def fraction_to_md5_hex(cls, fraction: float) -> str:
        # Maximum count for the upper 32 bits of MD5: 2^32
        max_md5_value = (2**32) - 1

        # Calculate the fraction of this value
        threshold_int = max_md5_value * int(1_000_000_000 * fraction) // 1_000_000_000

        # Convert to hexadecimal string with padding
        return format(threshold_int, '08x') + 'ffffffffffffffffffffffff'


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
        seed_text = "'" + str(self._seed) + "'" if self._seed is not None else "''"
        sql_expr: sql.ColumnElement = sql.cast(sql.literal_column(seed_text), sql.Text)
        for e in self.components[1:]:
            sql_expr = sql_expr + sql.literal_column("'___'") + sql.cast(e.sql_expr(sql_elements), sql.Text)
        sql_expr = sql.func.md5(sql_expr)
        return sql_expr
