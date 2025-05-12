from __future__ import annotations

from typing import Optional

import sqlalchemy as sql

from pixeltable import exprs, type_system as ts
from pixeltable.exprs.expr import DataRow, Expr
from pixeltable.exprs.literal import Literal
from pixeltable.exprs.row_builder import RowBuilder
from pixeltable.exprs.sql_element_cache import SqlElementCache


class SampleClause(Expr):
    """
    An `Expr` that computes a sample of the associated table.
    """

    CURRENT_VERSION = 1

    def __init__(
        self,
        version: Optional[int],
        n: Optional[int],
        n_per_stratum: Optional[int],
        fract: Optional[float],
        seed: Optional[int],
        stratify_list: Optional[list[Expr]],
    ):
        super().__init__(ts.StringType(nullable=True))
        self.components = [
            exprs.Literal(version),
            exprs.Literal(n),
            exprs.Literal(n_per_stratum),
            exprs.Literal(fract),
            exprs.Literal(seed),
            *stratify_list,
        ]
        self.__post_init__()

    def __post_init__(self) -> None:
        """If no version was provided, provide the default version"""
        if self.components[0].val is None:
            self.components[0] = exprs.Literal(self.CURRENT_VERSION)
            self.id: Optional[int] = self._create_id()

    def _equals(self, other: SampleClause) -> bool:
        return True

    @property
    def _version(self) -> int:
        v = self.components[0].val
        assert isinstance(v, int)
        return v

    @property
    def _n(self) -> Optional[int]:
        assert isinstance(self._n_expr.val, (int, type(None)))
        return self._n_expr.val

    @property
    def _n_per_stratum(self) -> Optional[int]:
        assert isinstance(self._n_per_stratum_expr.val, int) or self._n_per_stratum_expr.val is None
        return self._n_per_stratum_expr.val

    @property
    def _fraction(self) -> Optional[float]:
        assert isinstance(self._fraction_expr.val, float) or self._fraction_expr.val is None
        return self._fraction_expr.val

    @property
    def _seed(self) -> Optional[int]:
        assert isinstance(self._seed_expr.val, (int, type(None)))
        return self._seed_expr.val

    @property
    def _n_expr(self) -> Expr:
        return self.components[1]

    @property
    def _n_per_stratum_expr(self) -> Optional[Expr]:
        return self.components[2]

    @property
    def _fraction_expr(self) -> Expr:
        return self.components[3]

    @property
    def _seed_expr(self) -> Expr:
        return self.components[4]

    @property
    def _stratify_list(self) -> list[Expr]:
        return self.components[5:]

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        raise NotImplementedError

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        raise NotImplementedError

    def as_literal(self) -> Optional[Literal]:
        return None

    def _as_dict(self) -> dict:
        return {**super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> SampleClause:
        return SampleClause(
            int(components[0].val),
            int(components[1].val),
            int(components[2].val),
            float(components[3].val),
            int(components[4].val),
            components[5:],
        )

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
