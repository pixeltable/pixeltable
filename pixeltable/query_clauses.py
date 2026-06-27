from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import pixeltable.exceptions as excs

if TYPE_CHECKING:
    import pixeltable.catalog as catalog
    import pixeltable.exprs as exprs


class JoinType(enum.Enum):
    INNER = 0
    LEFT = 1
    # TODO: implement
    # RIGHT = 2
    FULL_OUTER = 3
    CROSS = 4

    LiteralType = Literal['inner', 'left', 'full_outer', 'cross']

    @classmethod
    def validated(cls, name: str, error_prefix: str) -> JoinType:
        try:
            return cls[name.upper()]
        except KeyError as exc:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{error_prefix} must be one of: [{val_strs}]'
            ) from exc


@dataclasses.dataclass
class JoinClause:
    """Corresponds to a single 'JOIN ... ON (...)' clause in a SELECT statement; excludes the joined table."""

    join_type: JoinType
    join_predicate: exprs.Expr | None  # None for join_type == CROSS


@dataclasses.dataclass
class FromClause:
    """Corresponds to the FROM clause ('FROM <tbl> JOIN ... ON (...) JOIN ...') of a SELECT statement."""

    tbls: list[catalog.TablePath]
    join_clauses: list[JoinClause] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        from pixeltable.catalog import TableVersionPath

        assert len(self.tbls) > 0
        first_type = type(self.tbls[0])
        assert all(type(t) is first_type for t in self.tbls), (
            'cannot mix TableMdPath and TableVersionPath in a FromClause'
        )
        if not isinstance(self.tbls[0], TableVersionPath):
            # all hosted: every table must resolve to the same catalog (no cross-catalog joins)
            uris = {cast('catalog.TableMdPath', t).catalog_uri for t in self.tbls}
            if len(uris) > 1:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'all tables in a query must be in the same catalog; got {sorted(str(u) for u in uris)}',
                )

    @property
    def tvps(self) -> list[catalog.TableVersionPath]:
        # string type: catalog is TYPE_CHECKING-only (a runtime import here would be circular), and cast()
        # ignores its type arg at runtime anyway
        return cast('list[catalog.TableVersionPath]', self.tbls)

    @property
    def is_local(self) -> bool:
        from pixeltable.catalog import TableVersionPath

        return isinstance(self.tbls[0], TableVersionPath)

    @property
    def catalog_uri(self) -> catalog.Path:
        """The catalog all tables resolve to."""
        from pixeltable.catalog import ROOT_PATH

        if self.is_local:
            return ROOT_PATH
        return cast('catalog.TableMdPath', self.tbls[0]).catalog_uri

    @property
    def _first_tbl(self) -> catalog.TablePath:
        assert len(self.tbls) == 1
        return self.tbls[0]


@dataclasses.dataclass
class SampleClause:
    """Defines a sampling clause for a table."""

    version: int | None
    n: int | None
    n_per_stratum: int | None
    fraction: float | None
    seed: int | None
    stratify_exprs: list[exprs.Expr] | None

    CURRENT_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.version is None:
            self.version = self.CURRENT_VERSION

    @property
    def is_stratified(self) -> bool:
        return self.stratify_exprs is not None and len(self.stratify_exprs) > 0

    @property
    def is_repeatable(self) -> bool:
        return not self.is_stratified and self.fraction is not None

    def display_str(self, inline: bool = False) -> str:
        return str(self)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d['_classname'] = self.__class__.__name__
        if self.is_stratified:
            d['stratify_exprs'] = [e.as_dict() for e in self.stratify_exprs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SampleClause:
        from pixeltable import exprs as _exprs

        d_cleaned = {key: value for key, value in d.items() if key != '_classname'}
        s = cls(**d_cleaned)
        if s.is_stratified:
            s.stratify_exprs = [_exprs.Expr.from_dict(e) for e in d_cleaned.get('stratify_exprs', [])]
        return s

    def __repr__(self) -> str:
        s = ','.join(e.display_str(inline=True) for e in self.stratify_exprs)
        return (
            f'sample_{self.version}(n={self.n}, n_per_stratum={self.n_per_stratum}, '
            f'fraction={self.fraction}, seed={self.seed}, [{s}])'
        )

    @classmethod
    def fraction_to_md5_hex(cls, fraction: float) -> str:
        """Return the string representation of an approximation (to ~1e-9) of a fraction of the total space
        of md5 hash values. Used for fractional sampling.
        """
        max_md5_value = (2**32) - 1
        threshold_int = max_md5_value * int(1_000_000_000 * fraction) // 1_000_000_000
        return format(threshold_int, '08x') + 'ffffffffffffffffffffffff'
