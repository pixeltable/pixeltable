"""The query a model declares, before that model is bound to a table."""

from __future__ import annotations

import dataclasses
from typing import Any, Self

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, exprs
from pixeltable._query_base import QueryBase
from pixeltable.exprs import ColumnRefByName
from pixeltable.query_clauses import FromClause

from .declaration import MODEL_BY_DECLARED_TBL_ID, TableModelMeta


class ModelQuery(QueryBase):
    """A query declared over a model, before that model is bound to a table.

    Its from-clause is the shape the model declares, so it carries every clause a query can, but nothing
    that runs one: the columns it references belong to a table that does not exist yet. bind() produces the
    equivalent query over a real table.
    """

    @property
    def model_cls(self) -> TableModelMeta:
        """The model whose declared shape this query is written against."""
        model_cls = MODEL_BY_DECLARED_TBL_ID.get(self._from_clause.tbls[0].tbl_id)
        assert model_cls is not None, self._from_clause.tbls[0].tbl_id
        return model_cls

    @classmethod
    def for_model(cls, model_cls: TableModelMeta) -> ModelQuery:
        """A query over everything model_cls declares."""
        return cls(from_clause=FromClause(tbls=[model_cls.table_path()]))

    def as_dict(self) -> dict[str, Any]:
        """Refuses: a query over a model has no serialized form.

        The shape it queries is synthesized from the model and carries no durable table identity, so a
        serialized one could not be read back. bind() produces the query that can be stored.
        """
        raise excs.Error(
            excs.ErrorCode.INTERNAL_ERROR,
            f'A query over model `{self.model_cls.__name__}` cannot be serialized; bind it to a table first.',
        )

    @property
    def _effective_select_list(self) -> list[tuple[exprs.Expr, str]]:
        """The select list with an implicit select * expanded, as references to the model's columns by name."""
        if self.select_list is not None:
            return super()._effective_select_list
        declared_path = self._from_clause.tbls[0]
        return [
            (ColumnRefByName(col_md.name, col_md.col_type), col_md.name)
            for col_md in declared_path.column_md()
            if col_md.name is not None
        ]

    def referenced_column_names(self) -> set[str]:
        """The names of the model columns this query references, across all of its clauses."""
        all_exprs = [*(e for e, _ in self._effective_select_list), *self._component_exprs()]
        result = {e.name for expr in all_exprs for e in expr.subexprs(ColumnRefByName)}
        # a similarity expression identifies the column it is indexed on, rather than referencing it by name
        declared_path = self._from_clause.tbls[0]
        for sim in (s for expr in all_exprs for s in expr.subexprs(exprs.SimilarityExpr)):
            assert sim.qcol_id is not None
            col_name = declared_path.get_column_md(sim.qcol_id).name
            assert col_name is not None
            result.add(col_name)
        return result

    def validate(self, model_name: str) -> None:
        """Validate that this query can be used to define a view."""
        from ..view import View

        View.validate_view_query(self, prefix=f'{model_name}: ')

        # a view model turns each select() item into a class attribute, so every item needs a name
        if self.select_list is None:
            return
        for item, name in self.select_list:
            if name is None and not item.is_column_ref:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'{model_name}: `base` select() list may contain only direct column references '
                    f'or named expressions, but contains an anonymous compound expression: {item}\n'
                    f'Use kwargs syntax to give it an explicit name: select(my_name=...)',
                )

    def to_declared_query(self) -> pxt.Query:
        """The equivalent query whose column references identify the columns of the model's declared shape.

        Metadata assembly distinguishes a column reference from a computed expression, which a query that only
        names its columns cannot support.
        """
        declared_path = self._from_clause.tbls[0]
        subst: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
        for col_md in declared_path.column_md():
            if col_md.name is not None:
                subst[ColumnRefByName(col_md.name)] = exprs.ColumnRef(col_md)
        return self._substituted(declared_path, subst)

    def bind(self, catalog_dir: str) -> pxt.Query:
        """The equivalent query over the table this query's model resolves to under catalog_dir."""
        tbl = self.model_cls._bind(catalog_dir)
        subst: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
        for col_name in tbl.columns():
            subst[ColumnRefByName(col_name)] = getattr(tbl, col_name)
        return self._substituted(tbl._tbl_path, subst)

    def _substituted(self, path: catalog.TablePath, subst: exprs.ExprDict[exprs.Expr]) -> pxt.Query:
        """A plain Query over path, with this query's clauses rewritten by subst."""
        # a similarity expression names its indexed column and the table version holding the index, neither of
        # which a substitution by column name reaches
        declared_path = self._from_clause.tbls[0]
        for sim in {s.id: s for e in self._component_exprs() for s in e.subexprs(exprs.SimilarityExpr)}.values():
            assert sim.qcol_id is not None
            if sim.qcol_id.tbl_id == path.tbl_id:
                continue  # already indexed against this path
            col_name = declared_path.get_column_md(sim.qcol_id).name
            assert col_name is not None
            new_md = path.get_column_md_by_name(col_name)
            if new_md is None:
                raise excs.RequestError(
                    excs.ErrorCode.COLUMN_NOT_FOUND,
                    f'Table {path.tbl_name()!r} has no column {col_name!r}, which a similarity() call references.',
                )
            subst[sim] = exprs.SimilarityExpr(
                sim.components[0].copy().substitute(subst),
                idx_name=sim.idx_name,
                qcol_id=new_md.qcolid,
                table_version_key=catalog.TableVersionKey(new_md.qcolid.tbl_id, new_md.col_effective_version),
            )

        def rebound(e: exprs.Expr) -> exprs.Expr:
            return e.copy().substitute(subst)

        return pxt.Query(
            from_clause=FromClause(tbls=[path]),
            select_list=None if self.select_list is None else [(rebound(e), n) for e, n in self.select_list],
            where_clause=None if self.where_clause is None else rebound(self.where_clause),
            group_by_clause=None if self.group_by_clause is None else [rebound(e) for e in self.group_by_clause],
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=None
            if self.order_by_clause is None
            else [(rebound(e), asc) for e, asc in self.order_by_clause],
            limit=None if self.limit_val is None else rebound(self.limit_val),
            offset=None if self.offset_val is None else rebound(self.offset_val),
            sample_clause=None
            if self.sample_clause is None
            else dataclasses.replace(
                self.sample_clause, stratify_exprs=[rebound(e) for e in self.sample_clause.stratify_exprs]
            ),
        )

    def join(self, *args: Any, **kwargs: Any) -> Self:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'join(): a query over model `{self.model_cls.__name__}` cannot be joined; '
            'join the tables the models are bound to instead.',
        )
