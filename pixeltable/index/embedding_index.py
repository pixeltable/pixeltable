from __future__ import annotations

import enum
from typing import Any, ClassVar, Literal

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import sqlalchemy as sql
from sqlalchemy import cast

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable.env import Env

from .base import IndexBase

MAX_EMBEDDING_VECTOR_LENGTH = 2000
MAX_EMBEDDING_HALFVEC_LENGTH = 4000


class EmbeddingIndex(IndexBase):
    """
    Interface to the pgvector access method in Postgres.
    - pgvector converts the cosine metric to '1 - metric' and the inner product metric to '-metric', in order to
      satisfy the Postgres requirement that an index scan requires an ORDER BY ... ASC clause
    - similarity_clause() converts those metrics back to their original form; it is used in expressions outside
      the Order By clause
    - order_by_clause() is used exclusively in the ORDER BY clause
    - embedding function parameters are named '<type-name>_embed', where type-name is ColumnType.Type.name
    """

    class Metric(enum.Enum):
        COSINE = 1
        IP = 2
        L2 = 3

    class Precision(enum.Enum):
        FP16 = 'fp16'
        FP32 = 'fp32'

    PGVECTOR_OPS: ClassVar[dict[Metric, str]] = {
        Metric.COSINE: 'vector_cosine_ops',
        Metric.IP: 'vector_ip_ops',
        Metric.L2: 'vector_l2_ops',
    }
    HALFVEC_OPS: ClassVar[dict[Metric, str]] = {
        Metric.COSINE: 'halfvec_cosine_ops',
        Metric.IP: 'halfvec_ip_ops',
        Metric.L2: 'halfvec_l2_ops',
    }

    metric: Metric
    embeddings: dict[ts.ColumnType.Type, func.Function]
    precision: Precision

    def __init__(
        self,
        metric: str,
        precision: Literal['fp16', 'fp32'],
        embed: func.Function | None = None,
        string_embed: func.Function | None = None,
        image_embed: func.Function | None = None,
        audio_embed: func.Function | None = None,
        video_embed: func.Function | None = None,
    ):
        if embed is None and string_embed is None and image_embed is None:
            raise excs.Error('At least one of `embed`, `string_embed`, or `image_embed` must be specified')
        metric_names = [m.name.lower() for m in self.Metric]
        if metric.lower() not in metric_names:
            raise excs.Error(f'Invalid metric {metric}, must be one of {metric_names}')

        self.embeddings = {}

        # Resolve the specific embedding functions corresponding to the user-provided embedding functions.
        # For string embeddings, for example, `string_embed` will be used if specified; otherwise, `embed` will
        # be used as a fallback, if it has a matching signature.

        for embed_type, embed_fn in (
            (ts.ColumnType.Type.STRING, string_embed),
            (ts.ColumnType.Type.IMAGE, image_embed),
            (ts.ColumnType.Type.AUDIO, audio_embed),
            (ts.ColumnType.Type.VIDEO, video_embed),
        ):
            if embed_fn is not None:
                # Embedding function for the requisite type is specified directly; it MUST be valid.
                resolved_fn = self._resolve_embedding_fn(embed_fn, embed_type)
                if resolved_fn is None:
                    raise excs.Error(
                        f'The function `{embed_fn.name}` is not a valid {embed_type.name.lower()} '
                        f'embedding: it must take a single {embed_type.name.lower()} parameter'
                    )
                self.embeddings[embed_type] = resolved_fn
            elif embed is not None:
                # General `embed` is specified; see if it has a matching signature.
                resolved_fn = self._resolve_embedding_fn(embed, embed_type)
                if resolved_fn is not None:
                    self.embeddings[embed_type] = resolved_fn

        if len(self.embeddings) == 0:
            # `embed` was specified and contains no matching signatures.
            assert embed is not None
            raise excs.Error(
                f'The function `{embed.name}` is not a valid embedding: '
                'it must take a single string, image, audio, or video parameter'
            )

        # Now validate the return types of the embedding functions.
        for _, embed_fn in self.embeddings.items():
            self._validate_embedding_fn(embed_fn)

        self.metric = self.Metric[metric.upper()]
        try:
            self.precision = self.Precision(precision)
        except ValueError:
            valid_values = [p.value for p in self.Precision]
            raise excs.Error(f"Invalid precision '{precision}'. Must be one of: {valid_values}") from None

    def create_value_expr(self, c: catalog.Column) -> exprs.Expr:
        if c.col_type._type not in (
            ts.ColumnType.Type.STRING,
            ts.ColumnType.Type.IMAGE,
            ts.ColumnType.Type.AUDIO,
            ts.ColumnType.Type.VIDEO,
        ):
            raise excs.Error(f'Type `{c.col_type}` of column {c.name!r} is not a valid type for an embedding index.')
        if c.col_type._type not in self.embeddings:
            raise excs.Error(
                f'The specified embedding function does not support the type `{c.col_type}` of column {c.name!r}.'
            )

        embed_fn = self.embeddings[c.col_type._type]
        return embed_fn(exprs.ColumnRef(c))

    def records_value_errors(self) -> bool:
        return True

    def get_index_sa_type(self, val_col_type: ts.ColumnType) -> sql.types.TypeEngine:
        assert isinstance(val_col_type, ts.ArrayType) and val_col_type.shape is not None
        assert len(val_col_type.shape) == 1
        vector_length = val_col_type.shape[0]
        assert vector_length is not None
        assert vector_length > 0

        # TODO(PXT-941): Revisit embedding index precision behavior for cloud launch
        # CockroachDB doesn't have HALFVEC. For now, always use Vector type.
        if Env.get().is_using_cockroachdb:
            return pgvector.sqlalchemy.Vector(vector_length)

        match self.precision:
            case self.Precision.FP32:
                if vector_length > MAX_EMBEDDING_VECTOR_LENGTH:
                    raise excs.Error(
                        f"Embedding index's vector dimensionality {vector_length} exceeds maximum of"
                        f' {MAX_EMBEDDING_VECTOR_LENGTH} for {self.precision.value} precision'
                    )
                return pgvector.sqlalchemy.Vector(vector_length)
            case self.Precision.FP16:
                if vector_length > MAX_EMBEDDING_HALFVEC_LENGTH:
                    raise excs.Error(
                        f"Embedding index's vector dimensionality {vector_length} exceeds maximum of"
                        f' {MAX_EMBEDDING_HALFVEC_LENGTH} for {self.precision.value} precision'
                    )
                return pgvector.sqlalchemy.HALFVEC(vector_length)
            case _:
                raise AssertionError(self.precision)

    def sa_create_stmt(self, store_index_name: str, sa_value_col: sql.Column) -> sql.Compiled:
        """Return a sqlalchemy statement for creating the index"""
        if isinstance(sa_value_col.type, pgvector.sqlalchemy.Vector):
            metric = self.PGVECTOR_OPS[self.metric]
        elif isinstance(sa_value_col.type, pgvector.sqlalchemy.HALFVEC):
            metric = self.HALFVEC_OPS[self.metric]
        else:
            raise AssertionError(f'Unsupported index column type: {sa_value_col.type}')
        stmt = Env.get().dbms.create_vector_index_stmt(store_index_name, sa_value_col, metric=metric)
        return stmt

    def drop_index(self, index_name: str, index_value_col: catalog.Column) -> None:
        """Drop the index on the index value column"""
        # TODO: implement
        raise NotImplementedError()

    def similarity_clause(self, val_column: catalog.Column, item: exprs.Literal) -> sql.ColumnElement:
        """Create a ColumnElement that represents '<val_column> <op> <item>'"""
        assert item.col_type._type in self.embeddings
        embedding = self.embeddings[item.col_type._type].exec([item.val], {})
        assert isinstance(embedding, np.ndarray)

        # In arithmetic operations between floats and ints (or between vector and int), CockroachDB requires an explicit
        # cast. Otherwise the query fails.
        cast_ints = Env.get().is_using_cockroachdb
        one = cast(1, sql.types.Float) if cast_ints else 1
        neg_one = cast(-1, sql.types.Float) if cast_ints else -1
        if self.metric == self.Metric.COSINE:
            return val_column.sa_col.cosine_distance(embedding) * neg_one + one
        elif self.metric == self.Metric.IP:
            return val_column.sa_col.max_inner_product(embedding) * neg_one
        else:
            assert self.metric == self.Metric.L2
            return val_column.sa_col.l2_distance(embedding)

    def order_by_clause(self, val_column: catalog.Column, item: exprs.Literal, is_asc: bool) -> sql.ColumnElement:
        """Create a ColumnElement that is used in an ORDER BY clause"""
        assert item.col_type._type in self.embeddings
        embedding = self.embeddings[item.col_type._type].exec([item.val], {})
        assert isinstance(embedding, np.ndarray)

        if self.metric == self.Metric.COSINE:
            result = val_column.sa_col.cosine_distance(embedding)
            result = result.desc() if is_asc else result
        elif self.metric == self.Metric.IP:
            result = val_column.sa_col.max_inner_product(embedding)
            result = result.desc() if is_asc else result
        else:
            assert self.metric == self.Metric.L2
            result = val_column.sa_col.l2_distance(embedding)
        return result

    @classmethod
    def display_name(cls) -> str:
        return 'embedding'

    @classmethod
    def _resolve_embedding_fn(cls, embed_fn: func.Function, expected_type: ts.ColumnType.Type) -> func.Function | None:
        """Find an overload resolution for `embed_fn` that matches the given type."""
        assert isinstance(embed_fn, func.Function)
        for resolved_fn in embed_fn._resolved_fns:
            # The embedding function must be a 1-ary function of the correct type. But it's ok if the function signature
            # has more than one parameter, as long as it has at most one *required* parameter.
            sig = resolved_fn.signature
            if (
                len(sig.parameters) >= 1
                and len(sig.required_parameters) <= 1
                and sig.parameters_by_pos[0].col_type.type_enum == expected_type
            ):
                # We found a valid signature. Now, if it has more than one parameter, we need to transform it into a
                # 1-ary function by fixing all the other parameters to their defaults. This is to ensure that
                # conditional_return_type resolves correctly.
                if len(sig.parameters) == 1:
                    unary_fn = resolved_fn
                else:
                    assert all(sig.parameters_by_pos[i].has_default for i in range(1, len(sig.parameters)))
                    defaults = {param.name: param.default for param in sig.parameters_by_pos[1:]}
                    unary_fn = resolved_fn.using(**defaults)
                assert not unary_fn.is_polymorphic
                assert len(unary_fn.signature.parameters) == 1
                return unary_fn
        return None

    @classmethod
    def _validate_embedding_fn(cls, embed_fn: func.Function) -> None:
        """Validate the given embedding function."""
        assert not embed_fn.is_polymorphic

        return_type = embed_fn.signature.return_type

        if not isinstance(return_type, ts.ArrayType):
            raise excs.Error(
                f'The function `{embed_fn.name}` is not a valid embedding: '
                f'it must return an array, but returns {return_type}'
            )

        shape = return_type.shape
        if len(shape) != 1 or shape[0] is None:
            raise excs.Error(
                f'The function `{embed_fn.name}` is not a valid embedding: '
                f'it must return a 1-dimensional array of a specific length, but returns {return_type}'
            )
        if shape[0] <= 0:
            raise excs.Error(
                f'The function `{embed_fn.name}` is not a valid embedding: '
                f'it returns an array of invalid length {shape[0]}'
            )

    def as_dict(self) -> dict:
        d: dict[str, Any] = {'metric': self.metric.name.lower(), 'precision': self.precision.value}
        for embed_type, embed_fn in self.embeddings.items():
            key = f'{embed_type.name.lower()}_embed'
            d[key] = embed_fn.as_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EmbeddingIndex:
        string_embed = func.Function.from_dict(d['string_embed']) if d.get('string_embed') is not None else None
        image_embed = func.Function.from_dict(d['image_embed']) if d.get('image_embed') is not None else None
        audio_embed = func.Function.from_dict(d['audio_embed']) if d.get('audio_embed') is not None else None
        video_embed = func.Function.from_dict(d['video_embed']) if d.get('video_embed') is not None else None
        return cls(
            metric=d['metric'],
            string_embed=string_embed,
            image_embed=image_embed,
            audio_embed=audio_embed,
            video_embed=video_embed,
            precision=d['precision'],
        )
