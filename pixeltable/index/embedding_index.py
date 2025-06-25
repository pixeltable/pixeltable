from __future__ import annotations

import enum
from typing import Any, ClassVar, Optional

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import PIL.Image
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import catalog, exprs, func
from pixeltable.env import Env

from .base import IndexBase


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

    PGVECTOR_OPS: ClassVar[dict[Metric, str]] = {
        Metric.COSINE: 'vector_cosine_ops',
        Metric.IP: 'vector_ip_ops',
        Metric.L2: 'vector_l2_ops',
    }

    metric: Metric
    value_expr: exprs.FunctionCall
    string_embed: Optional[func.Function]
    image_embed: Optional[func.Function]
    string_embed_signature_idx: int
    image_embed_signature_idx: int
    index_col_type: pgvector.sqlalchemy.Vector

    def __init__(
        self,
        c: catalog.Column,
        metric: str,
        embed: Optional[func.Function] = None,
        string_embed: Optional[func.Function] = None,
        image_embed: Optional[func.Function] = None,
    ):
        if embed is None and string_embed is None and image_embed is None:
            raise excs.Error('At least one of `embed`, `string_embed`, or `image_embed` must be specified')
        metric_names = [m.name.lower() for m in self.Metric]
        if metric.lower() not in metric_names:
            raise excs.Error(f'Invalid metric {metric}, must be one of {metric_names}')
        if not c.col_type.is_string_type() and not c.col_type.is_image_type():
            raise excs.Error('Embedding index requires string or image column')

        self.string_embed = None
        self.image_embed = None

        # Resolve the specific embedding functions corresponding to the user-provided `string_embed`, `image_embed`,
        # and/or `embed`. For string embeddings, `string_embed` will be used if specified; otherwise, `embed` will
        # be used as a fallback, if it has a matching signature. Likewise for image embeddings.

        if string_embed is not None:
            # `string_embed` is specified; it MUST be valid.
            self.string_embed = self._resolve_embedding_fn(string_embed, ts.ColumnType.Type.STRING)
            if self.string_embed is None:
                raise excs.Error(
                    f'The function `{string_embed.name}` is not a valid string embedding: '
                    'it must take a single string parameter'
                )
        elif embed is not None:
            # `embed` is specified; see if it has a string signature.
            self.string_embed = self._resolve_embedding_fn(embed, ts.ColumnType.Type.STRING)

        if image_embed is not None:
            # `image_embed` is specified; it MUST be valid.
            self.image_embed = self._resolve_embedding_fn(image_embed, ts.ColumnType.Type.IMAGE)
            if self.image_embed is None:
                raise excs.Error(
                    f'The function `{image_embed.name}` is not a valid image embedding: '
                    'it must take a single image parameter'
                )
        elif embed is not None:
            # `embed` is specified; see if it has an image signature.
            self.image_embed = self._resolve_embedding_fn(embed, ts.ColumnType.Type.IMAGE)

        if self.string_embed is None and self.image_embed is None:
            # No string OR image signature was found. This can only happen if `embed` was specified and
            # contains no matching signatures.
            assert embed is not None
            raise excs.Error(
                f'The function `{embed.name}` is not a valid embedding: it must take a single string or image parameter'
            )

        # Now validate the return types of the embedding functions.

        if self.string_embed is not None:
            self._validate_embedding_fn(self.string_embed)

        if self.image_embed is not None:
            self._validate_embedding_fn(self.image_embed)

        if c.col_type.is_string_type() and self.string_embed is None:
            raise excs.Error(f"Text embedding function is required for column {c.name} (parameter 'string_embed')")
        if c.col_type.is_image_type() and self.image_embed is None:
            raise excs.Error(f"Image embedding function is required for column {c.name} (parameter 'image_embed')")

        self.metric = self.Metric[metric.upper()]
        self.value_expr = (
            self.string_embed(exprs.ColumnRef(c))
            if c.col_type.is_string_type()
            else self.image_embed(exprs.ColumnRef(c))
        )
        assert isinstance(self.value_expr.col_type, ts.ArrayType)
        vector_size = self.value_expr.col_type.shape[0]
        assert vector_size is not None
        self.index_col_type = pgvector.sqlalchemy.Vector(vector_size)

    def index_value_expr(self) -> exprs.Expr:
        """Return expression that computes the value that goes into the index"""
        return self.value_expr

    def records_value_errors(self) -> bool:
        return True

    def index_sa_type(self) -> sql.types.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        return self.index_col_type

    def create_index(self, index_name: str, index_value_col: catalog.Column) -> None:
        """Create the index on the index value column"""
        idx = sql.Index(
            index_name,
            index_value_col.sa_col,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={index_value_col.sa_col.name: self.PGVECTOR_OPS[self.metric]},
        )
        conn = Env.get().conn
        idx.create(bind=conn)

    def drop_index(self, index_name: str, index_value_col: catalog.Column) -> None:
        """Drop the index on the index value column"""
        # TODO: implement
        raise NotImplementedError()

    def similarity_clause(self, val_column: catalog.Column, item: Any) -> sql.ColumnElement:
        """Create a ColumnElement that represents '<val_column> <op> <item>'"""
        assert isinstance(item, (str, PIL.Image.Image))
        if isinstance(item, str):
            assert self.string_embed is not None
            embedding = self.string_embed.exec([item], {})
        if isinstance(item, PIL.Image.Image):
            assert self.image_embed is not None
            embedding = self.image_embed.exec([item], {})

        if self.metric == self.Metric.COSINE:
            return val_column.sa_col.cosine_distance(embedding) * -1 + 1
        elif self.metric == self.Metric.IP:
            return val_column.sa_col.max_inner_product(embedding) * -1
        else:
            assert self.metric == self.Metric.L2
            return val_column.sa_col.l2_distance(embedding)

    def order_by_clause(self, val_column: catalog.Column, item: Any, is_asc: bool) -> sql.ColumnElement:
        """Create a ColumnElement that is used in an ORDER BY clause"""
        assert isinstance(item, (str, PIL.Image.Image))
        embedding: Optional[np.ndarray] = None
        if isinstance(item, str):
            assert self.string_embed is not None
            embedding = self.string_embed.exec([item], {})
        if isinstance(item, PIL.Image.Image):
            assert self.image_embed is not None
            embedding = self.image_embed.exec([item], {})
        assert embedding is not None

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
    def _resolve_embedding_fn(
        cls, embed_fn: func.Function, expected_type: ts.ColumnType.Type
    ) -> Optional[func.Function]:
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

    def as_dict(self) -> dict:
        return {
            'metric': self.metric.name.lower(),
            'string_embed': None if self.string_embed is None else self.string_embed.as_dict(),
            'image_embed': None if self.image_embed is None else self.image_embed.as_dict(),
        }

    @classmethod
    def from_dict(cls, c: catalog.Column, d: dict) -> EmbeddingIndex:
        string_embed = func.Function.from_dict(d['string_embed']) if d['string_embed'] is not None else None
        image_embed = func.Function.from_dict(d['image_embed']) if d['image_embed'] is not None else None
        return cls(c, metric=d['metric'], string_embed=string_embed, image_embed=image_embed)
