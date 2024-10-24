from __future__ import annotations

import enum
from typing import Any, Optional

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import PIL.Image
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import catalog, exprs, func

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

    PGVECTOR_OPS = {
        Metric.COSINE: 'vector_cosine_ops',
        Metric.IP: 'vector_ip_ops',
        Metric.L2: 'vector_l2_ops'
    }

    def __init__(
            self, c: catalog.Column, metric: str, string_embed: Optional[func.Function] = None,
            image_embed: Optional[func.Function] = None):
        metric_names = [m.name.lower() for m in self.Metric]
        if metric.lower() not in metric_names:
            raise excs.Error(f'Invalid metric {metric}, must be one of {metric_names}')
        if not c.col_type.is_string_type() and not c.col_type.is_image_type():
            raise excs.Error(f'Embedding index requires string or image column')
        if c.col_type.is_string_type() and string_embed is None:
                raise excs.Error(f"Text embedding function is required for column {c.name} (parameter 'string_embed')")
        if c.col_type.is_image_type() and image_embed is None:
            raise excs.Error(f"Image embedding function is required for column {c.name} (parameter 'image_embed')")
        if string_embed is not None:
            # verify signature
            self._validate_embedding_fn(string_embed, 'string_embed', ts.ColumnType.Type.STRING)
        if image_embed is not None:
            # verify signature
            self._validate_embedding_fn(image_embed, 'image_embed', ts.ColumnType.Type.IMAGE)

        self.metric = self.Metric[metric.upper()]
        self.value_expr = string_embed(exprs.ColumnRef(c)) if c.col_type.is_string_type() else image_embed(exprs.ColumnRef(c))
        assert isinstance(self.value_expr.col_type, ts.ArrayType)
        self.string_embed = string_embed
        self.image_embed = image_embed
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

    def create_index(self, index_name: str, index_value_col: catalog.Column, conn: sql.engine.Connection) -> None:
        """Create the index on the index value column"""
        idx = sql.Index(
            index_name, index_value_col.sa_col,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={index_value_col.sa_col.name: self.PGVECTOR_OPS[self.metric]}
        )
        idx.create(bind=conn)

    def similarity_clause(self, val_column: catalog.Column, item: Any) -> sql.ColumnElement:
        """Create a ColumnElement that represents '<val_column> <op> <item>'"""
        assert isinstance(item, (str, PIL.Image.Image))
        if isinstance(item, str):
            assert self.string_embed is not None
            embedding = self.string_embed.exec(item)
        if isinstance(item, PIL.Image.Image):
            assert self.image_embed is not None
            embedding = self.image_embed.exec(item)

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
            embedding = self.string_embed.exec(item)
        if isinstance(item, PIL.Image.Image):
            assert self.image_embed is not None
            embedding = self.image_embed.exec(item)
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
    def _validate_embedding_fn(cls, embed_fn: func.Function, name: str, expected_type: ts.ColumnType.Type) -> None:
        """Validate the signature"""
        assert isinstance(embed_fn, func.Function)
        sig = embed_fn.signature
        if len(sig.parameters) != 1 or sig.parameters_by_pos[0].col_type.type_enum != expected_type:
            raise excs.Error(
                f'{name} must take a single {expected_type.name.lower()} parameter, but has signature {sig}')

        # validate return type
        param_name = sig.parameters_by_pos[0].name
        if expected_type == ts.ColumnType.Type.STRING:
            return_type = embed_fn.call_return_type({param_name: 'dummy'})
        else:
            assert expected_type == ts.ColumnType.Type.IMAGE
            img = PIL.Image.new('RGB', (512, 512))
            return_type = embed_fn.call_return_type({param_name: img})
        assert return_type is not None
        if not isinstance(return_type, ts.ArrayType):
            raise excs.Error(f'{name} must return an array, but returns {return_type}')
        else:
            shape = return_type.shape
            if len(shape) != 1 or shape[0] == None:
                raise excs.Error(f'{name} must return a 1D array of a specific length, but returns {return_type}')

    def as_dict(self) -> dict:
        return {
            'metric': self.metric.name.lower(),
            'string_embed': None if self.string_embed is None else self.string_embed.as_dict(),
            'image_embed': None if self.image_embed is None else self.image_embed.as_dict()
        }

    @classmethod
    def from_dict(cls, c: catalog.Column, d: dict) -> EmbeddingIndex:
        string_embed = func.Function.from_dict(d['string_embed']) if d['string_embed'] is not None else None
        image_embed = func.Function.from_dict(d['image_embed']) if d['image_embed'] is not None else None
        return cls(c, metric=d['metric'], string_embed=string_embed, image_embed=image_embed)
