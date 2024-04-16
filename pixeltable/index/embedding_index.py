from __future__ import annotations

from typing import Optional

import pgvector.sqlalchemy
import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.func as func
import pixeltable.type_system as ts
from .base import IndexBase


class EmbeddingIndex(IndexBase):
    """
    Internal interface used by the catalog and runtime system to interact with (embedding) indices:
    - types and expressions needed to create and populate the index value column
    - creating/dropping the index
    - translating 'matches' queries into sqlalchemy predicates
    """

    def __init__(
            self, c: catalog.Column, text_embed: Optional[func.Function] = None,
            img_embed: Optional[func.Function] = None):
        if not c.col_type.is_string_type() and not c.col_type.is_image_type():
            raise excs.Error(f'Embedding index requires string or image column')
        if c.col_type.is_string_type() and text_embed is None:
                raise excs.Error(f'Text embedding function is required for column {c.name} (parameter `txt_embed`)')
        if c.col_type.is_image_type() and img_embed is None:
            raise excs.Error(f'Image embedding function is required for column {c.name} (parameter `img_embed`)')
        if text_embed is not None:
            # verify signature
            self._validate_embedding_fn(text_embed, 'txt_embed', ts.ColumnType.Type.STRING)
        if img_embed is not None:
            # verify signature
            self._validate_embedding_fn(img_embed, 'img_embed', ts.ColumnType.Type.IMAGE)

        from pixeltable.exprs import ColumnRef
        self.value_expr = text_embed(ColumnRef(c)) if c.col_type.is_string_type() else img_embed(ColumnRef(c))
        assert self.value_expr.col_type.is_array_type()
        self.txt_embed = text_embed
        self.img_embed = img_embed
        vector_size = self.value_expr.col_type.shape[0]
        assert vector_size is not None
        self.index_col_type = pgvector.sqlalchemy.Vector(vector_size)

    def index_value_expr(self) -> 'pixeltable.exprs.Expr':
        """Return expression that computes the value that goes into the index"""
        return self.value_expr

    def index_sa_type(self) -> sql.sqltypes.TypeEngine:
        """Return the sqlalchemy type of the index value column"""
        return self.index_col_type

    def create_index(self, index_name: str, index_value_col: catalog.Column, conn: sql.engine.Connection) -> None:
        """Create the index on the index value column"""
        idx = sql.Index(
            index_name, index_value_col.sa_col,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={index_value_col.sa_col.name: 'vector_cosine_ops'}
        )
        idx.create(bind=conn)

    @classmethod
    def display_name(cls) -> str:
        return 'embedding'

    @classmethod
    def _validate_embedding_fn(cls, embed_fn: func.Function, name: str, expected_type: ts.ColumnType.Type) -> None:
        """Validate the signature"""
        assert isinstance(embed_fn, func.Function)
        sig = embed_fn.signature
        if not sig.return_type.is_array_type():
            raise excs.Error(f'{name} must return an array, but returns {sig.return_type}')
        else:
            shape = sig.return_type.shape
            if len(shape) != 1 or shape[0] == None:
                raise excs.Error(f'{name} must return a 1D array of a specific length, but returns {sig.return_type}')
        if len(sig.parameters) != 1 or sig.parameters_by_pos[0].col_type.type_enum != expected_type:
            raise excs.Error(
                f'{name} must take a single {expected_type.name.lower()} parameter, but has signature {sig}')

    def as_dict(self) -> dict:
        return {
            'txt_embed': None if self.txt_embed is None else self.txt_embed.as_dict(),
            'img_embed': None if self.img_embed is None else self.img_embed.as_dict()
        }

    @classmethod
    def from_dict(cls, c: catalog.Column, d: dict) -> EmbeddingIndex:
        txt_embed = func.Function.from_dict(d['txt_embed']) if d['txt_embed'] is not None else None
        img_embed = func.Function.from_dict(d['img_embed']) if d['img_embed'] is not None else None
        return cls(c, text_embed=txt_embed, img_embed=img_embed)
