from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterator, Sequence, cast
from uuid import UUID

import numpy as np
import PIL.Image
import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import func
from pixeltable.catalog.table_version import TableVersionKey
from pixeltable.env import Env
from pixeltable.runtime import get_runtime

from ..utils.description_helper import DescriptionHelper
from ..utils.filecache import FileCache
from ..utils.http import fetch_url
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

if TYPE_CHECKING:
    from pixeltable._query import Query, ResultSet


class ColumnRef(Expr):
    """
    A Pixeltable expression that references a column of a table. A `ColumnRef` is created by column access
    on a [`Table`][pixeltable.Table], such as `t.col`.

    Not thread-safe.
    """

    # When this reference is created in the context of a view, it can also refer to a column of the view base.
    # For that reason, a ColumnRef needs to be serialized with the qualifying table id (column ids are only
    # unique in the context of a particular table).

    # Media validation:
    # - media validation is potentially cpu-intensive, and it's desirable to schedule and parallelize it during
    #   general expr evaluation
    # - media validation on read is done in ColumnRef.eval()
    # - a validating ColumnRef cannot be translated to SQL (because the validation is done in Python)
    # - in that case, the ColumnRef also instantiates a second non-validating ColumnRef as a component (= dependency)
    # - the non-validating ColumnRef is used for SQL translation

    # TODO:
    # separate Exprs (like validating ColumnRefs) from the logical expression tree and instead have RowBuilder
    # insert them into the EvalCtxs as needed

    col_md: catalog.ColumnVersionMd
    _col: catalog.ColumnHandle

    needs_iterator_evaluation: bool
    perform_validation: bool  # if True, performs media validation
    iter_arg_ctx: RowBuilder.EvalCtx | None
    iter_outputs: list[ColumnRef] | None
    base_rowid_len: int  # number of rowid columns in the base table

    # execution state
    base_rowid: Sequence[Any | None]
    iterator: Iterator
    pos_idx: int

    def __init__(self, col_md: catalog.ColumnVersionMd, perform_validation: bool = False):
        super().__init__(col_md.col_type)
        self.col_md = col_md
        key = TableVersionKey(col_md.qcolid.tbl_id, col_md.col_effective_version)
        self._col = catalog.ColumnHandle(catalog.TableVersionHandle(key), col_md.qcolid.col_id)

        # pos (id=0) is an unstored iterator column, but its value comes from the PK, not the iterator output dict
        self.needs_iterator_evaluation = col_md.is_iterator_col and not col_md.is_stored and col_md.id != 0
        self.iter_arg_ctx = None
        self.iter_outputs = None
        self.base_rowid_len = 0
        self.base_rowid = []
        self.iterator = None
        self.pos_idx = 0

        self.perform_validation = perform_validation
        if self.perform_validation:
            self.components = [ColumnRef(col_md, perform_validation=False)]
        self.id = self._create_id()

    @property
    def column_md(self) -> catalog.ColumnVersionMd:
        return self.col_md

    @property
    def col(self) -> catalog.Column:
        return self._col.get()

    @property
    def tbl_version(self) -> catalog.TableVersionHandle:
        # the path-context table (e.g. the view a base column is accessed through), where column-level metadata
        # such as indexes lives - as opposed to _col, which is the column's physical owner
        key = TableVersionKey(self.col_md.tbl_id, self.col_md.effective_version)
        return catalog.TableVersionHandle(key)

    def set_iter_arg_ctx(self, iter_arg_ctx: RowBuilder.EvalCtx, iter_outputs: list[ColumnRef]) -> None:
        self.iter_arg_ctx = iter_arg_ctx
        self.iter_outputs = iter_outputs
        # If this is an unstored iterator column, then the iterator outputs may be needed in order to properly set the
        # iterator position. Therefore, we need to add them as components in order to ensure they're marked as
        # eval dependencies.
        self.components.extend(iter_outputs)
        assert len(self.iter_arg_ctx.target_slot_idxs) == 1  # a single inline dict

    def _id_attrs(self) -> list[tuple[str, Any]]:
        # Identity is the physical column (owner), not the path-context: the same physical column reached through
        # different views/snapshots is the same expr. Must mirror the fields in _equals().
        return [
            *super()._id_attrs(),
            ('col_tbl_id', self.col_md.qcolid.tbl_id),
            ('col_id', self.col_md.qcolid.col_id),
            ('col_effective_version', self.col_md.col_effective_version),
            ('perform_validation', self.perform_validation),
        ]

    def _equals(self, other: ColumnRef) -> bool:
        # identity is the physical column (owner) + its version, independent of path-context; see _id_attrs()
        return (
            self.col_md.qcolid == other.col_md.qcolid
            # the same physical column of a snapshot and of its live table share a qcolid but differ here
            and self.col_md.col_effective_version == other.col_md.col_effective_version
            and self.perform_validation == other.perform_validation
        )

    # override
    def _retarget(self, tbl_versions: dict[UUID, catalog.TableVersion]) -> ColumnRef:
        # retarget only the column's physical owner (qcolid.tbl_id) to the given TableVersion while preserving the
        # context path
        qcolid = self.col_md.qcolid
        target = tbl_versions[qcolid.tbl_id]
        assert qcolid.col_id in target.cols_by_id, f'{target}: {qcolid.col_id} not in {list(target.cols_by_id.keys())}'
        new_col_md = self.col_md.retarget(target.effective_version)
        return ColumnRef(new_col_md, self.perform_validation)

    def __getattr__(self, name: str) -> Expr:
        from .column_property_ref import ColumnPropertyRef

        # resolve column properties
        if name == ColumnPropertyRef.Property.CELLMD.name.lower():
            # This is not user accessible, but used internally to store cell metadata
            return super().__getattr__(name)

        col_md = self.column_md
        if (
            name == ColumnPropertyRef.Property.ERRORTYPE.name.lower()
            or name == ColumnPropertyRef.Property.ERRORMSG.name.lower()
        ):
            is_valid = (col_md.is_computed or col_md.col_type.is_media_type()) and col_md.is_stored
            if not is_valid:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{name} only valid for a stored computed or media column: {self}',
                )
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])
        if (
            name == ColumnPropertyRef.Property.FILEURL.name.lower()
            or name == ColumnPropertyRef.Property.LOCALPATH.name.lower()
        ):
            if not col_md.col_type.is_media_type():
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'{name} only valid for image/video/audio/document columns: {self}',
                )
            if col_md.is_computed and not col_md.is_stored:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'{name} not valid for computed unstored columns: {self}'
                )
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])

        if self.col_type.is_json_type():
            from .json_path import JsonPath

            return JsonPath(self, [name])

        return super().__getattr__(name)

    def recompute(self, *, cascade: bool = True, errors_only: bool = False) -> catalog.UpdateStatus:
        cat = get_runtime().catalog
        with cat.begin_xact(for_write=False):
            tbl = cat.get_table_by_id(self.col_md.tbl_id)
            tvp = tbl._tbl_version_path

        # lock_mutable_tree=True: we need to be able to see whether any transitive view has column dependents
        with cat.begin_xact(for_write=True, write_tvps=[tvp], lock_mutable_tree=True):
            col = self.col
            tbl_version = col.tbl_handle.get()
            if tbl_version.id != self.col_md.tbl_id:
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Cannot recompute column of a base.')
            if tbl_version.is_snapshot:
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Cannot recompute column of a snapshot.')
            status = tbl_version.recompute_columns([col.name], errors_only=errors_only, cascade=cascade)
            FileCache.get().emit_eviction_warnings()
            return status

    def similarity(
        self,
        item: Any = None,
        *,
        string: str | None = None,
        image: str | PIL.Image.Image | None = None,
        audio: str | None = None,
        video: str | None = None,
        document: str | None = None,
        vector: np.ndarray | None = None,
        idx: str | None = None,
    ) -> Expr:
        """
        Return a new expression representing the similarity score between the values of this column and the given
        (constant) item. In order for this to work, there must be an embedding index defined on this column that
        supports the modality of the given item (string, image, audio, video, document). Similarity will be scored
        according to the metric defined by the embedding index.

        Exactly one of `string`, `image`, `audio`, `video`, `document`, or `vector` must be provided. The `item`
        parameter is deprecated and exists for backward compatibility only.

        If `string`, `image`, `audio`, `video`, or `document` is provided, then an embedding vector will be computed
        for the given input as defined by the embedding index and used to determine similarity. If `vector` is
        provided, then it must be a 1-dimensional array of the same dimensionality as the index, and similarity will
        be determined directly against the vector.

        The optional `idx` parameter specifies the name of the embedding index to use. If there is more than one
        embedding index defined on this column, then `idx` _must_ be provided.

        Args:
            string: A string to compare against the values of this column.
            image: An image to compare against the values of this column (either a local file path, a URL, or an
                in-memory `PIL.Image.Image`).
            audio: An audio file to compare against the values of this column (a local file path or a URL).
            video: A video file to compare against the values of this column (a local file path or a URL).
            document: A document file to compare against the values of this column (a local file path or a URL).
            vector: A 1-dimensional NumPy array to compare against the values of this column.
            idx: An optional embedding index name. _Required_ if there is more than one embedding index defined on
                this column.
            item: **Deprecated** as of version 0.5.7.

        Returns:
            A new expression representing the similarity score between the values of this column and the given item.

        Examples:
            All of these examples assume that `t` is a table with an image column `t.image`.

            Add an embedding index to `t.image` using the `clip()`
            embedding (this only needs to be done once):

            >>> from pixeltable.functions.huggingface import clip
            ...
            ... t.add_embedding_index(
            ...     t.image, clip.using(model_id='openai/clip-vit-base-patch32')
            ... )

            Do a nearest neighbor search against a string (with `k=5`):

            >>> sim = t.image.similarity(string='a photograph of a cat')
            ... t.select(t.image, sim).order_by(sim, asc=False).head(5)

            Do a nearest neighbor search against an image:

            >>> sim = t.image.similarity(image='https://example.com/reference-cat.jpg')
            ... t.select(t.image, sim).order_by(sim, asc=False).head(5)
        """

        from .similarity_expr import SimilarityExpr

        if item is not None:
            warnings.warn(
                'Use of similarity() without specifying an explicit modality is deprecated -- '
                'since version 0.5.7. Please use one of the following instead:\n'
                '  .similarity(string=...)\n'
                '  .similarity(image=...)\n'
                '  .similarity(audio=...)\n'
                '  .similarity(video=...)\n'
                '  .similarity(document=...)\n'
                '  .similarity(vector=...)',
                DeprecationWarning,
                stacklevel=2,
            )

        arg_count = (
            (string is not None)
            + (image is not None)
            + (audio is not None)
            + (video is not None)
            + (document is not None)
            + (vector is not None)
        )

        if item is not None and arg_count != 0:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'similarity(): `item` is deprecated and cannot be used together with modality arguments',
            )

        if arg_count > 1:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'similarity(): expected exactly one of string=..., image=..., audio=..., video=..., document=...,'
                ' vector=...',
            )

        expr: Expr

        # TODO: For audio/video/document, we're storing the local file path in the Literal for the similarity
        #     expression. This is problematic in scenarios where the similarity expression is serialized.
        if item is not None:
            if isinstance(item, Expr):  # This can happen when using similarity() with @query
                if not (item.col_type.is_string_type() or item.col_type.is_image_type()):
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'similarity(): expected `String` or `Image`; got `{item.col_type}`',
                    )
                expr = item
            else:
                if not isinstance(item, (str, PIL.Image.Image)):
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'similarity(): expected `str` or `PIL.Image.Image`; got `{type(item).__name__}`',
                    )
                expr = Expr.from_object(item)
                assert expr.col_type.is_string_type() or expr.col_type.is_image_type()

        if string is not None:
            if isinstance(string, Expr):
                if not string.col_type.is_string_type():
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'similarity(string=...): expected `String`; got `{string.col_type}`',
                    )
                expr = string
            else:
                if not isinstance(string, str):
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'similarity(string=...): expected `str`; got `{type(string).__name__}`',
                    )
                expr = Expr.from_object(string)
                assert expr.col_type.is_string_type()

        if image is not None:
            if isinstance(image, Expr):
                if not image.col_type.is_image_type():
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH, f'similarity(image=...): expected `Image`; got `{image.col_type}`'
                    )
                expr = image
            else:
                if not isinstance(image, (str, PIL.Image.Image)):
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'similarity(image=...): expected `str` or `PIL.Image.Image`; got `{type(image).__name__}`',
                    )
                if isinstance(image, str):
                    image_path = fetch_url(image, allow_local_file=True)
                    image = PIL.Image.open(image_path)
                    image.load()
                expr = Expr.from_object(image)
                assert expr.col_type.is_image_type()

        if audio is not None:
            if isinstance(audio, Expr):
                if not audio.col_type.is_audio_type():
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH, f'similarity(audio=...): expected `Audio`; got `{audio.col_type}`'
                    )
                expr = audio
            else:
                if not isinstance(audio, str):
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'similarity(audio=...): expected `str` (path to audio file); got `{type(audio).__name__}`',
                    )
                audio_path = fetch_url(audio, allow_local_file=True)
                expr = Literal(str(audio_path), ts.AudioType())

        if video is not None:
            if isinstance(video, Expr):
                if not video.col_type.is_video_type():
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH, f'similarity(video=...): expected `Video`; got `{video.col_type}`'
                    )
                expr = video
            else:
                if not isinstance(video, str):
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'similarity(video=...): expected `str` (path to video file); got `{type(video).__name__}`',
                    )
                video_path = fetch_url(video, allow_local_file=True)
                expr = Literal(str(video_path), ts.VideoType())

        if document is not None:
            if isinstance(document, Expr):
                if not document.col_type.is_document_type():
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'similarity(document=...): expected `Document`; got `{document.col_type}`',
                    )
                expr = document
            else:
                if not isinstance(document, str):
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        'similarity(document=...): expected `str` (path to document file); '
                        f'got `{type(document).__name__}`',
                    )
                document_path = fetch_url(document, allow_local_file=True)
                expr = Literal(str(document_path), ts.DocumentType())

        if vector is not None:
            if isinstance(vector, Expr):
                if not vector.col_type.is_array_type():
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'similarity(vector=...): expected `Array`; got `{vector.col_type}`',
                    )
                expr = vector
            else:
                if not isinstance(vector, np.ndarray):
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'similarity(vector=...): expected `numpy.ndarray`, or array `Expr`; '
                        f'got `{type(vector).__name__}`',
                    )
                if vector.ndim != 1:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'similarity(vector=...): expected 1-dimensional array; got shape {vector.shape}',
                    )

                if not np.issubdtype(vector.dtype, np.floating):
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'similarity(vector=...): expected float array; got dtype {vector.dtype}',
                    )

                col_type = ts.ColumnType.infer_literal_type(vector)
                expr = Literal(vector, col_type=col_type)

        from pixeltable.index import EmbeddingIndex

        # Resolve the table through its owning catalog (which may be a proxy) so the index lookup works
        # uniformly for local and hosted tables. get_table_by_id() manages its own transaction.
        tbl = get_runtime().get_table_by_id(self.col_md.tbl_id, version=self.col_md.effective_version)
        assert tbl is not None
        # get_idx_md() resolves the concrete index, raising if idx is ambiguous or doesn't exist.
        idx_md = tbl._tbl_path.get_idx_md(self.col_md.qcolid, idx, EmbeddingIndex)

        # init_args carries one '<modality>_embed' entry per supported modality (see EmbeddingIndex.as_dict()).
        # Array columns are exempt: similarity search uses the raw vector directly.
        if not expr.col_type.is_array_type():
            type_str = expr.col_type._type.name.lower()
            if f'{type_str}_embed' not in idx_md.init_args:
                article = 'an' if type_str[0] in 'aeiou' else 'a'
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Embedding index {idx_md.name!r} on column {self.col_md.name!r} does not have {article} '
                    f'{type_str} embedding and does not support {type_str} queries',
                )

        table_version_key = TableVersionKey(self.col_md.tbl_id, self.col_md.effective_version)
        return SimilarityExpr(
            expr, idx_name=idx_md.name, qcol_id=self.col_md.qcolid, table_version_key=table_version_key
        )

    def embedding(self, *, idx: str | None = None) -> ColumnRef:
        """
        Return a reference to the values of an embedding index on this column.

        If an embedding index is defined on a column, the usual way to use that index is via a
        [`similarity()`][pixeltable.exprs.ColumnRef.similarity] lookup. Sometimes it is also useful to directly access
        the index values (i.e., the embedding vectors themselves). Calling `embedding()` returns a new `ColumnRef`
        expression of type `pxt.Array[(dim,), prec]`, where `dim` and `prec` are the dimensionality and precision
        of the column's embedding index.

        If there is more than one embedding index defined on this column, then the `idx` parameter must be provided to
        specify which index to reference. If there is only one index, then `idx` is optional.

        Args:
            idx: An optional embedding index name. _Required_ if there is more than one embedding index defined on
                this column.

        Returns:
            A new `ColumnRef` referencing the values of the specified embedding index on this column.

        Raises:
            `pxt.Error` if there is no embedding index defined on this column, if `idx` is not provided when there are
            multiple embedding indices, or if `idx` does not match any embedding index defined on this column.

        Examples:
            All of these examples assume that `t` is a table with an image column `t.image`.

            Add an embedding index to `t.image` using the `clip()`
            embedding (this only needs to be done once):

            >>> from pixeltable.functions.huggingface import clip
            ...
            ... t.add_embedding_index(
            ...     t.image, clip.using(model_id='openai/clip-vit-base-patch32')
            ... )

            Reference the embedding index values directly:

            >>> t.select(t.image, t.image.embedding())
        """

        from pixeltable.index import EmbeddingIndex

        idx_info = self.tbl_version.get().get_idx(self.col, idx, EmbeddingIndex)
        val_col = idx_info.val_col
        return ColumnRef(val_col.column_version_md())

    def default_column_name(self) -> str | None:
        return self.column_md.name

    def select(self) -> 'Query':
        from pixeltable._query import Query
        from pixeltable.query_clauses import FromClause

        # Resolve the column's table against the catalog it belongs to (which may be a hosted/proxy catalog),
        # at its effective_version so a column accessed via a snapshot/view resolves against the pinned version
        # rather than the live table. get_table_by_id() manages its own transaction, so no begin_xact is needed
        # here (and a proxy catalog has none).
        cat = get_runtime().get_catalog(Env.get().tbl_catalog_uri(self.col_md.tbl_id))
        tbl = cat.get_table_by_id(self.col_md.tbl_id, version=self.col_md.effective_version)
        return Query(FromClause([tbl._tbl_path])).select(self)

    def show(self, *args: Any, **kwargs: Any) -> 'ResultSet':
        return self.select().show(*args, **kwargs)

    def head(self, *args: Any, **kwargs: Any) -> 'ResultSet':
        return self.select().head(*args, **kwargs)

    def tail(self, *args: Any, **kwargs: Any) -> 'ResultSet':
        return self.select().tail(*args, **kwargs)

    def count(self) -> int:
        return self.select().count()

    def distinct(self) -> 'Query':
        """Return distinct values in this column."""
        return self.select().distinct()

    def __str__(self) -> str:
        col_md = self.column_md
        return col_md.name if col_md.name is not None else f'<unnamed column {col_md.id}>'

    def __repr__(self) -> str:
        return self._descriptors().to_string()

    def _repr_html_(self) -> str:
        return self._descriptors().to_html()

    def _descriptors(self) -> DescriptionHelper:
        col_name = self.col_md.name
        with get_runtime().catalog.begin_xact():
            tbl = get_runtime().catalog.get_table_by_id(self.col_md.qcolid.tbl_id)
        # TODO: is this what we want? it's printing the path of the containing table, not of the context table
        helper = DescriptionHelper()
        helper.append(f'Column\n{col_name!r}\n(of table {tbl._path()!r})')
        col_df, _ = tbl._col_descriptor([col_name])
        helper.append(col_df)
        idxs = tbl._index_descriptor([col_name])
        if len(idxs) > 0:
            helper.append(idxs)
        return helper

    def prepare(self, args: dict[str, Any], bound_args: dict[str, Any]) -> None:
        from pixeltable import store

        if not self.needs_iterator_evaluation:
            return
        col = self.col
        self.base_rowid_len = col.get_tbl().base.get().num_rowid_columns()
        self.base_rowid = [None] * self.base_rowid_len
        assert isinstance(col.get_tbl().store_tbl, store.StoreComponentView)
        self.pos_idx = cast(store.StoreComponentView, col.get_tbl().store_tbl).pos_col_idx

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        if self.perform_validation:
            return None
        return self.col.sa_col

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        col = self.col
        if self.perform_validation:
            # validate media file of our input ColumnRef and if successful, replicate the state of that slot
            # to our slot
            unvalidated_slot_idx = self.components[0].slot_idx
            if data_row.file_paths[unvalidated_slot_idx] is None:
                # no media file to validate, we still need to replicate the value
                assert data_row.file_urls[unvalidated_slot_idx] is None
                val = data_row.vals[unvalidated_slot_idx]
                data_row.vals[self.slot_idx] = val
                data_row.has_val[self.slot_idx] = True
                return

            try:
                col.col_type.validate_media(data_row.file_paths[unvalidated_slot_idx])
                # access the value only after successful validation
                val = data_row[unvalidated_slot_idx]
                data_row.vals[self.slot_idx] = val
                data_row.has_val[self.slot_idx] = True
                # make sure that the validated slot points to the same file as the unvalidated slot
                data_row.file_paths[self.slot_idx] = data_row.file_paths[unvalidated_slot_idx]
                data_row.file_urls[self.slot_idx] = data_row.file_urls[unvalidated_slot_idx]
                return
            except excs.Error as exc:
                # propagate the exception, but ignore it otherwise;
                # media validation errors don't cause exceptions during query execution
                # TODO: allow for different error-handling behavior
                row_builder.set_exc(data_row, self.slot_idx, exc)
                return

        if not self.needs_iterator_evaluation:
            # supply default
            data_row[self.slot_idx] = None
            return

        # if this is a new base row, we need to instantiate a new iterator
        if self.base_rowid != data_row.pk[: self.base_rowid_len]:
            assert self.iter_arg_ctx is not None
            row_builder.eval(data_row, self.iter_arg_ctx)
            iterator_args = data_row[self.iter_arg_ctx.target_slot_idxs[0]]
            self.iterator = col.get_tbl().iterator_call.eval(iterator_args)
            self.base_rowid = data_row.pk[: self.base_rowid_len]
        stored_outputs = {col_ref.col.name: data_row[col_ref.slot_idx] for col_ref in self.iter_outputs}
        assert all(name is not None for name in stored_outputs)
        assert isinstance(self.iterator, func.PxtIterator)  # Otherwise we could not have an unstored column
        self.iterator.seek(data_row.pk[self.pos_idx], **stored_outputs)
        res = next(self.iterator)
        data_row[self.slot_idx] = res[col.name]

    def _as_dict(self) -> dict:
        # we omit self.components, even if this is a validating ColumnRef, because init() will recreate it
        return {
            'tbl_id': str(self.col_md.tbl_id),
            'effective_version': self.col_md.effective_version,
            'col_tbl_id': str(self.col_md.qcolid.tbl_id),
            'col_tbl_effective_version': self.col_md.col_effective_version,
            'col_id': self.col_md.qcolid.col_id,
            'perform_validation': self.perform_validation,
        }

    @classmethod
    def get_column_id(cls, d: dict) -> catalog.QColumnId:
        return catalog.QColumnId(UUID(d['col_tbl_id']), d['col_id'])

    @classmethod
    def _from_dict(
        cls, d: dict, _: list[Expr], tbl_versions: dict[UUID, catalog.TableVersion] | None = None
    ) -> ColumnRef:
        col_tbl_id = UUID(d['col_tbl_id'])
        col_id = d['col_id']
        col_tbl_effective_version: int | None = d['col_tbl_effective_version']
        qcolid = catalog.QColumnId(col_tbl_id, col_id)

        tbl_id = UUID(d['tbl_id'])
        effective_version: int | None = d['effective_version']

        if tbl_versions is not None:
            target = tbl_versions[col_tbl_id]
            if col_id not in target.cols_by_id:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND,
                    f'Column was dropped (no record for column ID {col_id} in table {target.versioned_name!r})',
                )
            tvp = catalog.TableVersionPath(target.handle)
            col_md = tvp.get_column_md(qcolid)
        else:
            # validate_initialized=False: this can be called during TableVersion.__init__() while the TV is
            # still being loaded (e.g. deserializing iterator args), so we must not trigger a re-entrant load.
            key = TableVersionKey(col_tbl_id, col_tbl_effective_version)
            tv = get_runtime().catalog.get_tbl_version(key, validate_initialized=False)
            col = tv.cols_by_id.get(col_id)
            if col is None:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND,
                    f'Column was dropped (no record for column ID {col_id} in table {tv.versioned_name!r})',
                )
            col_md = col.column_version_md()

        if tbl_id != col_tbl_id or effective_version != col_tbl_effective_version:
            col_md = col_md.with_context(tbl_id, effective_version)
        return cls(col_md, d['perform_validation'])
