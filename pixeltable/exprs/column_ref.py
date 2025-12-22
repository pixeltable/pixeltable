from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence, cast
from uuid import UUID

import PIL.Image
import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.iterators as iters
import pixeltable.type_system as ts
from pixeltable.catalog.table_version import TableVersionKey

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
    """A reference to a table column

    When this reference is created in the context of a view, it can also refer to a column of the view base.
    For that reason, a ColumnRef needs to be serialized with the qualifying table id (column ids are only
    unique in the context of a particular table).

    Media validation:
    - media validation is potentially cpu-intensive, and it's desirable to schedule and parallelize it during
      general expr evaluation
    - media validation on read is done in ColumnRef.eval()
    - a validating ColumnRef cannot be translated to SQL (because the validation is done in Python)
    - in that case, the ColumnRef also instantiates a second non-validating ColumnRef as a component (= dependency)
    - the non-validating ColumnRef is used for SQL translation

    A ColumnRef may have an optional reference table, which carries the context of the ColumnRef resolution. Thus
    if `v` is a view of `t` (for example), then `v.my_col` and `t.my_col` refer to the same underlying column, but
    their reference tables will be `v` and `t`, respectively. This is to ensure correct behavior of expressions such
    as `v.my_col.head()`.

    TODO:
    separate Exprs (like validating ColumnRefs) from the logical expression tree and instead have RowBuilder
    insert them into the EvalCtxs as needed
    """

    col: catalog.Column  # TODO: merge with col_handle
    col_handle: catalog.ColumnHandle
    reference_tbl: catalog.TableVersionPath | None
    is_unstored_iter_col: bool
    perform_validation: bool  # if True, performs media validation
    iter_arg_ctx: RowBuilder.EvalCtx | None
    iter_outputs: list[ColumnRef] | None
    base_rowid_len: int  # number of rowid columns in the base table

    # execution state
    base_rowid: Sequence[Any | None]
    iterator: iters.ComponentIterator | None
    pos_idx: int

    def __init__(
        self,
        col: catalog.Column,
        reference_tbl: catalog.TableVersionPath | None = None,
        perform_validation: bool | None = None,
    ):
        super().__init__(col.col_type)
        self.col = col
        self.reference_tbl = reference_tbl
        self.col_handle = col.handle

        self.is_unstored_iter_col = col.is_iterator_col and not col.is_stored
        self.iter_arg_ctx = None
        self.iter_outputs = None
        self.base_rowid_len = 0
        self.base_rowid = []
        self.iterator = None
        self.pos_idx = 0

        self.perform_validation = False
        if col.col_type.is_media_type():
            # we perform media validation if the column is a media type and the validation is set to ON_READ,
            # unless we're told not to
            if perform_validation is not None:
                self.perform_validation = perform_validation
            else:
                self.perform_validation = (
                    col.col_type.is_media_type() and col.media_validation == catalog.MediaValidation.ON_READ
                )
        else:
            assert perform_validation is None or not perform_validation
        if self.perform_validation:
            non_validating_col_ref = ColumnRef(col, perform_validation=False)
            self.components = [non_validating_col_ref]
        self.id = self._create_id()

    def set_iter_arg_ctx(self, iter_arg_ctx: RowBuilder.EvalCtx, iter_outputs: list[ColumnRef]) -> None:
        self.iter_arg_ctx = iter_arg_ctx
        self.iter_outputs = iter_outputs
        # If this is an unstored iterator column, then the iterator outputs may be needed in order to properly set the
        # iterator position. Therefore, we need to add them as components in order to ensure they're marked as
        # eval dependencies.
        self.components.extend(iter_outputs)
        assert len(self.iter_arg_ctx.target_slot_idxs) == 1  # a single inline dict

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [
            *super()._id_attrs(),
            ('tbl_id', self.col.tbl_handle.id),
            ('col_id', self.col.id),
            ('perform_validation', self.perform_validation),
        ]

    # override
    def _retarget(self, tbl_versions: dict[UUID, catalog.TableVersion]) -> ColumnRef:
        target = tbl_versions[self.col.tbl_handle.id]
        assert self.col.id in target.cols_by_id, f'{target}: {self.col.id} not in {list(target.cols_by_id.keys())}'
        col = target.cols_by_id[self.col.id]
        return ColumnRef(col, self.reference_tbl)

    def __getattr__(self, name: str) -> Expr:
        from .column_property_ref import ColumnPropertyRef

        # resolve column properties
        if name == ColumnPropertyRef.Property.CELLMD.name.lower():
            # This is not user accessible, but used internally to store cell metadata
            return super().__getattr__(name)

        if (
            name == ColumnPropertyRef.Property.ERRORTYPE.name.lower()
            or name == ColumnPropertyRef.Property.ERRORMSG.name.lower()
        ):
            is_valid = (self.col.is_computed or self.col.col_type.is_media_type()) and self.col.is_stored
            if not is_valid:
                raise excs.Error(f'{name} only valid for a stored computed or media column: {self}')
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])
        if (
            name == ColumnPropertyRef.Property.FILEURL.name.lower()
            or name == ColumnPropertyRef.Property.LOCALPATH.name.lower()
        ):
            if not self.col.col_type.is_media_type():
                raise excs.Error(f'{name} only valid for image/video/audio/document columns: {self}')
            if self.col.is_computed and not self.col.is_stored:
                raise excs.Error(f'{name} not valid for computed unstored columns: {self}')
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])

        if self.col_type.is_json_type():
            from .json_path import JsonPath

            return JsonPath(self, [name])

        return super().__getattr__(name)

    def recompute(self, *, cascade: bool = True, errors_only: bool = False) -> catalog.UpdateStatus:
        cat = catalog.Catalog.get()
        # lock_mutable_tree=True: we need to be able to see whether any transitive view has column dependents
        with cat.begin_xact(tbl=self.reference_tbl, for_write=True, lock_mutable_tree=True):
            tbl_version = self.col_handle.tbl_version.get()
            if tbl_version.id != self.reference_tbl.tbl_id:
                raise excs.Error('Cannot recompute column of a base.')
            if tbl_version.is_snapshot:
                raise excs.Error('Cannot recompute column of a snapshot.')
            col_name = self.col_handle.get().name
            status = tbl_version.recompute_columns([col_name], errors_only=errors_only, cascade=cascade)
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
        idx: str | None = None,
    ) -> Expr:
        from .similarity_expr import SimilarityExpr

        if item is not None:
            warnings.warn(
                'Use of similarity() without specifying an explicit modality is deprecated -- '
                'since version 0.5.7. Please use one of the following instead:\n'
                '  .similarity(string=...)\n'
                '  .similarity(image=...)\n'
                '  .similarity(audio=...)\n'
                '  .similarity(video=...)',
                DeprecationWarning,
                stacklevel=2,
            )

        arg_count = (string is not None) + (image is not None) + (audio is not None) + (video is not None)

        if item is not None and arg_count != 0:
            raise excs.Error('similarity(): `item` is deprecated and cannot be used together with modality arguments')

        if arg_count > 1:
            raise excs.Error('similarity(): expected exactly one of string=..., image=..., audio=..., video=...')

        expr: Expr

        if item is not None:
            if isinstance(item, Expr):  # This can happen when using similarity() with @query
                if not (item.col_type.is_string_type() or item.col_type.is_image_type()):
                    raise excs.Error(f'similarity(): expected `String` or `Image`; got `{item.col_type}`')
                expr = item
            else:
                if not isinstance(item, (str, PIL.Image.Image)):
                    raise excs.Error(f'similarity(): expected `str` or `PIL.Image.Image`; got `{type(item).__name__}`')
                expr = Expr.from_object(item)
                assert expr.col_type.is_string_type() or expr.col_type.is_image_type()

        if string is not None:
            if isinstance(string, Expr):
                if not string.col_type.is_string_type():
                    raise excs.Error(f'similarity(string=...): expected `String`; got `{expr.col_type}`')
                expr = string
            else:
                if not isinstance(string, str):
                    raise excs.Error(f'similarity(string=...): expected `str`; got `{type(string).__name__}`')
                expr = Expr.from_object(string)
                assert expr.col_type.is_string_type()

        if image is not None:
            if isinstance(image, Expr):
                if not image.col_type.is_image_type():
                    raise excs.Error(f'similarity(image=...): expected `Image`; got `{image.col_type}`')
                expr = image
            else:
                if not isinstance(image, (str, PIL.Image.Image)):
                    raise excs.Error(
                        f'similarity(image=...): expected `str` or `PIL.Image.Image`; got `{type(image).__name__}`'
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
                    raise excs.Error(f'similarity(audio=...): expected `Audio`; got `{audio.col_type}`')
                expr = audio
            else:
                if not isinstance(audio, str):
                    raise excs.Error(
                        f'similarity(audio=...): expected `str` (path to audio file); got `{type(audio).__name__}`'
                    )
                expr = Literal(audio, ts.AudioType())

        if video is not None:
            if isinstance(video, Expr):
                if not video.col_type.is_video_type():
                    raise excs.Error(f'similarity(video=...): expected `Video`; got `{video.col_type}`')
                expr = video
            else:
                if not isinstance(video, str):
                    raise excs.Error(
                        f'similarity(video=...): expected `str` (path to video file); got `{type(video).__name__}`'
                    )
                expr = Literal(video, ts.VideoType())

        return SimilarityExpr(self, expr, idx_name=idx)

    def embedding(self, *, idx: str | None = None) -> ColumnRef:
        from pixeltable.index import EmbeddingIndex

        idx_info = self.tbl.get().get_idx(self.col, idx, EmbeddingIndex)
        return ColumnRef(idx_info.val_col)

    @property
    def tbl(self) -> catalog.TableVersionHandle:
        return self.reference_tbl.tbl_version if self.reference_tbl is not None else self.col.tbl_handle

    def default_column_name(self) -> str | None:
        return self.col.name if self.col is not None else None

    def _equals(self, other: ColumnRef) -> bool:
        return self.col == other.col and self.perform_validation == other.perform_validation

    def select(self) -> 'Query':
        import pixeltable.plan as plan
        from pixeltable._query import Query

        if self.reference_tbl is None:
            # No reference table; use the current version of the table to which the column belongs
            tbl = catalog.Catalog.get().get_table_by_id(self.col.tbl_handle.id)
            return tbl.select(self)
        else:
            # Explicit reference table; construct a Query directly from it
            return Query(plan.FromClause([self.reference_tbl])).select(self)

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
        if self.col.name is None:
            return f'<unnamed column {self.col.id}>'
        else:
            return self.col.name

    def __repr__(self) -> str:
        return self._descriptors().to_string()

    def _repr_html_(self) -> str:
        return self._descriptors().to_html()

    def _descriptors(self) -> DescriptionHelper:
        with catalog.Catalog.get().begin_xact():
            tbl = catalog.Catalog.get().get_table_by_id(self.col.tbl_handle.id)
        helper = DescriptionHelper()
        helper.append(f'Column\n{self.col.name!r}\n(of table {tbl._path()!r})')
        helper.append(tbl._col_descriptor([self.col.name]))
        idxs = tbl._index_descriptor([self.col.name])
        if len(idxs) > 0:
            helper.append(idxs)
        return helper

    def prepare(self) -> None:
        from pixeltable import store

        if not self.is_unstored_iter_col:
            return
        col = self.col_handle.get()
        self.base_rowid_len = col.get_tbl().base.get().num_rowid_columns()
        self.base_rowid = [None] * self.base_rowid_len
        assert isinstance(col.get_tbl().store_tbl, store.StoreComponentView)
        self.pos_idx = cast(store.StoreComponentView, col.get_tbl().store_tbl).pos_col_idx

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        if self.perform_validation:
            return None
        self.col = self.col_handle.get()
        return self.col.sa_col

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
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
                self.col.col_type.validate_media(data_row.file_paths[unvalidated_slot_idx])
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

        if not self.is_unstored_iter_col:
            # supply default
            data_row[self.slot_idx] = None
            return

        # if this is a new base row, we need to instantiate a new iterator
        if self.base_rowid != data_row.pk[: self.base_rowid_len]:
            assert self.iter_arg_ctx is not None
            row_builder.eval(data_row, self.iter_arg_ctx)
            iterator_args = data_row[self.iter_arg_ctx.target_slot_idxs[0]]
            self.iterator = self.col.get_tbl().iterator_cls(**iterator_args)
            self.base_rowid = data_row.pk[: self.base_rowid_len]
        stored_outputs = {col_ref.col.name: data_row[col_ref.slot_idx] for col_ref in self.iter_outputs}
        assert all(name is not None for name in stored_outputs)
        self.iterator.set_pos(data_row.pk[self.pos_idx], **stored_outputs)
        res = next(self.iterator)
        data_row[self.slot_idx] = res[self.col.name]

    def _as_dict(self) -> dict:
        tbl_handle = self.col.tbl_handle
        # we omit self.components, even if this is a validating ColumnRef, because init() will recreate the
        # non-validating component ColumnRef
        assert tbl_handle.anchor_tbl_id is None  # TODO: support anchor_tbl_id for view-over-replica
        return {
            'tbl_id': str(tbl_handle.id),
            'tbl_version': tbl_handle.effective_version,
            'col_id': self.col.id,
            'reference_tbl': self.reference_tbl.as_dict() if self.reference_tbl is not None else None,
            'perform_validation': self.perform_validation,
        }

    @classmethod
    def get_column_id(cls, d: dict) -> catalog.QColumnId:
        tbl_id, col_id = UUID(d['tbl_id']), d['col_id']
        return catalog.QColumnId(tbl_id, col_id)

    @classmethod
    def get_column(cls, d: dict) -> catalog.Column:
        tbl_id, version, col_id = UUID(d['tbl_id']), d['tbl_version'], d['col_id']
        # validate_initialized=False: this gets called as part of TableVersion.init()
        # TODO: When we have views on replicas, we will need to store anchor_tbl_id in metadata as well.
        tbl_version = catalog.Catalog.get().get_tbl_version(
            TableVersionKey(tbl_id, version, None), validate_initialized=False
        )
        # don't use tbl_version.cols_by_id here, this might be a snapshot reference to a column that was then dropped
        col = next(col for col in tbl_version.cols if col.id == col_id)
        return col

    @classmethod
    def _from_dict(cls, d: dict, _: list[Expr]) -> ColumnRef:
        col = cls.get_column(d)
        reference_tbl = None if d['reference_tbl'] is None else catalog.TableVersionPath.from_dict(d['reference_tbl'])
        perform_validation = d['perform_validation']
        return cls(col, reference_tbl, perform_validation=perform_validation)
