from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import pandas as pd
from typing_extensions import overload

from pixeltable import exceptions as excs

from .globals import is_valid_identifier
from .schema_object import SchemaObject

if TYPE_CHECKING:
    from pathlib import Path

    import pydantic
    import torch.utils.data

    from pixeltable import exprs, type_system as ts
    from pixeltable._query import Query, ResultCursor, ResultSet
    from pixeltable.func.function import Function
    from pixeltable.query_clauses import JoinType
    from pixeltable.row import RowBatch
    from pixeltable.types import ColumnSpec

    from ..exprs import ColumnRef
    from ..globals import TableDataSource
    from .table_metadata import TableMetadata, VersionMetadata
    from .table_path import TablePath
    from .update_status import UpdateStatus


class Table(SchemaObject):
    """
    A handle to a table, view, or snapshot. This class is the primary interface through which table operations
    (queries, insertions, updates, etc.) are performed in Pixeltable.

    Thread-safe.
    """

    @property
    @abc.abstractmethod
    def _tbl_path(self) -> 'TablePath':
        """The metadata path backing this handle."""

    def _get_schema(self) -> dict[str, 'ts.ColumnType']:
        """Return the schema (column names and types) of this table, including columns inherited from bases."""
        return {md.name: md.col_type for md in self._tbl_path.column_md() if md.name is not None}

    @abc.abstractmethod
    def get_metadata(self) -> 'TableMetadata':
        """
        Retrieves metadata associated with this table.

        Returns:
            A [TableMetadata][pixeltable.TableMetadata] instance containing this table's metadata.
        """

    @abc.abstractmethod
    def __getattr__(self, name: str) -> 'exprs.ColumnRef':
        """Return a ColumnRef for the given name."""

    @abc.abstractmethod
    def __getitem__(self, name: str) -> 'exprs.ColumnRef':
        """Return a ColumnRef for the given name."""

    @abc.abstractmethod
    def list_views(self, *, recursive: bool = True) -> list[str]:
        """
        Returns a list of all views and snapshots of this `Table`.

        Args:
            recursive: If `False`, returns only the immediate successor views of this `Table`. If `True`, returns
                all sub-views (including views of views, etc.)

        Returns:
            A list of view paths.
        """

    def select(self, *items: Any, **named_items: Any) -> 'Query':
        """Select columns or expressions from this table.

        See [`Query.select`][pixeltable.Query.select] for more details.
        """
        from pixeltable._query import Query
        from pixeltable.query_clauses import FromClause

        query = Query(FromClause(tbls=[self._tbl_path]))
        if len(items) == 0 and len(named_items) == 0:
            return query
        return query.select(*items, **named_items)

    def where(self, pred: 'exprs.Expr') -> 'Query':
        """Filter rows from this table based on the expression.

        See [`Query.where`][pixeltable.Query.where] for more details.
        """
        return self.select().where(pred)

    def join(self, other: 'Table', *, on: 'exprs.Expr' | None = None, how: 'JoinType.LiteralType' = 'inner') -> 'Query':
        """Join this table with another table."""
        return self.select().join(other, on=on, how=how)

    def order_by(self, *items: 'exprs.Expr', asc: bool = True) -> 'Query':
        """Order the rows of this table based on the expression.

        See [`Query.order_by`][pixeltable.Query.order_by] for more details.
        """
        return self.select().order_by(*items, asc=asc)

    @overload
    def group_by(self, grouping_tbl: 'Table', /) -> 'Query': ...
    @overload
    def group_by(self, *grouping_items: 'exprs.Expr') -> 'Query': ...
    def group_by(self, *items: 'exprs.Expr | Table') -> 'Query':
        """Group the rows of this table based on the expression.

        See [`Query.group_by`][pixeltable.Query.group_by] for more details.
        """
        return self.select().group_by(*items)  # type: ignore[arg-type]

    def distinct(self) -> 'Query':
        """Remove duplicate rows from table."""
        return self.select().distinct()

    def limit(self, n: int, offset: int | None = None) -> 'Query':
        """Select a limited number of rows from the Table, optionally skipping rows for pagination.

        Args:
            n: Number of rows to select.
            offset: Number of rows to skip before returning results. Default is None (no offset).

        Returns:
            A Query with the specified limited rows.

        Examples:
            Get the first 10 rows:

            >>> t.limit(10).collect()

            Get rows 21-30 (skip first 20, return next 10):

            >>> t.limit(10, offset=20).collect()
        """
        return self.select().limit(n, offset=offset)

    def sample(
        self,
        n: int | None = None,
        n_per_stratum: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
        stratify_by: Any = None,
    ) -> Query:
        """Choose a shuffled sample of rows

        See [`Query.sample`][pixeltable.Query.sample] for more details.
        """
        return self.select().sample(
            n=n, n_per_stratum=n_per_stratum, fraction=fraction, seed=seed, stratify_by=stratify_by
        )

    def collect(self) -> 'ResultSet':
        """Return rows from this table."""
        return self.select().collect()

    def cursor(self) -> 'ResultCursor':
        """Return a [`ResultCursor`][pixeltable.ResultCursor] that iterates over this table's rows.

        See [`ResultCursor`][pixeltable.ResultCursor] for usage examples and lifecycle details.
        """
        return self.select().cursor()

    def show(self, n: int = 20) -> 'ResultSet':
        """Return the first n rows from this table."""
        return self.select().show(n)

    def head(self, n: int = 10) -> 'ResultSet':
        """Return the first n rows inserted into this table."""
        return self.select().head(n)

    def tail(self, n: int = 10) -> 'ResultSet':
        """Return the last n rows inserted into this table."""
        return self.select().tail(n)

    def count(self) -> int:
        """Return the number of rows in this table."""
        return self.select().count()

    @abc.abstractmethod
    def columns(self) -> list[str]:
        """Return the names of the columns in this table."""

    def get_base_table(self) -> 'Table' | None:
        return self._get_base_table()

    @abc.abstractmethod
    def _get_base_table(self) -> 'Table' | None:
        """The base's Table instance. Requires a transaction context"""

    @abc.abstractmethod
    def describe(self) -> None:
        """
        Print the table schema.
        """

    @abc.abstractmethod
    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset':
        """Return a PyTorch Dataset for this table.
        See Query.to_pytorch_dataset()
        """

    @abc.abstractmethod
    def to_coco_dataset(self) -> Path:
        """Return the path to a COCO json file for this table.
        See Query.to_coco_dataset()
        """

    @abc.abstractmethod
    def add_columns(
        self,
        schema: Mapping[str, type | ColumnSpec],
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> UpdateStatus:
        """
        Adds multiple columns to the table. The columns must be concrete (non-computed) columns; to add computed
        columns, use [`add_computed_column()`][pixeltable.catalog.Table.add_computed_column] instead.

        The format of the `schema` argument is a dict mapping column names to their types.

        Args:
            schema: A dictionary mapping column names to a `type` or a [`ColumnSpec`][pixeltable.ColumnSpec] dict.
            if_exists: Determines the behavior if a column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace' or 'replace_force'`: drop the existing column and add the new column, if it has no
                    dependents.

                Note that the `if_exists` parameter is applied to all columns in the schema.
                To apply different behaviors to different columns, please use
                [`add_column()`][pixeltable.Table.add_column] for each column.

        Returns:
            Information about the execution status of the operation.

        Raises:
            Error: If any column name is invalid, or already exists and `if_exists='error'`,
                or `if_exists='replace*'` but the column has dependents or is a basetable column.

        Examples:
            Add multiple columns to the table `my_table`:

            >>> tbl = pxt.get_table('my_table')
            ... schema = {'new_col_1': pxt.Int, 'new_col_2': pxt.String}
            ... tbl.add_columns(schema)

            It is also possible to specify column metadata using a dict:

            >>> tbl = pxt.get_table('my_table')
            ... schema = {
            ...     'new_col_1': {
            ...         'type': pxt.Image,
            ...         'stored': True,
            ...         'media_validation': 'on_write',
            ...     },
            ...     'new_col_2': pxt.String,
            ... }
            ... tbl.add_columns(schema)
        """

    @abc.abstractmethod
    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        **kwargs: type | ColumnSpec,
    ) -> UpdateStatus:
        """
        Adds an ordinary (non-computed) column to the table.

        Args:
            kwargs: Exactly one keyword argument of the form `col_name=type` or `col_name=col_spec_dict`,
                where `col_spec_dict` is a [`ColumnSpec`][pixeltable.ColumnSpec] dict.
            if_exists: Determines the behavior if the column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace'` or `'replace_force'`: drop the existing column and add the new column, if it has
                    no dependents.

        Returns:
            Information about the execution status of the operation.

        Raises:
            Error: If the column name is invalid, or already exists and `if_exists='error'`,
                or `if_exists='replace*'` but the column has dependents or is a basetable column.

        Examples:
            Add an int column:

            >>> tbl.add_column(new_col=pxt.Int)

            Add a column with column metadata using a dict:

            >>> tbl.add_column(
            ...     img_col={
            ...         'type': pxt.Image,
            ...         'stored': True,
            ...         'media_validation': 'on_write',
            ...     }
            ... )

            Alternatively, adding a column can also be expressed using `add_columns`:

            >>> tbl.add_columns({'new_col': pxt.Int})

            As well as with column metadata:

            >>> tbl.add_columns(
            ...     {
            ...         'img_col': {
            ...             'type': pxt.Image,
            ...             'stored': True,
            ...             'media_validation': 'on_write',
            ...         }
            ...     }
            ... )
        """

    @abc.abstractmethod
    def add_computed_column(
        self,
        *,
        stored: bool | None = None,
        destination: str | Path | None = None,
        custom_metadata: Any = None,
        comment: str = '',
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
        if_exists: Literal['error', 'ignore', 'replace'] = 'error',
        **kwargs: exprs.Expr,
    ) -> UpdateStatus:
        """
        Adds a computed column to the table.

        Args:
            kwargs: Exactly one keyword argument of the form `col_name=expression`.
            stored: Whether the column is materialized and stored or computed on demand.
            destination: An object store reference for persisting computed files.
            custom_metadata: Optional user-defined metadata to associate with the column. Must be a valid
                JSON-serializable object.
            comment: An optional comment; its meaning is user-defined.
            print_stats: If `True`, print execution metrics during evaluation.
            on_error: Determines the behavior if an error occurs while evaluating the column expression for at least one
                row.

                - `'abort'`: an exception will be raised and the column will not be added.
                - `'ignore'`: execution will continue and the column will be added. Any rows
                    with errors will have a `None` value for the column, with information about the error stored in the
                    corresponding `tbl.col_name.errormsg` and `tbl.col_name.errortype` fields.
            if_exists: Determines the behavior if the column already exists. Must be one of the following:

                - `'error'`: an exception will be raised.
                - `'ignore'`: do nothing and return.
                - `'replace' or 'replace_force'`: drop the existing column and add the new column, iff it has
                    no dependents.

        Returns:
            Information about the execution status of the operation.

        Raises:
            Error: If the column name is invalid or already exists and `if_exists='error'`,
                or `if_exists='replace*'` but the column has dependents or is a basetable column.

        Examples:
            For a table with an image column `frame`, add an image column `rotated` that rotates the image by
            90 degrees:

            >>> tbl.add_computed_column(rotated=tbl.frame.rotate(90))

            Do the same, but now the column is unstored:

            >>> tbl.add_computed_column(rotated=tbl.frame.rotate(90), stored=False)
        """

    @abc.abstractmethod
    def drop_column(self, column: str | ColumnRef, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        """Drop a column from the table.

        Args:
            column: The name or reference of the column to drop.
            if_not_exists: Directive for handling a non-existent column. Must be one of the following:

                - `'error'`: raise an error if the column does not exist.
                - `'ignore'`: do nothing if the column does not exist.

        Raises:
            Error: If the column does not exist and `if_exists='error'`,
                or if it is referenced by a dependent computed column.

        Examples:
            Drop the column `col` from the table `my_table` by column name:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_column('col')

            Drop the column `col` from the table `my_table` by column reference:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_column(tbl.col)

            Drop the column `col` from the table `my_table` if it exists, otherwise do nothing:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_col(tbl.col, if_not_exists='ignore')
        """

    @abc.abstractmethod
    def rename_column(self, old_name: str, new_name: str) -> None:
        """Rename a column.

        Args:
            old_name: The current name of the column.
            new_name: The new name of the column.

        Raises:
            Error: If the column does not exist, or if the new name is invalid or already exists.

        Examples:
            Rename the column `col1` to `col2` of the table `my_table`:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.rename_column('col1', 'col2')
        """

    @abc.abstractmethod
    def alter_column(self, column: str | ColumnRef, *, type_: type) -> None:
        """Alter the type of a column.

        Currently, the only supported change is widening a non-computed column from non-nullable to
        nullable.

        Args:
            column: The name or reference of the column to alter.
            type_: The new type for the column.

        Raises:
            Error: If the column does not exist, is a computed column, if `type_` is not a supported widening of the
            current type, or if the change cannot be performed for any other reason.

        Examples:
            Make a previously required column nullable:

            >>> tbl = pxt.create_table('my_table', {'col': pxt.Required[pxt.String]})
            ... tbl.alter_column('col', type_=pxt.String)
        """

    @abc.abstractmethod
    def add_embedding_index(
        self,
        column: str | ColumnRef,
        *,
        idx_name: str | None = None,
        embedding: Function | None = None,
        string_embed: Function | None = None,
        image_embed: Function | None = None,
        audio_embed: Function | None = None,
        video_embed: Function | None = None,
        document_embed: Function | None = None,
        metric: Literal['cosine', 'ip', 'l2'] = 'cosine',
        precision: Literal['fp16', 'fp32'] = 'fp16',
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> None:
        """
        Add an embedding index to the table. Once the index is created, it will be automatically kept up-to-date as new
        rows are inserted into the table.

        To add an embedding index, specify the column to be indexed and, if the column is not an `Array` column, an
        embedding UDF. `String`, `Image`, `Audio`, `Video`, `Document`, and `Array` columns are currently supported.

        Multimodal embeddings can be specified in one of two ways: via a single `embedding` argument with a
        multi-signature UDF (one signature per modality), or via separate modality-specific arguments (`string_embed`,
        `image_embed`, etc.). If both are provided, the modality-specific arguments will supersede the corresponding
        signatures of the `embedding` UDF.

        For `Array` columns, which are assumed to contain precomputed embeddings, an embedding function is optional;
        if provided, it will be used to convert query values into embeddings for similarity search.

        Args:
            column: The name of, or reference to, the column to be indexed; must be a `String`, `Image`, `Audio`,
                `Video`, `Document`, or `Array` column.
            idx_name: An optional name for the index. If not specified, a name such as `'idx0'` will be generated
                automatically. If specified, the name must be unique for this table and a valid pixeltable column name.
                When `idx_name` is omitted, duplicates are detected by the index definition (the embedding
                function(s), `metric`, and `precision`) on the column: re-adding an index with an identical
                definition is governed by `if_exists`.
            embedding: The UDF to use for the embedding. Must be a UDF that accepts a single argument of type `String`
                or `Image` (as appropriate for the column being indexed) and returns a fixed-size 1-dimensional
                array of floats. If omitted, then at least one of the modality-specific `*_embed` arguments must be
                supplied.
            string_embed: An optional UDF to use for the string embedding component of this index.
            image_embed: An optional UDF to use for the image embedding component of this index.
            audio_embed: An optional UDF to use for the audio embedding component of this index.
            video_embed: An optional UDF to use for the video embedding component of this index.
            document_embed: An optional UDF to use for the document embedding component of this index.
            metric: Distance metric to use for the index; one of `'cosine'`, `'ip'`, or `'l2'`.
                The default is `'cosine'`.
            precision: level of precision for the embeddings; one of `'fp16'` or `'fp32'`.
            if_exists: Directive for handling an existing index. The existing index is the one with the same name
                (if `idx_name` is given), or one with an identical definition on the same column (if `idx_name` is
                omitted). Must be one of the following:

                - `'error'`: raise an error if such an index already exists.
                - `'ignore'`: do nothing if such an index already exists.
                - `'replace'` or `'replace_force'`: replace the existing index with the new one.

        Raises:
            Error: If a matching index already exists for the table and `if_exists='error'`, or if
                the specified column does not exist.

        Examples:
            Add an index to the `img` column of the table `my_table`:

            >>> from pixeltable.functions.huggingface import clip
            >>> tbl = pxt.get_table('my_table')
            >>> embedding_fn = clip.using(model_id='openai/clip-vit-base-patch32')
            >>> tbl.add_embedding_index(tbl.img, embedding=embedding_fn)

            Alternatively, the `img` column may be specified by name:

            >>> tbl.add_embedding_index('img', embedding=embedding_fn)

            Once the index is created, similarity lookups can be performed using the `similarity` pseudo-function:

            >>> sim = tbl.img.similarity(
            ...     image='/path/to/my-image.jpg'  # can also be a URL or a PIL image
            ... )
            >>> tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)

            If the embedding UDF is a multimodal embedding (supporting more than one data type), then lookups may be
            performed using any of its supported modalities. In our example, CLIP supports both text and images, so we
            can also search for images using a text description:

            >>> sim = tbl.img.similarity(string='a picture of a train')
            >>> tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)

            Audio and video lookups would look like this:

            >>> sim = tbl.img.similarity(audio='/path/to/audio.flac')
            >>> sim = tbl.img.similarity(video='/path/to/video.mp4')

            Multiple indexes can be defined on each column. Add a second index to the `img` column, using the inner
            product as the distance metric, and with a specific name:

            >>> tbl.add_embedding_index(
            ...     tbl.img, idx_name='ip_idx', embedding=embedding_fn, metric='ip'
            ... )

            Add an index using separately specified string and image embeddings:

            >>> tbl.add_embedding_index(
            ...     tbl.img,
            ...     string_embed=string_embedding_fn,
            ...     image_embed=image_embedding_fn,
            ... )
        """

    @abc.abstractmethod
    def drop_embedding_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        """
        Drop an embedding index from the table. Either a column name or an index name (but not both) must be
        specified. If a column name or reference is specified, it must be a column containing exactly one
        embedding index; otherwise the specific index name must be provided instead.

        Args:
            column: The name of, or reference to, the column from which to drop the index.
                    The column must have only one embedding index.
            idx_name: The name of the index to drop.
            if_not_exists: Directive for handling a non-existent index. Must be one of the following:

                - `'error'`: raise an error if the index does not exist.
                - `'ignore'`: do nothing if the index does not exist.

                Note that `if_not_exists` parameter is only applicable when an `idx_name` is specified
                and it does not exist, or when `column` is specified and it has no index.
                `if_not_exists` does not apply to non-exisitng column.

        Raises:
            Error: If `column` is specified, but the column does not exist, or it contains no embedding
                indices and `if_not_exists='error'`, or the column has multiple embedding indices.
            Error: If `idx_name` is specified, but the index is not an embedding index, or
                the index does not exist and `if_not_exists='error'`.

        Examples:
            Drop the embedding index on the `img` column of the table `my_table` by column name:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(column='img')

            Drop the embedding index on the `img` column of the table `my_table` by column reference:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(column=tbl.img)

            Drop the embedding index `idx1` of the table `my_table` by index name:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(idx_name='idx1')

            Drop the embedding index `idx1` of the table `my_table` by index name, if it exists, otherwise do nothing:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_embedding_index(idx_name='idx1', if_not_exists='ignore')
        """

    @abc.abstractmethod
    def drop_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        """
        Drop an index from the table. Either a column name or an index name (but not both) must be
        specified. If a column name or reference is specified, it must be a column containing exactly one index;
        otherwise the specific index name must be provided instead.

        Args:
            column: The name of, or reference to, the column from which to drop the index.
                    The column must have only one embedding index.
            idx_name: The name of the index to drop.
            if_not_exists: Directive for handling a non-existent index. Must be one of the following:

                - `'error'`: raise an error if the index does not exist.
                - `'ignore'`: do nothing if the index does not exist.

                Note that `if_not_exists` parameter is only applicable when an `idx_name` is specified
                and it does not exist, or when `column` is specified and it has no index.
                `if_not_exists` does not apply to non-exisitng column.

        Raises:
            Error: If `column` is specified, but the column does not exist, or it contains no
                indices or multiple indices.
            Error: If `idx_name` is specified, but the index does not exist.

        Examples:
            Drop the index on the `img` column of the table `my_table` by column name:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(column_name='img')

            Drop the index on the `img` column of the table `my_table` by column reference:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(tbl.img)

            Drop the index `idx1` of the table `my_table` by index name:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(idx_name='idx1')

            Drop the index `idx1` of the table `my_table` by index name, if it exists, otherwise do nothing:
            >>> tbl = pxt.get_table('my_table')
            ... tbl.drop_index(idx_name='idx1', if_not_exists='ignore')

        """

    @overload
    def insert(
        self,
        source: TableDataSource,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self,
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @abc.abstractmethod
    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        """Inserts rows into this table.

        You can insert rows directly by providing a list of dictionaries as the `source`.

        >>> tbl.insert([{'col1': 1, 'col2': 'egg'}, {'col1': 2, 'col2': 'fish'}])

        You can also insert data from any recognized data source by providing a file path or URL.

        >>> tbl.insert('path/to/file.csv')
        >>> tbl.insert('https://example.com/data.xlsx')
        >>> tbl.insert('s3://my-bucket/data.parquet')

        Pixeltable will attempt to infer the format of the source data, unless the optional `source_format`
        parameter is specified. Pixeltable will also attempt to infer the schema of the source data; you can
        override the inferred schema by providing a `schema_overrides` dictionary (which may include all
        columns or just a subset of columns).

        The `source` can also be another table or a [`Query`][pixeltable.Query]:

        >>> tbl.insert(
        ...     other_tbl.select(
        ...         col1=other_tbl.other_col, col2=other_tbl.yet_another_col
        ...     )
        ... )

        For inserting just a single row, there is a convenient shorthand key/value syntax:

        >>> tbl.insert(col1=1, col2='egg')

        Args:
            source: A data source from which data can be imported. Can be any of the following:

                - A list of dictionaries
                - A list of Pydantic model instances
                - A file path or URI of a recognized data source
                - A Pandas `DataFrame`
                - Another Pixeltable table or a `Query`
                - A Hugging Face dataset
            kwargs: (if inserting a single row) Keyword-argument pairs representing column names and values.
                (if inserting multiple rows) Additional keyword arguments are passed to the data source.
            source_format: A hint about the format of the source data. If not specified, Pixeltable will attempt
                to infer the format.
            schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types.
                Any columns not included in `schema_overrides` will have their types inferred as usual.
            on_error: Determines the behavior if an error occurs while evaluating a computed column or detecting an
                invalid media file (such as a corrupt image) for one of the inserted rows.

                - If `on_error='abort'`, then an exception will be raised and the rows will not be inserted.
                - If `on_error='ignore'`, then execution will continue and the rows will be inserted. Any cells
                    with errors will have a `None` value for that cell, with information about the error stored in the
                    corresponding `tbl.col_name.errortype` and `tbl.col_name.errormsg` fields.
            print_stats: If `True`, print statistics about the cost of computed columns.
            return_rows: If `True`, populate `UpdateStatus.rows` with one dict per inserted row, mapping column names
                to their stored or computed values. If `False` (default), `UpdateStatus.rows` is `None`.

        Returns:
            An [`UpdateStatus`][pixeltable.UpdateStatus] object containing information about the update.

        Raises:
            Error: If one of the following conditions occurs:

                - The table is a view or snapshot.
                - The table has been dropped.
                - One of the rows being inserted does not conform to the table schema.
                - An error occurs during processing of computed columns, and `on_error='abort'`.
                - An error occurs while importing data from a source, and `on_error='abort'`.

        Examples:
            Insert two rows into the table `my_table` with three int columns `a`, `b`, and `c`.
            Column `c` is nullable:

            >>> tbl = pxt.get_table('my_table')
            ... tbl.insert([{'a': 1, 'b': 1, 'c': 1}, {'a': 2, 'b': 2}])

            Insert a single row using the alternative syntax:

            >>> tbl.insert(a=3, b=3, c=3)

            Insert rows from a CSV file:

            >>> tbl.insert('path/to/file.csv')

            Insert Pydantic model instances into a table with two `pxt.Int` columns `a` and `b`:

            >>> class MyModel(pydantic.BaseModel):
            ...     a: int
            ...     b: int
            ...
            ...
            ... models = [MyModel(a=1, b=2), MyModel(a=3, b=4)]
            ... tbl.insert(models)
        """

    @abc.abstractmethod
    def compute(
        self,
        source: Sequence[dict[str, Any]] | Sequence[pydantic.BaseModel],
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
    ) -> RowBatch:
        """
        Materialize the computed columns of this table for the given input rows and return the resulting rows
        without persisting them.

        If this table is a view, the input rows are applied to the view's insertable base table (i.e., the root of the
        view hierarchy) and the output rows are the resulting rows of the view, as if the input had been inserted into
        the base:
        - rows that don't satisfy the view's filter are dropped
        - an iterator view can produce multiple output rows per input row

        Args:
            source: Rows to compute, as a sequence of dictionaries or Pydantic model instances. Rows contain
                values for the base table's columns (for a view) or this table's columns; each row must supply
                values for every required (non-nullable, non-computed) column; the same rules as
                [`insert()`][pixeltable.Table.insert] apply.

            on_error: Determines the behavior if an error occurs while evaluating a computed column or detecting an
                invalid media file (such as a corrupt image).

                - If `on_error='abort'`, an exception will be raised.
                - If `on_error='ignore'`, execution will continue and the (possibly partially) completed rows will be
                    returned. Any cells with errors will have a `None` value for that cell, with information about the
                    error recorded in the row's [`errors`][pixeltable.Row]. A row whose view filter fails to
                    evaluate is dropped.

        Returns:
            A [`RowBatch`][pixeltable.RowBatch] of output rows, in input row order (with an iterator's output
            rows in iteration order). Each [`Row`][pixeltable.Row] contains a value for every column of the
            table. [`Row.errors`][pixeltable.Row] holds `{'errortype': ..., 'errormsg': ...}` for each cell that raised,
            keyed by column or index name (only with `on_error='ignore'`).

        Raises:
            Error: If one of the following conditions occurs:

                - The table is a snapshot, a view of a snapshot, or a view defined with a sample clause.
                - The table has been dropped.
                - One of the input rows does not conform to the base table schema.
                - An error occurs during processing of computed columns, and `on_error='abort'`.

        Examples:
            Compute output rows for a table with int columns `a`, `b` and a computed column `c = a + b`:

            >>> tbl = pxt.get_table('my_table')
            ... rows = tbl.compute([{'a': 1, 'b': 1}, {'a': 2, 'b': 2}])
            ... # rows == [{'a': 1, 'b': 1, 'c': 2}, {'a': 2, 'b': 2, 'c': 4}]

            Same with Pydantic model inputs:

            >>> class MyModel(pydantic.BaseModel):
            ...     a: int
            ...     b: int
            ...
            ...
            ... rows = tbl.compute([MyModel(a=1, b=2), MyModel(a=3, b=4)])

            Continue past per-row failures and inspect the per-cell error info:

            >>> rows = tbl.compute(
            ...     [{'a': 0, 'b': 1}, {'a': 2, 'b': 2}], on_error='ignore'
            ... )
            ... # If `c` raised on row 0, rows[0]['c'] is None and rows[0].errors['c']
            ... # contains {'errortype': ..., 'errormsg': ...}.

            Compute view rows from base table input, for a view `my_view` defined over `my_table` with a
            filter `a > 0`:

            >>> v = pxt.get_table('my_view')
            ... rows = v.compute([{'a': 0, 'b': 1}, {'a': 2, 'b': 2}])
            ... # only the second input row satisfies the filter; rows contains its view row
        """

    def _validate_update_value_spec(self, value_spec: dict[str, Any]) -> None:
        from pixeltable import exprs

        for col_name, val in value_spec.items():
            if not isinstance(col_name, str):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'Update specification: dict key must be column name; got {col_name!r}',
                )
            if not is_valid_identifier(col_name):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f'Update specification: invalid column name {col_name!r}'
                )
            if exprs.Expr.from_object(val) is None:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Column {col_name!r}: value is not a recognized literal or expression: {val!r}',
                )

    def _validate_where(self, where: 'exprs.Expr' | None) -> None:
        from pixeltable import exprs

        if where is not None and not isinstance(where, exprs.Expr):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_EXPRESSION,
                f'`where` argument must be a valid Pixeltable expression; got `{type(where)}`',
            )

    def _validate_column_schema(self, schema: Mapping[str, type | ColumnSpec]) -> None:
        from .column import Column

        for name, spec in schema.items():
            if isinstance(spec, dict):
                Column._validate_column_spec(name, spec)

    def _validate_insert_source(self, source: TableDataSource | None) -> None:
        if source is not None and isinstance(source, Sequence) and len(source) == 0:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Cannot insert an empty sequence.')

    def _validate_compute(self) -> None:
        """Raises if compute() is not supported for this table's path."""
        if self._tbl_path.has_snapshot():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'{self._display_str()}: compute() is not supported for snapshots.',
            )
        if self._tbl_path.has_sample_clause():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'{self._display_str()}: compute() is not supported for views defined with a sample clause.',
            )

    def _validate_embedding_args(
        self, embedding: Function | None, string_embed: Function | None, image_embed: Function | None
    ) -> None:
        from pixeltable.func.function import Function

        for name, fn in (('embedding', embedding), ('string_embed', string_embed), ('image_embed', image_embed)):
            if fn is not None and not isinstance(fn, Function):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'`{name}` must be a Pixeltable function; got `{type(fn).__name__}`',
                )

    def _check_mutable(self, op_descr: str) -> None:
        if self._tbl_path.is_snapshot():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'{self._display_str()}: Cannot {op_descr} a {self._display_name()}.',
            )

    def _check_single_column_kwarg(self, method: str, value_form: str, kwargs: Mapping[str, Any]) -> None:
        """Enforce that a single-column method (add_column/add_computed_column) got exactly one col_name= kwarg."""
        if len(kwargs) != 1:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'{method}() requires exactly one keyword argument of the form {value_form}; '
                f'got {len(kwargs)} arguments instead ({", ".join(kwargs.keys())})',
            )

    @abc.abstractmethod
    def update(
        self,
        value_spec: dict[str, Any],
        where: 'exprs.Expr' | None = None,
        cascade: bool = True,
        return_rows: bool = False,
    ) -> UpdateStatus:
        """Update rows in this table.

        Args:
            value_spec: a dictionary mapping column names to literal values or Pixeltable expressions.
            where: a predicate to filter rows to update.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
            return_rows: If `True`, populate `UpdateStatus.rows` with one dict per updated row, mapping column
                names to their new stored values.  If `False` (default), `UpdateStatus.rows` is `None`.

        Returns:
            An [`UpdateStatus`][pixeltable.UpdateStatus] object containing information about the update.

        Examples:
            Set column `int_col` to 1 for all rows:

            >>> tbl.update({'int_col': 1})

            Set column `int_col` to 1 for all rows where `int_col` is 0:

            >>> tbl.update({'int_col': 1}, where=tbl.int_col == 0)

            Set `int_col` to the value of `other_int_col` + 1:

            >>> tbl.update({'int_col': tbl.other_int_col + 1})

            Increment `int_col` by 1 for all rows where `int_col` is 0:

            >>> tbl.update({'int_col': tbl.int_col + 1}, where=tbl.int_col == 0)
        """

    @abc.abstractmethod
    def batch_update(
        self,
        rows: Iterable[dict[str, Any]],
        cascade: bool = True,
        if_not_exists: Literal['error', 'ignore', 'insert'] = 'error',
        return_rows: bool = False,
    ) -> UpdateStatus:
        """Update rows in this table.

        Args:
            rows: an Iterable of dictionaries containing values for the updated columns plus values for the primary key
                  columns.
            cascade: if True, also update all computed columns that transitively depend on the updated columns.
            if_not_exists: Specifies the behavior if a row to update does not exist:

                - `'error'`: Raise an error.
                - `'ignore'`: Skip the row silently.
                - `'insert'`: Insert the row.

            return_rows: If `True`, populate `UpdateStatus.rows` with one dict per affected row, mapping column
                names to their new stored values. Rows newly inserted via `if_not_exists='insert'` are included.
                If `False` (default), `UpdateStatus.rows` is `None`.

        Examples:
            Update the `name` and `age` columns for the rows with ids 1 and 2 (assuming `id` is the primary key).
            If either row does not exist, this raises an error:

            >>> tbl.batch_update(
            ...     [
            ...         {'id': 1, 'name': 'Alice', 'age': 30},
            ...         {'id': 2, 'name': 'Bob', 'age': 40},
            ...     ]
            ... )

            Update the `name` and `age` columns for the row with `id` 1 (assuming `id` is the primary key) and insert
            the row with new `id` 3 (assuming this key does not exist):

            >>> tbl.batch_update(
            ...     [
            ...         {'id': 1, 'name': 'Alice', 'age': 30},
            ...         {'id': 3, 'name': 'Bob', 'age': 40},
            ...     ],
            ...     if_not_exists='insert',
            ... )
        """

    @abc.abstractmethod
    def recompute_columns(
        self,
        *columns: str | ColumnRef,
        where: 'exprs.Expr' | None = None,
        errors_only: bool = False,
        cascade: bool = True,
    ) -> UpdateStatus:
        """Recompute the values in one or more computed columns of this table.

        Args:
            columns: The names or references of the computed columns to recompute.
            where: A predicate to filter rows to recompute.
            errors_only: If True, only run the recomputation for rows that have errors in the column (ie, the column's
                `errortype` property indicates that an error occurred). Only allowed for recomputing a single column.
            cascade: if True, also update all computed columns that transitively depend on the recomputed columns.

        Examples:
            Recompute computed columns `c1` and `c2` for all rows in this table, and everything that transitively
            depends on them:

            >>> tbl.recompute_columns('c1', 'c2')

            Recompute computed column `c1` for all rows in this table, but don't recompute other columns that depend on
            it:

            >>> tbl.recompute_columns(tbl.c1, tbl.c2, cascade=False)

            Recompute column `c1` and its dependents, but only for rows with `c2` == 0:

            >>> tbl.recompute_columns('c1', where=tbl.c2 == 0)

            Recompute column `c1` and its dependents, but only for rows that have errors in it:

            >>> tbl.recompute_columns('c1', errors_only=True)
        """

    @abc.abstractmethod
    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
        """Delete rows in this table.

        Args:
            where: a predicate to filter rows to delete.

        Examples:
            Delete all rows in a table:

            >>> tbl.delete()

            Delete all rows in a table where column `a` is greater than 5:

            >>> tbl.delete(tbl.a > 5)
        """

    @abc.abstractmethod
    def revert(self) -> None:
        """Reverts the table to the previous version.

        .. warning::
            This operation is irreversible.
        """

    @abc.abstractmethod
    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        """
        Returns information about versions of this table, most recent first.

        `get_versions()` is intended for programmatic access to version metadata; for human-readable
        output, use [`history()`][pixeltable.Table.history] instead.

        Args:
            n: if specified, will return at most `n` versions

        Returns:
            A list of [VersionMetadata][pixeltable.VersionMetadata] dictionaries, one per version retrieved, most
            recent first.

        Examples:
            Retrieve metadata about all versions of the table `tbl`:

            >>> tbl.get_versions()

            Retrieve metadata about the most recent 5 versions of the table `tbl`:

            >>> tbl.get_versions(n=5)
        """

    def history(self, n: int | None = None) -> pd.DataFrame:
        """
        Returns a human-readable report about versions of this table.

        `history()` is intended for human-readable output of version metadata; for programmatic access,
        use [`get_versions()`][pixeltable.Table.get_versions] instead.

        Args:
            n: if specified, will return at most `n` versions

        Returns:
            A report with information about each version, one per row, most recent first.

        Examples:
            Report all versions of the table:

            >>> tbl.history()

            Report only the most recent 5 changes to the table:

            >>> tbl.history(n=5)
        """
        versions = self.get_versions(n)
        assert len(versions) > 0
        return pd.DataFrame([list(v.values()) for v in versions], columns=list(versions[0].keys()))
