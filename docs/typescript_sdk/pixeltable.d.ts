// Generated TypeScript definitions for Pixeltable
// This file is auto-generated from the Pixeltable public API


// Pixeltable Media Types
// These represent the serialized form of Pixeltable media objects

export interface PixeltableImage {
  /** Base64-encoded image data or URL */
  data?: string;
  /** Image width in pixels */
  width?: number;
  /** Image height in pixels */
  height?: number;
  /** Image mode (e.g., 'RGB', 'RGBA', 'L') */
  mode?: string;
  /** MIME type */
  mimeType?: string;
}

export interface PixeltableVideo {
  /** Base64-encoded video data or URL */
  data?: string;
  /** Video duration in seconds */
  duration?: number;
  /** Video width in pixels */
  width?: number;
  /** Video height in pixels */
  height?: number;
  /** Frame rate */
  fps?: number;
  /** MIME type */
  mimeType?: string;
}

export interface PixeltableAudio {
  /** Base64-encoded audio data or URL */
  data?: string;
  /** Audio duration in seconds */
  duration?: number;
  /** Sample rate in Hz */
  sampleRate?: number;
  /** Number of channels */
  channels?: number;
  /** MIME type */
  mimeType?: string;
}

export interface PixeltableDocument {
  /** Base64-encoded document data or URL */
  data?: string;
  /** Document text content */
  text?: string;
  /** MIME type */
  mimeType?: string;
  /** Number of pages (for PDFs) */
  pages?: number;
}

export interface PixeltableColumnType {
  /** The type class name */
  _classname: string;
  /** Whether the column is nullable */
  nullable: boolean;
  /** Additional type-specific properties */
  [key: string]: any;
}


/**
 * A handle to a table, view, or snapshot. This class is the primary interface through which table operations
 * (queries, insertions, updates, etc.) are performed in Pixeltable.
 * 
 * Every user-invoked operation that runs an ExecNode tree (directly or indirectly) needs to call
 * FileCache.emit_eviction_warnings() at the end of the operation.
 */
export interface Table {
  /**
   * Retrieves metadata associated with this table.
   * 
   * Returns:
   *     A [TableMetadata][pixeltable.TableMetadata] instance containing this table's metadata.
   */
  get_metadata(): Promise<TableMetadata>;
  /**
   * Returns a list of all views and snapshots of this `Table`.
   * 
   * Args:
   *     recursive: If `False`, returns only the immediate successor views of this `Table`. If `True`, returns
   *         all sub-views (including views of views, etc.)
   * 
   * Returns:
   *     A list of view paths.
   */
  list_views(recursive?: boolean): Promise<any[]>;
  /**
   * Return the number of rows in this table.
   */
  count(): Promise<number>;
  /**
   * Return the names of the columns in this table.
   */
  columns(): Promise<any[]>;
  /**
   * Print the table schema.
   */
  describe(): Promise<null>;
  /**
   * Adds multiple columns to the table. The columns must be concrete (non-computed) columns; to add computed
   * columns, use [`add_computed_column()`][pixeltable.catalog.Table.add_computed_column] instead.
   * 
   * The format of the `schema` argument is a dict mapping column names to their types.
   * 
   * Args:
   *     schema: A dictionary mapping column names to types.
   *     if_exists: Determines the behavior if a column already exists. Must be one of the following:
   * 
   *         - `'error'`: an exception will be raised.
   *         - `'ignore'`: do nothing and return.
   *         - `'replace' or 'replace_force'`: drop the existing column and add the new column, if it has no
   *             dependents.
   * 
   *         Note that the `if_exists` parameter is applied to all columns in the schema.
   *         To apply different behaviors to different columns, please use
   *         [`add_column()`][pixeltable.Table.add_column] for each column.
   * 
   * Returns:
   *     Information about the execution status of the operation.
   * 
   * Raises:
   *     Error: If any column name is invalid, or already exists and `if_exists='error'`,
   *         or `if_exists='replace*'` but the column has dependents or is a basetable column.
   * 
   * Examples:
   *     Add multiple columns to the table `my_table`:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... schema = {
   *     ...     'new_col_1': pxt.Int,
   *     ...     'new_col_2': pxt.String,
   *     ... }
   *     ... tbl.add_columns(schema)
   */
  add_columns(schema: Record<string, any>, if_exists?: any): Promise<UpdateStatus>;
  /**
   * Adds an ordinary (non-computed) column to the table.
   * 
   * Args:
   *     kwargs: Exactly one keyword argument of the form `col_name=col_type`.
   *     if_exists: Determines the behavior if the column already exists. Must be one of the following:
   * 
   *         - `'error'`: an exception will be raised.
   *         - `'ignore'`: do nothing and return.
   *         - `'replace' or 'replace_force'`: drop the existing column and add the new column, if it has
   *             no dependents.
   * 
   * Returns:
   *     Information about the execution status of the operation.
   * 
   * Raises:
   *     Error: If the column name is invalid, or already exists and `if_exists='erorr'`,
   *         or `if_exists='replace*'` but the column has dependents or is a basetable column.
   * 
   * Examples:
   *     Add an int column:
   * 
   *     >>> tbl.add_column(new_col=pxt.Int)
   * 
   *     Alternatively, this can also be expressed as:
   * 
   *     >>> tbl.add_columns({'new_col': pxt.Int})
   */
  add_column(kwargs: any | Expr, if_exists?: any): Promise<UpdateStatus>;
  /**
   * Adds a computed column to the table.
   * 
   * Args:
   *     kwargs: Exactly one keyword argument of the form `col_name=expression`.
   *     stored: Whether the column is materialized and stored or computed on demand.
   *     destination: An object store reference for persisting computed files.
   *     print_stats: If `True`, print execution metrics during evaluation.
   *     on_error: Determines the behavior if an error occurs while evaluating the column expression for at least one
   *         row.
   * 
   *         - `'abort'`: an exception will be raised and the column will not be added.
   *         - `'ignore'`: execution will continue and the column will be added. Any rows
   *             with errors will have a `None` value for the column, with information about the error stored in the
   *             corresponding `tbl.col_name.errormsg` and `tbl.col_name.errortype` fields.
   *     if_exists: Determines the behavior if the column already exists. Must be one of the following:
   * 
   *         - `'error'`: an exception will be raised.
   *         - `'ignore'`: do nothing and return.
   *         - `'replace' or 'replace_force'`: drop the existing column and add the new column, iff it has
   *             no dependents.
   * 
   * Returns:
   *     Information about the execution status of the operation.
   * 
   * Raises:
   *     Error: If the column name is invalid or already exists and `if_exists='error'`,
   *         or `if_exists='replace*'` but the column has dependents or is a basetable column.
   * 
   * Examples:
   *     For a table with an image column `frame`, add an image column `rotated` that rotates the image by
   *     90 degrees:
   * 
   *     >>> tbl.add_computed_column(rotated=tbl.frame.rotate(90))
   * 
   *     Do the same, but now the column is unstored:
   * 
   *     >>> tbl.add_computed_column(rotated=tbl.frame.rotate(90), stored=False)
   */
  add_computed_column(stored: boolean | null, destination: string | any | null, kwargs: Expr, print_stats?: boolean, on_error?: any, if_exists?: any): Promise<UpdateStatus>;
  /**
   * Drop a column from the table.
   * 
   * Args:
   *     column: The name or reference of the column to drop.
   *     if_not_exists: Directive for handling a non-existent column. Must be one of the following:
   * 
   *         - `'error'`: raise an error if the column does not exist.
   *         - `'ignore'`: do nothing if the column does not exist.
   * 
   * Raises:
   *     Error: If the column does not exist and `if_exists='error'`,
   *         or if it is referenced by a dependent computed column.
   * 
   * Examples:
   *     Drop the column `col` from the table `my_table` by column name:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_column('col')
   * 
   *     Drop the column `col` from the table `my_table` by column reference:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_column(tbl.col)
   * 
   *     Drop the column `col` from the table `my_table` if it exists, otherwise do nothing:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_col(tbl.col, if_not_exists='ignore')
   */
  drop_column(column: string | any, if_not_exists?: any): Promise<null>;
  /**
   * Rename a column.
   * 
   * Args:
   *     old_name: The current name of the column.
   *     new_name: The new name of the column.
   * 
   * Raises:
   *     Error: If the column does not exist, or if the new name is invalid or already exists.
   * 
   * Examples:
   *     Rename the column `col1` to `col2` of the table `my_table`:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.rename_column('col1', 'col2')
   */
  rename_column(old_name: string, new_name: string): Promise<null>;
  /**
   * Add an embedding index to the table. Once the index is created, it will be automatically kept up-to-date as new
   * rows are inserted into the table.
   * 
   * To add an embedding index, one must specify, at minimum, the column to be indexed and an embedding UDF.
   * Only `String` and `Image` columns are currently supported.
   * 
   * Examples:
   *     Here's an example that uses a
   *     [CLIP embedding][pixeltable.functions.huggingface.clip] to index an image column:
   * 
   *     >>> from pixeltable.functions.huggingface import clip
   *     >>> embedding_fn = clip.using(model_id='openai/clip-vit-base-patch32')
   *     >>> tbl.add_embedding_index(tbl.img, embedding=embedding_fn)
   * 
   *     Once the index is created, similarity lookups can be performed using the `similarity` pseudo-function:
   * 
   *     >>> reference_img = PIL.Image.open('my_image.jpg')
   *     >>> sim = tbl.img.similarity(reference_img)
   *     >>> tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)
   * 
   *     If the embedding UDF is a multimodal embedding (supporting more than one data type), then lookups may be
   *     performed using any of its supported types. In our example, CLIP supports both text and images, so we can
   *     also search for images using a text description:
   * 
   *     >>> sim = tbl.img.similarity('a picture of a train')
   *     >>> tbl.select(tbl.img, sim).order_by(sim, asc=False).limit(5)
   * 
   * Args:
   *     column: The name of, or reference to, the column to be indexed; must be a `String` or `Image` column.
   *     idx_name: An optional name for the index. If not specified, a name such as `'idx0'` will be generated
   *         automatically. If specified, the name must be unique for this table and a valid pixeltable column name.
   *     embedding: The UDF to use for the embedding. Must be a UDF that accepts a single argument of type `String`
   *         or `Image` (as appropriate for the column being indexed) and returns a fixed-size 1-dimensional
   *         array of floats.
   *     string_embed: An optional UDF to use for the string embedding component of this index.
   *         Can be used in conjunction with `image_embed` to construct multimodal embeddings manually, by
   *         specifying different embedding functions for different data types.
   *     image_embed: An optional UDF to use for the image embedding component of this index.
   *         Can be used in conjunction with `string_embed` to construct multimodal embeddings manually, by
   *         specifying different embedding functions for different data types.
   *     metric: Distance metric to use for the index; one of `'cosine'`, `'ip'`, or `'l2'`.
   *         The default is `'cosine'`.
   *     if_exists: Directive for handling an existing index with the same name. Must be one of the following:
   * 
   *         - `'error'`: raise an error if an index with the same name already exists.
   *         - `'ignore'`: do nothing if an index with the same name already exists.
   *         - `'replace'` or `'replace_force'`: replace the existing index with the new one.
   * 
   * Raises:
   *     Error: If an index with the specified name already exists for the table and `if_exists='error'`, or if
   *         the specified column does not exist.
   * 
   * Examples:
   *     Add an index to the `img` column of the table `my_table`:
   * 
   *     >>> from pixeltable.functions.huggingface import clip
   *     >>> tbl = pxt.get_table('my_table')
   *     >>> embedding_fn = clip.using(model_id='openai/clip-vit-base-patch32')
   *     >>> tbl.add_embedding_index(tbl.img, embedding=embedding_fn)
   * 
   *     Alternatively, the `img` column may be specified by name:
   * 
   *     >>> tbl.add_embedding_index('img', embedding=embedding_fn)
   * 
   *     Add a second index to the `img` column, using the inner product as the distance metric,
   *     and with a specific name:
   * 
   *     >>> tbl.add_embedding_index(
   *     ...     tbl.img,
   *     ...     idx_name='ip_idx',
   *     ...     embedding=embedding_fn,
   *     ...     metric='ip'
   *     ... )
   * 
   *     Add an index using separately specified string and image embeddings:
   * 
   *     >>> tbl.add_embedding_index(
   *     ...     tbl.img,
   *     ...     string_embed=string_embedding_fn,
   *     ...     image_embed=image_embedding_fn
   *     ... )
   */
  add_embedding_index(): Promise<any>;
  /**
   * Drop an embedding index from the table. Either a column name or an index name (but not both) must be
   * specified. If a column name or reference is specified, it must be a column containing exactly one
   * embedding index; otherwise the specific index name must be provided instead.
   * 
   * Args:
   *     column: The name of, or reference to, the column from which to drop the index.
   *             The column must have only one embedding index.
   *     idx_name: The name of the index to drop.
   *     if_not_exists: Directive for handling a non-existent index. Must be one of the following:
   * 
   *         - `'error'`: raise an error if the index does not exist.
   *         - `'ignore'`: do nothing if the index does not exist.
   * 
   *         Note that `if_not_exists` parameter is only applicable when an `idx_name` is specified
   *         and it does not exist, or when `column` is specified and it has no index.
   *         `if_not_exists` does not apply to non-exisitng column.
   * 
   * Raises:
   *     Error: If `column` is specified, but the column does not exist, or it contains no embedding
   *         indices and `if_not_exists='error'`, or the column has multiple embedding indices.
   *     Error: If `idx_name` is specified, but the index is not an embedding index, or
   *         the index does not exist and `if_not_exists='error'`.
   * 
   * Examples:
   *     Drop the embedding index on the `img` column of the table `my_table` by column name:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_embedding_index(column='img')
   * 
   *     Drop the embedding index on the `img` column of the table `my_table` by column reference:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_embedding_index(column=tbl.img)
   * 
   *     Drop the embedding index `idx1` of the table `my_table` by index name:
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_embedding_index(idx_name='idx1')
   * 
   *     Drop the embedding index `idx1` of the table `my_table` by index name, if it exists, otherwise do nothing:
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_embedding_index(idx_name='idx1', if_not_exists='ignore')
   */
  drop_embedding_index(column: string | any | null, idx_name: string | null, if_not_exists?: any): Promise<null>;
  /**
   * Drop an index from the table. Either a column name or an index name (but not both) must be
   * specified. If a column name or reference is specified, it must be a column containing exactly one index;
   * otherwise the specific index name must be provided instead.
   * 
   * Args:
   *     column: The name of, or reference to, the column from which to drop the index.
   *             The column must have only one embedding index.
   *     idx_name: The name of the index to drop.
   *     if_not_exists: Directive for handling a non-existent index. Must be one of the following:
   * 
   *         - `'error'`: raise an error if the index does not exist.
   *         - `'ignore'`: do nothing if the index does not exist.
   * 
   *         Note that `if_not_exists` parameter is only applicable when an `idx_name` is specified
   *         and it does not exist, or when `column` is specified and it has no index.
   *         `if_not_exists` does not apply to non-exisitng column.
   * 
   * Raises:
   *     Error: If `column` is specified, but the column does not exist, or it contains no
   *         indices or multiple indices.
   *     Error: If `idx_name` is specified, but the index does not exist.
   * 
   * Examples:
   *     Drop the index on the `img` column of the table `my_table` by column name:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_index(column_name='img')
   * 
   *     Drop the index on the `img` column of the table `my_table` by column reference:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_index(tbl.img)
   * 
   *     Drop the index `idx1` of the table `my_table` by index name:
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_index(idx_name='idx1')
   * 
   *     Drop the index `idx1` of the table `my_table` by index name, if it exists, otherwise do nothing:
   *     >>> tbl = pxt.get_table('my_table')
   *     ... tbl.drop_index(idx_name='idx1', if_not_exists='ignore')
   */
  drop_index(column: string | any | null, idx_name: string | null, if_not_exists?: any): Promise<null>;
  /**
   * Update rows in this table.
   * 
   * Args:
   *     value_spec: a dictionary mapping column names to literal values or Pixeltable expressions.
   *     where: a predicate to filter rows to update.
   *     cascade: if True, also update all computed columns that transitively depend on the updated columns.
   * 
   * Returns:
   *     An [`UpdateStatus`][pixeltable.UpdateStatus] object containing information about the update.
   * 
   * Examples:
   *     Set column `int_col` to 1 for all rows:
   * 
   *     >>> tbl.update({'int_col': 1})
   * 
   *     Set column `int_col` to 1 for all rows where `int_col` is 0:
   * 
   *     >>> tbl.update({'int_col': 1}, where=tbl.int_col == 0)
   * 
   *     Set `int_col` to the value of `other_int_col` + 1:
   * 
   *     >>> tbl.update({'int_col': tbl.other_int_col + 1})
   * 
   *     Increment `int_col` by 1 for all rows where `int_col` is 0:
   * 
   *     >>> tbl.update({'int_col': tbl.int_col + 1}, where=tbl.int_col == 0)
   */
  update(value_spec: Record<string, any>, where: Expr | null, cascade?: boolean): Promise<UpdateStatus>;
  /**
   * Update rows in this table.
   * 
   * Args:
   *     rows: an Iterable of dictionaries containing values for the updated columns plus values for the primary key
   *           columns.
   *     cascade: if True, also update all computed columns that transitively depend on the updated columns.
   *     if_not_exists: Specifies the behavior if a row to update does not exist:
   * 
   *         - `'error'`: Raise an error.
   *         - `'ignore'`: Skip the row silently.
   *         - `'insert'`: Insert the row.
   * 
   * Examples:
   *     Update the `name` and `age` columns for the rows with ids 1 and 2 (assuming `id` is the primary key).
   *     If either row does not exist, this raises an error:
   * 
   *     >>> tbl.update([{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 40}])
   * 
   *     Update the `name` and `age` columns for the row with `id` 1 (assuming `id` is the primary key) and insert
   *     the row with new `id` 3 (assuming this key does not exist):
   * 
   *     >>> tbl.update(
   *     ...     [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 3, 'name': 'Bob', 'age': 40}],
   *     ...     if_not_exists='insert')
   */
  batch_update(rows: any, cascade?: boolean, if_not_exists?: any): Promise<UpdateStatus>;
  /**
   * Recompute the values in one or more computed columns of this table.
   * 
   * Args:
   *     columns: The names or references of the computed columns to recompute.
   *     where: A predicate to filter rows to recompute.
   *     errors_only: If True, only run the recomputation for rows that have errors in the column (ie, the column's
   *         `errortype` property indicates that an error occurred). Only allowed for recomputing a single column.
   *     cascade: if True, also update all computed columns that transitively depend on the recomputed columns.
   * 
   * Examples:
   *     Recompute computed columns `c1` and `c2` for all rows in this table, and everything that transitively
   *     depends on them:
   * 
   *     >>> tbl.recompute_columns('c1', 'c2')
   * 
   *     Recompute computed column `c1` for all rows in this table, but don't recompute other columns that depend on
   *     it:
   * 
   *     >>> tbl.recompute_columns(tbl.c1, tbl.c2, cascade=False)
   * 
   *     Recompute column `c1` and its dependents, but only for rows with `c2` == 0:
   * 
   *     >>> tbl.recompute_columns('c1', where=tbl.c2 == 0)
   * 
   *     Recompute column `c1` and its dependents, but only for rows that have errors in it:
   * 
   *     >>> tbl.recompute_columns('c1', errors_only=True)
   */
  recompute_columns(): Promise<any>;
  /**
   * Delete rows in this table.
   * 
   * Args:
   *     where: a predicate to filter rows to delete.
   * 
   * Examples:
   *     Delete all rows in a table:
   * 
   *     >>> tbl.delete()
   * 
   *     Delete all rows in a table where column `a` is greater than 5:
   * 
   *     >>> tbl.delete(tbl.a > 5)
   */
  delete(where: Expr | null): Promise<UpdateStatus>;
  /**
   * Reverts the table to the previous version.
   * 
   * .. warning::
   *     This operation is irreversible.
   */
  revert(): Promise<null>;
  /**
   * Unlinks this table's external stores.
   * 
   * Args:
   *     stores: If specified, will unlink only the specified named store or list of stores. If not specified,
   *         will unlink all of this table's external stores.
   *     ignore_errors (bool): If `True`, no exception will be thrown if a specified store is not linked
   *         to this table.
   *     delete_external_data (bool): If `True`, then the external data store will also be deleted. WARNING: This
   *         is a destructive operation that will delete data outside Pixeltable, and cannot be undone.
   */
  unlink_external_stores(stores: string | any[] | null, delete_external_data?: boolean, ignore_errors?: boolean): Promise<null>;
  /**
   * Synchronizes this table with its linked external stores.
   * 
   * Args:
   *     stores: If specified, will synchronize only the specified named store or list of stores. If not specified,
   *         will synchronize all of this table's external stores.
   *     export_data: If `True`, data from this table will be exported to the external stores during synchronization.
   *     import_data: If `True`, data from the external stores will be imported to this table during synchronization.
   */
  sync(stores: string | any[] | null, export_data?: boolean, import_data?: boolean): Promise<UpdateStatus>;
  /**
   * Returns information about versions of this table, most recent first.
   * 
   * `get_versions()` is intended for programmatic access to version metadata; for human-readable
   * output, use [`history()`][pixeltable.Table.history] instead.
   * 
   * Args:
   *     n: if specified, will return at most `n` versions
   * 
   * Returns:
   *     A list of [VersionMetadata][pixeltable.VersionMetadata] dictionaries, one per version retrieved, most
   *     recent first.
   * 
   * Examples:
   *     Retrieve metadata about all versions of the table `tbl`:
   * 
   *     >>> tbl.get_versions()
   * 
   *     Retrieve metadata about the most recent 5 versions of the table `tbl`:
   * 
   *     >>> tbl.get_versions(n=5)
   */
  get_versions(n: number | null): Promise<any[]>;
  /**
   * Returns a human-readable report about versions of this table.
   * 
   * `history()` is intended for human-readable output of version metadata; for programmatic access,
   * use [`get_versions()`][pixeltable.Table.get_versions] instead.
   * 
   * Args:
   *     n: if specified, will return at most `n` versions
   * 
   * Returns:
   *     A report with information about each version, one per row, most recent first.
   * 
   * Examples:
   *     Report all versions of the table:
   * 
   *     >>> tbl.history()
   * 
   *     Report only the most recent 5 changes to the table:
   * 
   *     >>> tbl.history(n=5)
   */
  history(n: number | null): Promise<any>;
}

/**
 * Metadata for a Pixeltable table.
 */
export interface TableMetadata {
}

/**
 * Statistics about the counts of rows affected by a table operation.
 */
export interface RowCountStats {
}

/**
 * Information about changes to table data or table schema
 */
export interface UpdateStatus {
}

export interface DataFrameResultSet {
}

/**
 * Rules for using state in subclasses:
 * - all state except for components and slot_idx is shared between copies of an Expr
 * - slot_idx is set during analysis (DataFrame.show())
 * - during eval(), components can only be accessed via self.components; any Exprs outside of that won't
 *   have slot_idx set
 */
export interface Expr {
}

/**
 * !!! abstract "Usage Documentation"
 *     [Models](../concepts/models.md)
 * 
 * A base class for creating Pydantic models.
 * 
 */
export interface Tool {
}

/**
 * !!! abstract "Usage Documentation"
 *     [Models](../concepts/models.md)
 * 
 * A base class for creating Pydantic models.
 * 
 */
export interface Tools {
}

/**
 * Base class for Pixeltable iterators.
 */
export interface ComponentIterator {
}

/**
 * Iterator over tiles of an image. Each image will be divided into tiles of size `tile_size`, and the tiles will be
 * iterated over in row-major order (left-to-right, then top-to-bottom). An optional `overlap` parameter may be
 * specified. If the tiles do not exactly cover the image, then the rightmost and bottommost tiles will be padded with
 * blackspace, so that the output images all have the exact size `tile_size`.
 * 
 */
export interface TileIterator {
}

/**
 * Iterator over frames of a video. At most one of `fps` or `num_frames` may be specified. If `fps` is specified,
 * then frames will be extracted at the specified rate (frames per second). If `num_frames` is specified, then the
 * exact number of frames will be extracted. If neither is specified, then all frames will be extracted. The first
 * frame of the video will always be extracted, and the remaining frames will be spaced as evenly as possible.
 * 
 */
export interface FrameIterator {
}

/**
 * str(object='') -> str
 * str(bytes_or_buffer[, encoding[, errors]]) -> str
 * 
 * Create a new string object from the given object. If encoding or
 * errors is specified, then the object must expose a data buffer
 */
export interface Audio {
}

/**
 * Base class for the Pixeltable type-hint family. Subclasses of this class are meant to be used as type hints, both
 * in schema definitions and in UDF signatures. Whereas `ColumnType`s are instantiable and carry semantic information
 * about the Pixeltable type system, `_PxtType` subclasses are purely for convenience: they are not instantiable and
 * must be resolved to a `ColumnType` (by calling `ColumnType.from_python_type()`) in order to do anything meaningful
 * with them.
 */
export interface Json {
}

/**
 * str(object='') -> str
 * str(bytes_or_buffer[, encoding[, errors]]) -> str
 * 
 * Create a new string object from the given object. If encoding or
 * errors is specified, then the object must expose a data buffer
 */
export interface Video {
}

// Pixeltable Public API Functions
export interface Pixeltable {
  /**
   * Get the complete public API registry.
   * 
   * This is the directory of all Pixeltable public APIs with their metadata.
   * Useful for building tools, documentation, LLM integrations, and more.
   * 
   * Returns:
   *     Dictionary mapping qualified names to API metadata, including:
   *     - signature: Function signature
   *     - parameters: Parameter names, types, defaults
   *     - return_type: Return type annotation
   *     - docstring: Full documentation
   *     - module: Source module
   * 
   * Example:
   *     >>> registry = get_public_api_registry()
   *     >>> print(f"Total APIs: {len(registry)}")
   *     >>> print(registry['pixeltable.globals.create_table']['signature'])
   */
  get_public_api_registry(): Promise<Record<string, any>>;

  /**
   * Check if an object is marked as public API.
   * 
   * Args:
   *     obj: Object to check (function, class, method)
   * 
   * Returns:
   *     True if marked as public API, False otherwise
   * 
   * Example:
   *     >>> import pixeltable as pxt
   *     >>> is_public_api(pxt.create_table)
   *     True
   *     >>> is_public_api(some_internal_function)
   *     False
   */
  is_public_api(obj: any): Promise<boolean>;

  array(elements: any): Promise<Expr>;

  /**
   * Configure logging.
   * 
   * Args:
   *     to_stdout: if True, also log to stdout
   *     level: default log level
   *     add: comma-separated list of 'module name:log level' pairs; ex.: add='video:10'
   *     remove: comma-separated list of module names
   */
  configure_logging(to_stdout: boolean | null, level: number | null, add: string | null, remove: string | null): Promise<null>;

  /**
   * Create a directory.
   * 
   * Args:
   *     path: Path to the directory.
   *     if_exists: Directive regarding how to handle if the path already exists.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return the existing directory handle
   *         - `'replace'`: if the existing directory is empty, drop it and create a new one
   *         - `'replace_force'`: drop the existing directory and all its children, and create a new one
   *     parents: Create missing parent directories.
   * 
   * Returns:
   *     A handle to the newly created directory, or to an already existing directory at the path when
   *         `if_exists='ignore'`. Please note the existing directory may not be empty.
   * 
   * Raises:
   *     Error: If
   * 
   *         - the path is invalid, or
   *         - the path already exists and `if_exists='error'`, or
   *         - the path already exists and is not a directory, or
   *         - an error occurs while attempting to create the directory.
   * 
   * Examples:
   *     >>> pxt.create_dir('my_dir')
   * 
   *     Create a subdirectory:
   * 
   *     >>> pxt.create_dir('my_dir.sub_dir')
   * 
   *     Create a subdirectory only if it does not already exist, otherwise do nothing:
   * 
   *     >>> pxt.create_dir('my_dir.sub_dir', if_exists='ignore')
   * 
   *     Create a directory and replace if it already exists:
   * 
   *     >>> pxt.create_dir('my_dir', if_exists='replace_force')
   * 
   *     Create a subdirectory along with its ancestors:
   * 
   *     >>> pxt.create_dir('parent1.parent2.sub_dir', parents=True)
   */
  create_dir(path: string, if_exists?: any, parents?: boolean): Promise<any | null>;

  /**
   * Create a snapshot of an existing table object (which itself can be a view or a snapshot or a base table).
   * 
   * Args:
   *     path_str: A name for the snapshot; can be either a simple name such as `my_snapshot`, or a pathname such as
   *         `dir1.my_snapshot`.
   *     base: [`Table`][pixeltable.Table] (i.e., table or view or snapshot) or [`DataFrame`][pixeltable.DataFrame] to
   *         base the snapshot on.
   *     additional_columns: If specified, will add these columns to the snapshot once it is created. The format
   *         of the `additional_columns` parameter is identical to the format of the `schema_or_df` parameter in
   *         [`create_table`][pixeltable.create_table].
   *     iterator: The iterator to use for this snapshot. If specified, then this snapshot will be a one-to-many view of
   *         the base table.
   *     num_retained_versions: Number of versions of the view to retain.
   *     comment: Optional comment for the snapshot.
   *     media_validation: Media validation policy for the snapshot.
   * 
   *         - `'on_read'`: validate media files at query time
   *         - `'on_write'`: validate media files during insert/update operations
   *     if_exists: Directive regarding how to handle if the path already exists.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return the existing snapshot handle
   *         - `'replace'`: if the existing snapshot has no dependents, drop and replace it with a new one
   *         - `'replace_force'`: drop the existing snapshot and all its dependents, and create a new one
   * 
   * Returns:
   *     A handle to the [`Table`][pixeltable.Table] representing the newly created snapshot.
   *         Please note the schema or base of the existing snapshot may not match those provided in the call.
   * 
   * Raises:
   *     Error: if
   * 
   *         - the path is invalid, or
   *         - the path already exists and `if_exists='error'`, or
   *         - the path already exists and is not a snapshot, or
   *         - an error occurs while attempting to create the snapshot.
   * 
   * Examples:
   *     Create a snapshot `my_snapshot` of a table `my_table`:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... snapshot = pxt.create_snapshot('my_snapshot', tbl)
   * 
   *     Create a snapshot `my_snapshot` of a view `my_view` with additional int column `col3`,
   *     if `my_snapshot` does not already exist:
   * 
   *     >>> view = pxt.get_table('my_view')
   *     ... snapshot = pxt.create_snapshot(
   *     ...     'my_snapshot', view, additional_columns={'col3': pxt.Int}, if_exists='ignore'
   *     ... )
   * 
   *     Create a snapshot `my_snapshot` on a table `my_table`, and replace any existing snapshot named `my_snapshot`:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... snapshot = pxt.create_snapshot('my_snapshot', tbl, if_exists='replace_force')
   */
  create_snapshot(path_str: string, base: Table | any, additional_columns: Record<string, any> | null, iterator: any[] | null, num_retained_versions?: number, comment?: string, media_validation?: any, if_exists?: any): Promise<Table | null>;

  /**
   * Create a new base table. Exactly one of `schema` or `source` must be provided.
   * 
   * If a `schema` is provided, then an empty table will be created with the specified schema.
   * 
   * If a `source` is provided, then Pixeltable will attempt to infer a data source format and table schema from the
   * contents of the specified data, and the data will be imported from the specified source into the new table. The
   * source format and/or schema can be specified directly via the `source_format` and `schema_overrides` parameters.
   * 
   * Args:
   *     path: Pixeltable path (qualified name) of the table, such as `'my_table'` or `'my_dir.my_subdir.my_table'`.
   *     schema: Schema for the new table, mapping column names to Pixeltable types.
   *     source: A data source (file, URL, DataFrame, or list of rows) to import from.
   *     source_format: Must be used in conjunction with a `source`.
   *         If specified, then the given format will be used to read the source data. (Otherwise,
   *         Pixeltable will attempt to infer the format from the source data.)
   *     schema_overrides: Must be used in conjunction with a `source`.
   *         If specified, then columns in `schema_overrides` will be given the specified types.
   *         (Pixeltable will attempt to infer the types of any columns not specified.)
   *     on_error: Determines the behavior if an error occurs while evaluating a computed column or detecting an
   *         invalid media file (such as a corrupt image) for one of the inserted rows.
   * 
   *         - If `on_error='abort'`, then an exception will be raised and the rows will not be inserted.
   *         - If `on_error='ignore'`, then execution will continue and the rows will be inserted. Any cells
   *             with errors will have a `None` value for that cell, with information about the error stored in the
   *             corresponding `tbl.col_name.errortype` and `tbl.col_name.errormsg` fields.
   *     primary_key: An optional column name or list of column names to use as the primary key(s) of the
   *         table.
   *     num_retained_versions: Number of versions of the table to retain.
   *     comment: An optional comment; its meaning is user-defined.
   *     media_validation: Media validation policy for the table.
   * 
   *         - `'on_read'`: validate media files at query time
   *         - `'on_write'`: validate media files during insert/update operations
   *     if_exists: Determines the behavior if a table already exists at the specified path location.
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return the existing table handle
   *         - `'replace'`: if the existing table has no views or snapshots, drop and replace it with a new one;
   *             raise an error if the existing table has views or snapshots
   *         - `'replace_force'`: drop the existing table and all its views and snapshots, and create a new one
   *     extra_args: Must be used in conjunction with a `source`. If specified, then additional arguments will be
   *         passed along to the source data provider.
   * 
   * Returns:
   *     A handle to the newly created table, or to an already existing table at the path when `if_exists='ignore'`.
   *         Please note the schema of the existing table may not match the schema provided in the call.
   * 
   * Raises:
   *     Error: if
   * 
   *         - the path is invalid, or
   *         - the path already exists and `if_exists='error'`, or
   *         - the path already exists and is not a table, or
   *         - an error occurs while attempting to create the table, or
   *         - an error occurs while attempting to import data from the source.
   * 
   * Examples:
   *     Create a table with an int and a string column:
   * 
   *     >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.String})
   * 
   *     Create a table from a select statement over an existing table `orig_table` (this will create a new table
   *     containing the exact contents of the query):
   * 
   *     >>> tbl1 = pxt.get_table('orig_table')
   *     ... tbl2 = pxt.create_table('new_table', tbl1.where(tbl1.col1 < 10).select(tbl1.col2))
   * 
   *     Create a table if it does not already exist, otherwise get the existing table:
   * 
   *     >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.String}, if_exists='ignore')
   * 
   *     Create a table with an int and a float column, and replace any existing table:
   * 
   *     >>> tbl = pxt.create_table('my_table', schema={'col1': pxt.Int, 'col2': pxt.Float}, if_exists='replace')
   * 
   *     Create a table from a CSV file:
   * 
   *     >>> tbl = pxt.create_table('my_table', source='data.csv')
   */
  create_table(): Promise<any>;

  /**
   * Create a view of an existing table object (which itself can be a view or a snapshot or a base table).
   * 
   * Args:
   *     path: A name for the view; can be either a simple name such as `my_view`, or a pathname such as
   *         `dir1.my_view`.
   *     base: [`Table`][pixeltable.Table] (i.e., table or view or snapshot) or [`DataFrame`][pixeltable.DataFrame] to
   *         base the view on.
   *     additional_columns: If specified, will add these columns to the view once it is created. The format
   *         of the `additional_columns` parameter is identical to the format of the `schema_or_df` parameter in
   *         [`create_table`][pixeltable.create_table].
   *     is_snapshot: Whether the view is a snapshot. Setting this to `True` is equivalent to calling
   *         [`create_snapshot`][pixeltable.create_snapshot].
   *     iterator: The iterator to use for this view. If specified, then this view will be a one-to-many view of
   *         the base table.
   *     num_retained_versions: Number of versions of the view to retain.
   *     comment: Optional comment for the view.
   *     media_validation: Media validation policy for the view.
   * 
   *         - `'on_read'`: validate media files at query time
   *         - `'on_write'`: validate media files during insert/update operations
   *     if_exists: Directive regarding how to handle if the path already exists.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return the existing view handle
   *         - `'replace'`: if the existing view has no dependents, drop and replace it with a new one
   *         - `'replace_force'`: drop the existing view and all its dependents, and create a new one
   * 
   * Returns:
   *     A handle to the [`Table`][pixeltable.Table] representing the newly created view. If the path already
   *         exists and `if_exists='ignore'`, returns a handle to the existing view. Please note the schema
   *         or the base of the existing view may not match those provided in the call.
   * 
   * Raises:
   *     Error: if
   * 
   *         - the path is invalid, or
   *         - the path already exists and `if_exists='error'`, or
   *         - the path already exists and is not a view, or
   *         - an error occurs while attempting to create the view.
   * 
   * Examples:
   *     Create a view `my_view` of an existing table `my_table`, filtering on rows where `col1` is greater than 10:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... view = pxt.create_view('my_view', tbl.where(tbl.col1 > 10))
   * 
   *     Create a view `my_view` of an existing table `my_table`, filtering on rows where `col1` is greater than 10,
   *     and if it not already exist. Otherwise, get the existing view named `my_view`:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... view = pxt.create_view('my_view', tbl.where(tbl.col1 > 10), if_exists='ignore')
   * 
   *     Create a view `my_view` of an existing table `my_table`, filtering on rows where `col1` is greater than 100,
   *     and replace any existing view named `my_view`:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   *     ... view = pxt.create_view('my_view', tbl.where(tbl.col1 > 100), if_exists='replace_force')
   */
  create_view(path: string, base: Table | any, additional_columns: Record<string, any> | null, iterator: any[] | null, is_snapshot?: boolean, num_retained_versions?: number, comment?: string, media_validation?: any, if_exists?: any): Promise<Table | null>;

  /**
   * Remove a directory.
   * 
   * Args:
   *     path: Name or path of the directory.
   *     force: If `True`, will also drop all tables and subdirectories of this directory, recursively, along
   *         with any views or snapshots that depend on any of the dropped tables.
   *     if_not_exists: Directive regarding how to handle if the path does not exist.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return
   * 
   * Raises:
   *     Error: If the path
   * 
   *         - is invalid, or
   *         - does not exist and `if_not_exists='error'`, or
   *         - is not designate a directory, or
   *         - is a direcotory but is not empty and `force=False`.
   * 
   * Examples:
   *     Remove a directory, if it exists and is empty:
   *     >>> pxt.drop_dir('my_dir')
   * 
   *     Remove a subdirectory:
   * 
   *     >>> pxt.drop_dir('my_dir.sub_dir')
   * 
   *     Remove an existing directory if it is empty, but do nothing if it does not exist:
   * 
   *     >>> pxt.drop_dir('my_dir.sub_dir', if_not_exists='ignore')
   * 
   *     Remove an existing directory and all its contents:
   * 
   *     >>> pxt.drop_dir('my_dir', force=True)
   */
  drop_dir(path: string, force?: boolean, if_not_exists?: any): Promise<null>;

  /**
   * Drop a table, view, snapshot, or replica.
   * 
   * Args:
   *     table: Fully qualified name or table handle of the table to be dropped; or a remote URI of a cloud replica to
   *         be deleted.
   *     force: If `True`, will also drop all views and sub-views of this table.
   *     if_not_exists: Directive regarding how to handle if the path does not exist.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return
   * 
   * Raises:
   *     Error: if the qualified name
   * 
   *         - is invalid, or
   *         - does not exist and `if_not_exists='error'`, or
   *         - does not designate a table object, or
   *         - designates a table object but has dependents and `force=False`.
   * 
   * Examples:
   *     Drop a table by its fully qualified name:
   *     >>> pxt.drop_table('subdir.my_table')
   * 
   *     Drop a table by its handle:
   *     >>> t = pxt.get_table('subdir.my_table')
   *     ... pxt.drop_table(t)
   * 
   *     Drop a table if it exists, otherwise do nothing:
   *     >>> pxt.drop_table('subdir.my_table', if_not_exists='ignore')
   * 
   *     Drop a table and all its dependents:
   *     >>> pxt.drop_table('subdir.my_table', force=True)
   */
  drop_table(table: string | Table, force?: boolean, if_not_exists?: any): Promise<null>;

  /**
   * Get the contents of a Pixeltable directory.
   * 
   * Args:
   *     dir_path: Path to the directory. Defaults to the root directory.
   *     recursive: If `False`, returns only those tables and directories that are directly contained in specified
   *         directory; if `True`, returns all tables and directories that are descendants of the specified directory,
   *         recursively.
   * 
   * Returns:
   *     A [`DirContents`][pixeltable.DirContents] object representing the contents of the specified directory.
   * 
   * Raises:
   *     Error: If the path does not exist or does not designate a directory.
   * 
   * Examples:
   *     Get contents of top-level directory:
   * 
   *     >>> pxt.get_dir_contents()
   * 
   *     Get contents of 'dir1':
   * 
   *     >>> pxt.get_dir_contents('dir1')
   */
  get_dir_contents(): Promise<any>;

  /**
   * Get a handle to an existing table, view, or snapshot.
   * 
   * Args:
   *     path: Path to the table.
   *     if_not_exists: Directive regarding how to handle if the path does not exist.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return `None`
   * 
   * Returns:
   *     A handle to the [`Table`][pixeltable.Table].
   * 
   * Raises:
   *     Error: If the path does not exist or does not designate a table object.
   * 
   * Examples:
   *     Get handle for a table in the top-level directory:
   * 
   *     >>> tbl = pxt.get_table('my_table')
   * 
   *     For a table in a subdirectory:
   * 
   *     >>> tbl = pxt.get_table('subdir.my_table')
   * 
   *     Handles to views and snapshots are retrieved in the same way:
   * 
   *     >>> tbl = pxt.get_table('my_snapshot')
   * 
   *     Get a handle to a specific version of a table:
   * 
   *     >>> tbl = pxt.get_table('my_table:722')
   */
  get_table(path: string, if_not_exists?: any): Promise<Table | null>;

  /**
   * Initializes the Pixeltable environment.
   */
  init(config_overrides: Record<string, any> | null): Promise<null>;

  /**
   * List the directories in a directory.
   * 
   * Args:
   *     path: Name or path of the directory.
   *     recursive: If `True`, lists all descendants of this directory recursively.
   * 
   * Returns:
   *     List of directory paths.
   * 
   * Raises:
   *     Error: If `path_str` does not exist or does not designate a directory.
   * 
   * Examples:
   *     >>> cl.list_dirs('my_dir', recursive=True)
   *     ['my_dir', 'my_dir.sub_dir1']
   */
  list_dirs(path?: string, recursive?: boolean): Promise<any[]>;

  /**
   * Returns information about all registered functions.
   * 
   * Returns:
   *     Pandas DataFrame with columns 'Path', 'Name', 'Parameters', 'Return Type', 'Is Agg', 'Library'
   */
  list_functions(): Promise<any>;

  /**
   * List the [`Table`][pixeltable.Table]s in a directory.
   * 
   * Args:
   *     dir_path: Path to the directory. Defaults to the root directory.
   *     recursive: If `False`, returns only those tables that are directly contained in specified directory; if
   *         `True`, returns all tables that are descendants of the specified directory, recursively.
   * 
   * Returns:
   *     A list of [`Table`][pixeltable.Table] paths.
   * 
   * Raises:
   *     Error: If the path does not exist or does not designate a directory.
   * 
   * Examples:
   *     List tables in top-level directory:
   * 
   *     >>> pxt.list_tables()
   * 
   *     List tables in 'dir1':
   * 
   *     >>> pxt.list_tables('dir1')
   */
  list_tables(dir_path?: string, recursive?: boolean): Promise<any[]>;

  /**
   * List the contents of a Pixeltable directory.
   * 
   * This function returns a Pandas DataFrame representing a human-readable listing of the specified directory,
   * including various attributes such as version and base table, as appropriate.
   * 
   * To get a programmatic list of the directory's contents, use [get_dir_contents()][pixeltable.get_dir_contents]
   * instead.
   */
  ls(path?: string): Promise<any>;

  /**
   * Move a schema object to a new directory and/or rename a schema object.
   * 
   * Args:
   *     path: absolute path to the existing schema object.
   *     new_path: absolute new path for the schema object.
   *     if_exists: Directive regarding how to handle if a schema object already exists at the new path.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return
   *     if_not_exists: Directive regarding how to handle if the source path does not exist.
   *         Must be one of the following:
   * 
   *         - `'error'`: raise an error
   *         - `'ignore'`: do nothing and return
   * 
   * Raises:
   *     Error: If path does not exist or new_path already exists.
   * 
   * Examples:
   *     Move a table to a different directory:
   * 
   *     >>>> pxt.move('dir1.my_table', 'dir2.my_table')
   * 
   *     Rename a table:
   * 
   *     >>>> pxt.move('dir1.my_table', 'dir1.new_name')
   */
  move(path: string, new_path: string, if_exists?: any, if_not_exists?: any): Promise<null>;

  /**
   * Publishes a replica of a local Pixeltable table to Pixeltable cloud. A given table can be published to at most one
   * URI per Pixeltable cloud database.
   * 
   * Args:
   *     source: Path or table handle of the local table to be published.
   *     destination_uri: Remote URI where the replica will be published, such as `'pxt://org_name/my_dir/my_table'`.
   *     bucket_name: The name of the bucket to use to store replica's data. The bucket must be registered with
   *         Pixeltable cloud. If no `bucket_name` is provided, the default storage bucket for the destination
   *         database will be used.
   *     access: Access control for the replica.
   * 
   *         - `'public'`: Anyone can access this replica.
   *         - `'private'`: Only the host organization can access.
   */
  publish(source: string | Table, destination_uri: string, bucket_name: string | null, access?: any): Promise<null>;

  /**
   * Retrieve a replica from Pixeltable cloud as a local table. This will create a full local copy of the replica in a
   * way that preserves the table structure of the original source data. Once replicated, the local table can be
   * queried offline just as any other Pixeltable table.
   * 
   * Args:
   *     remote_uri: Remote URI of the table to be replicated, such as `'pxt://org_name/my_dir/my_table'`.
   *     local_path: Local table path where the replica will be created, such as `'my_new_dir.my_new_tbl'`. It can be
   *         the same or different from the cloud table name.
   * 
   * Returns:
   *     A handle to the newly created local replica table.
   */
  replicate(remote_uri: string, local_path: string): Promise<Table>;

  /**
   * Specifies a Pixeltable UDF to be used as an LLM tool with customizable metadata. See the documentation for
   * [pxt.tools()][pixeltable.tools] for more details.
   * 
   * Args:
   *     fn: The UDF to use as a tool.
   *     name: The name of the tool. If not specified, then the unqualified name of the UDF will be used by default.
   *     description: The description of the tool. If not specified, then the entire contents of the UDF docstring
   *         will be used by default.
   * 
   * Returns:
   *     A `Tool` instance that can be passed to an LLM tool-calling API.
   */
  tool(fn: any, name: string | null, description: string | null): Promise<Tool>;

  /**
   * Specifies a collection of UDFs to be used as LLM tools. Pixeltable allows any UDF to be used as an input into an
   * LLM tool-calling API. To use one or more UDFs as tools, wrap them in a `pxt.tools` call and pass the return value
   * to an LLM API.
   * 
   * The UDFs can be specified directly or wrapped inside a [pxt.tool()][pixeltable.tool] invocation. If a UDF is
   * specified directly, the tool name will be the (unqualified) UDF name, and the tool description will consist of the
   * entire contents of the UDF docstring. If a UDF is wrapped in a `pxt.tool()` invocation, then the name and/or
   * description may be customized.
   * 
   * Args:
   *     args: The UDFs to use as tools.
   * 
   * Returns:
   *     A `Tools` instance that can be passed to an LLM tool-calling API or invoked to generate tool results.
   * 
   * Examples:
   *     Create a tools instance with a single UDF:
   * 
   *     >>> tools = pxt.tools(stock_price)
   * 
   *     Create a tools instance with several UDFs:
   * 
   *     >>> tools = pxt.tools(stock_price, weather_quote)
   * 
   *     Create a tools instance, some of whose UDFs have customized metadata:
   * 
   *     >>> tools = pxt.tools(
   *     ...     stock_price,
   *     ...     pxt.tool(weather_quote, description='Returns information about the weather in a particular location.'),
   *     ...     pxt.tool(traffic_quote, name='traffic_conditions'),
   *     ... )
   */
  tools(args: any | Tool): Promise<Tools>;

  /**
   * Creates a new base table from a JSON file. This is a convenience method and is
   * equivalent to calling `import_data(table_path, json.loads(file_contents, **kwargs), ...)`, where `file_contents`
   * is the contents of the specified `filepath_or_url`.
   * 
   * Args:
   *     tbl_path: The name of the table to create.
   *     filepath_or_url: The path or URL of the JSON file.
   *     schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
   *         (see [`import_rows()`][pixeltable.io.import_rows]).
   *     primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
   *     num_retained_versions: The number of retained versions of the table
   *         (see [`create_table()`][pixeltable.create_table]).
   *     comment: A comment to attach to the table (see [`create_table()`][pixeltable.create_table]).
   *     kwargs: Additional keyword arguments to pass to `json.loads`.
   * 
   * Returns:
   *     A handle to the newly created [`Table`][pixeltable.Table].
   */
  import_json(tbl_path: string, filepath_or_url: string, schema_overrides: Record<string, any> | null, primary_key: string | any[] | null, kwargs: any, num_retained_versions?: number, comment?: string): Promise<Table>;

  /**
   * Creates a new base table from a list of dictionaries. The dictionaries must be of the
   * form `{column_name: value, ...}`. Pixeltable will attempt to infer the schema of the table from the
   * supplied data, using the most specific type that can represent all the values in a column.
   * 
   * If `schema_overrides` is specified, then for each entry `(column_name, type)` in `schema_overrides`,
   * Pixeltable will force the specified column to the specified type (and will not attempt any type inference
   * for that column).
   * 
   * All column types of the new table will be nullable unless explicitly specified as non-nullable in
   * `schema_overrides`.
   * 
   * Args:
   *     tbl_path: The qualified name of the table to create.
   *     rows: The list of dictionaries to import.
   *     schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
   *         as described above.
   *     primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
   *     num_retained_versions: The number of retained versions of the table
   *         (see [`create_table()`][pixeltable.create_table]).
   *     comment: A comment to attach to the table (see [`create_table()`][pixeltable.create_table]).
   * 
   * Returns:
   *     A handle to the newly created [`Table`][pixeltable.Table].
   */
  import_rows(tbl_path: string, rows: any[], schema_overrides: Record<string, any> | null, primary_key: string | any[] | null, num_retained_versions?: number, comment?: string): Promise<Table>;

  /**
   * Create a new Label Studio project and link it to the specified [`Table`][pixeltable.Table].
   * 
   * - A tutorial notebook with fully worked examples can be found here:
   *   [Using Label Studio for Annotations with Pixeltable](https://pixeltable.readme.io/docs/label-studio)
   * 
   * The required parameter `label_config` specifies the Label Studio project configuration,
   * in XML format, as described in the Label Studio documentation. The linked project will
   * have one column for each data field in the configuration; for example, if the
   * configuration has an entry
   * ```
   * <Image name="image_obj" value="$image"/>
   * ```
   * then the linked project will have a column named `image`. In addition, the linked project
   * will always have a JSON-typed column `annotations` representing the output.
   * 
   * By default, Pixeltable will link each of these columns to a column of the specified [`Table`][pixeltable.Table]
   * with the same name. If any of the data fields are missing, an exception will be raised. If
   * the `annotations` column is missing, it will be created. The default names can be overridden
   * by specifying an optional `col_mapping`, with Pixeltable column names as keys and Label
   * Studio field names as values. In all cases, the Pixeltable columns must have types that are
   * consistent with their corresponding Label Studio fields; otherwise, an exception will be raised.
   * 
   * The API key and URL for a valid Label Studio server must be specified in Pixeltable config. Either:
   * 
   * * Set the `LABEL_STUDIO_API_KEY` and `LABEL_STUDIO_URL` environment variables; or
   * * Specify `api_key` and `url` fields in the `label-studio` section of `$PIXELTABLE_HOME/config.toml`.
   * 
   * __Requirements:__
   * 
   * - `pip install label-studio-sdk`
   * - `pip install boto3` (if using S3 import storage)
   * 
   * Args:
   *     t: The table to link to.
   *     label_config: The Label Studio project configuration, in XML format.
   *     name: An optional name for the new project in Pixeltable. If specified, must be a valid
   *         Pixeltable identifier and must not be the name of any other external data store
   *         linked to `t`. If not specified, a default name will be used of the form
   *         `ls_project_0`, `ls_project_1`, etc.
   *     title: An optional title for the Label Studio project. This is the title that annotators
   *         will see inside Label Studio. Unlike `name`, it does not need to be an identifier and
   *         does not need to be unique. If not specified, the table name `t.name` will be used.
   *     media_import_method: The method to use when transferring media files to Label Studio:
   * 
   *         - `post`: Media will be sent to Label Studio via HTTP post. This should generally only be used for
   *             prototyping; due to restrictions in Label Studio, it can only be used with projects that have
   *             just one data field, and does not scale well.
   *         - `file`: Media will be sent to Label Studio as a file on the local filesystem. This method can be
   *             used if Pixeltable and Label Studio are running on the same host.
   *         - `url`: Media will be sent to Label Studio as externally accessible URLs. This method cannot be
   *             used with local media files or with media generated by computed columns.
   *         The default is `post`.
   *     col_mapping: An optional mapping of local column names to Label Studio fields.
   *     sync_immediately: If `True`, immediately perform an initial synchronization by
   *         exporting all rows of the table as Label Studio tasks.
   *     s3_configuration: If specified, S3 import storage will be configured for the new project. This can only
   *         be used with `media_import_method='url'`, and if `media_import_method='url'` and any of the media data is
   *         referenced by `s3://` URLs, then it must be specified in order for such media to display correctly
   *         in the Label Studio interface.
   * 
   *         The items in the `s3_configuration` dictionary correspond to kwarg
   *         parameters of the Label Studio `connect_s3_import_storage` method, as described in the
   *         [Label Studio connect_s3_import_storage docs](https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.connect_s3_import_storage).
   *         `bucket` must be specified; all other parameters are optional. If credentials are not specified explicitly,
   *         Pixeltable will attempt to retrieve them from the environment (such as from `~/.aws/credentials`).
   *         If a title is not specified, Pixeltable will use the default `'Pixeltable-S3-Import-Storage'`.
   *         All other parameters use their Label Studio defaults.
   *     kwargs: Additional keyword arguments are passed to the `start_project` method in the Label
   *         Studio SDK, as described in the
   *         [Label Studio start_project docs](https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.start_project).
   * 
   * Returns:
   *     An `UpdateStatus` representing the status of any synchronization operations that occurred.
   * 
   * Examples:
   *     Create a Label Studio project whose tasks correspond to videos stored in the `video_col`
   *     column of the table `tbl`:
   * 
   *     >>> config = """
   *     ... <View>
   *     ...     <Video name="video_obj" value="$video_col"/>
   *     ...     <Choices name="video-category" toName="video" showInLine="true">
   *     ...         <Choice value="city"/>
   *     ...         <Choice value="food"/>
   *     ...         <Choice value="sports"/>
   *     ...     </Choices>
   *     ... </View>
   *     ... """
   *     >>> create_label_studio_project(tbl, config)
   * 
   *     Create a Label Studio project with the same configuration, using `media_import_method='url'`,
   *     whose media are stored in an S3 bucket:
   * 
   *     >>> create_label_studio_project(
   *     ...     tbl,
   *     ...     config,
   *     ...     media_import_method='url',
   *     ...     s3_configuration={'bucket': 'my-bucket', 'region_name': 'us-east-2'}
   *     ... )
   */
  create_label_studio_project(t: Table, label_config: string, name: string | null, title: string | null, col_mapping: Record<string, any> | null, s3_configuration: Record<string, any> | null, kwargs: any, media_import_method?: any, sync_immediately?: boolean): Promise<UpdateStatus>;

  /**
   * Export images from a Pixeltable table as a Voxel51 dataset. The data must consist of a single column
   * (or expression) containing image data, along with optional additional columns containing labels. Currently, only
   * classification and detection labels are supported.
   * 
   * The [Working with Voxel51 in Pixeltable](https://docs.pixeltable.com/examples/vision/voxel51) tutorial contains a
   * fully worked example showing how to export data from a Pixeltable table and load it into Voxel51.
   * 
   * Images in the dataset that already exist on disk will be exported directly, in whatever format they
   * are stored in. Images that are not already on disk (such as frames extracted using a
   * [`FrameIterator`][pixeltable.iterators.FrameIterator]) will first be written to disk in the specified
   * `image_format`.
   * 
   * The label parameters accept one or more sets of labels of each type. If a single `Expr` is provided, then it will
   * be exported as a single set of labels with a default name such as `classifications`.
   * (The single set of labels may still containing multiple individual labels; see below.)
   * If a list of `Expr`s is provided, then each one will be exported as a separate set of labels with a default name
   * such as `classifications`, `classifications_1`, etc. If a dictionary of `Expr`s is provided, then each entry will
   * be exported as a set of labels with the specified name.
   * 
   * __Requirements:__
   * 
   * - `pip install fiftyone`
   * 
   * Args:
   *     tbl: The table from which to export data.
   *     images: A column or expression that contains the images to export.
   *     image_format: The format to use when writing out images for export.
   *     classifications: Optional image classification labels. If a single `Expr` is provided, it must be a table
   *         column or an expression that evaluates to a list of dictionaries. Each dictionary in the list corresponds
   *         to an image class and must have the following structure:
   * 
   *         ```python
   *         {'label': 'zebra', 'confidence': 0.325}
   *         ```
   * 
   *         If multiple `Expr`s are provided, each one must evaluate to a list of such dictionaries.
   *     detections: Optional image detection labels. If a single `Expr` is provided, it must be a table column or an
   *         expression that evaluates to a list of dictionaries. Each dictionary in the list corresponds to an image
   *         detection, and must have the following structure:
   * 
   *         ```python
   *         {
   *             'label': 'giraffe',
   *             'confidence': 0.99,
   *             'bounding_box': [0.081, 0.836, 0.202, 0.136]  # [x, y, w, h], fractional coordinates
   *         }
   *         ```
   * 
   *         If multiple `Expr`s are provided, each one must evaluate to a list of such dictionaries.
   * 
   * Returns:
   *     A Voxel51 dataset.
   * 
   * Example:
   *     Export the images in the `image` column of the table `tbl` as a Voxel51 dataset, using classification
   *     labels from `tbl.classifications`:
   * 
   *     >>> export_images_as_fo_dataset(
   *     ...     tbl,
   *     ...     tbl.image,
   *     ...     classifications=tbl.classifications
   *     ... )
   * 
   *     See the [Working with Voxel51 in Pixeltable](https://docs.pixeltable.com/examples/vision/voxel51) tutorial
   *     for a fully worked example.
   */
  export_images_as_fo_dataset(): Promise<any>;

  /**
   * Create a new base table from a Huggingface dataset, or dataset dict with multiple splits.
   *     Requires `datasets` library to be installed.
   * 
   * Args:
   *     table_path: Path to the table.
   *     dataset: Huggingface [`datasets.Dataset`](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset)
   *         or [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict)
   *         to insert into the table.
   *     schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
   *         name `name` will be given type `type`, instead of being inferred from the `Dataset` or `DatasetDict`.
   *         The keys in `schema_overrides` should be the column names of the `Dataset` or `DatasetDict` (whether or not
   *         they are valid Pixeltable identifiers).
   *     primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
   *     kwargs: Additional arguments to pass to `create_table`.
   *         An argument of `column_name_for_split` must be provided if the source is a DatasetDict.
   *         This column name will contain the split information. If None, no split information will be stored.
   * 
   * Returns:
   *     A handle to the newly created [`Table`][pixeltable.Table].
   */
  import_huggingface_dataset(): Promise<any>;

  /**
   * Creates a new base table from a csv file. This is a convenience method and is equivalent
   * to calling `import_pandas(table_path, pd.read_csv(filepath_or_buffer, **kwargs), schema=schema)`.
   * See the Pandas documentation for [`read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
   * for more details.
   * 
   * Returns:
   *     A handle to the newly created [`Table`][pixeltable.Table].
   */
  import_csv(tbl_name: string, filepath_or_buffer: string | any, schema_overrides: Record<string, any> | null, primary_key: string | any[] | null, kwargs: any, num_retained_versions?: number, comment?: string): Promise<Table>;

  /**
   * Creates a new base table from an Excel (.xlsx) file. This is a convenience method and is
   * equivalent to calling `import_pandas(table_path, pd.read_excel(io, *args, **kwargs), schema=schema)`.
   * See the Pandas documentation for [`read_excel`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)
   * for more details.
   * 
   * Returns:
   *     A handle to the newly created [`Table`][pixeltable.Table].
   */
  import_excel(tbl_name: string, io: string | any, schema_overrides: Record<string, any> | null, primary_key: string | any[] | null, kwargs: any, num_retained_versions?: number, comment?: string): Promise<Table>;

  /**
   * Creates a new base table from a Pandas
   * [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), with the
   * specified name. The schema of the table will be inferred from the DataFrame.
   * 
   * The column names of the new table will be identical to those in the DataFrame, as long as they are valid
   * Pixeltable identifiers. If a column name is not a valid Pixeltable identifier, it will be normalized according to
   * the following procedure:
   * - first replace any non-alphanumeric characters with underscores;
   * - then, preface the result with the letter 'c' if it begins with a number or an underscore;
   * - then, if there are any duplicate column names, suffix the duplicates with '_2', '_3', etc., in column order.
   * 
   * Args:
   *     tbl_name: The name of the table to create.
   *     df: The Pandas `DataFrame`.
   *     schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
   *         name `name` will be given type `type`, instead of being inferred from the `DataFrame`. The keys in
   *         `schema_overrides` should be the column names of the `DataFrame` (whether or not they are valid
   *         Pixeltable identifiers).
   * 
   * Returns:
   *     A handle to the newly created [`Table`][pixeltable.Table].
   */
  import_pandas(tbl_name: string, df: any, schema_overrides: Record<string, any> | null, primary_key: string | any[] | null, num_retained_versions?: number, comment?: string): Promise<Table>;

  /**
   * Exports a dataframe's data to one or more Parquet files. Requires pyarrow to be installed.
   * 
   * It additionally writes the pixeltable metadata in a json file, which would otherwise
   * not be available in the parquet format.
   * 
   * Args:
   *     table_or_df : Table or Dataframe to export.
   *     parquet_path : Path to directory to write the parquet files to.
   *     partition_size_bytes : The maximum target size for each chunk. Default 100_000_000 bytes.
   *     inline_images : If True, images are stored inline in the parquet file. This is useful
   *                     for small images, to be imported as pytorch dataset. But can be inefficient
   *                     for large images, and cannot be imported into pixeltable.
   *                     If False, will raise an error if the Dataframe has any image column.
   *                     Default False.
   */
  export_parquet(table_or_df: Table | any, parquet_path: any, partition_size_bytes?: number, inline_images?: boolean): Promise<null>;

  /**
   * Creates a new base table from a Parquet file or set of files. Requires pyarrow to be installed.
   * 
   * Args:
   *     table: Fully qualified name of the table to import the data into.
   *     parquet_path: Path to an individual Parquet file or directory of Parquet files.
   *     schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
   *         name `name` will be given type `type`, instead of being inferred from the Parquet dataset. The keys in
   *         `schema_overrides` should be the column names of the Parquet dataset (whether or not they are valid
   *         Pixeltable identifiers).
   *     primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
   *     kwargs: Additional arguments to pass to `create_table`.
   * 
   * Returns:
   *     A handle to the newly created table.
   */
  import_parquet(table: string, parquet_path: string, schema_overrides: Record<string, any> | null, primary_key: string | any[] | null, kwargs: any): Promise<Table>;

  /**
   * Helper for @overload to raise when called.
   */
  _overload_dummy(args: any, kwds: any): Promise<any>;

}

declare const pixeltable: Pixeltable;
export default pixeltable;