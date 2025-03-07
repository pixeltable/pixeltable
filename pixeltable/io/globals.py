from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import Table, exprs
from pixeltable.env import Env
from pixeltable.io.external_store import SyncStatus

if TYPE_CHECKING:
    import fiftyone as fo  # type: ignore[import-untyped]


from .utils import find_or_create_table, normalize_import_parameters, normalize_schema_names


def _infer_schema_from_rows(
    rows: list[dict[str, Any]], schema_overrides: dict[str, Any], primary_key: list[str]
) -> dict[str, pxt.ColumnType]:
    schema: dict[str, pxt.ColumnType] = {}
    cols_with_nones: set[str] = set()

    for n, row in enumerate(rows):
        for col_name, value in row.items():
            if col_name in schema_overrides:
                # We do the insertion here; this will ensure that the column order matches the order
                # in which the column names are encountered in the input data, even if `schema_overrides`
                # is specified.
                if col_name not in schema:
                    schema[col_name] = schema_overrides[col_name]
            elif value is not None:
                # If `key` is not in `schema_overrides`, then we infer its type from the data.
                # The column type will always be nullable by default.
                col_type = pxt.ColumnType.infer_literal_type(value, nullable=col_name not in primary_key)
                if col_type is None:
                    raise excs.Error(
                        f'Could not infer type for column `{col_name}`; the value in row {n} has an unsupported type: {type(value)}'
                    )
                if col_name not in schema:
                    schema[col_name] = col_type
                else:
                    supertype = schema[col_name].supertype(col_type)
                    if supertype is None:
                        raise excs.Error(
                            f'Could not infer type of column `{col_name}`; the value in row {n} does not match preceding type {schema[col_name]}: {value!r}\n'
                            'Consider specifying the type explicitly in `schema_overrides`.'
                        )
                    schema[col_name] = supertype
            else:
                cols_with_nones.add(col_name)

    entirely_none_cols = cols_with_nones - schema.keys()
    if len(entirely_none_cols) > 0:
        # A column can only end up in `entirely_none_cols` if it was not in `schema_overrides` and
        # was not encountered in any row with a non-None value.
        raise excs.Error(
            f'The following columns have no non-null values: {", ".join(entirely_none_cols)}\n'
            'Consider specifying the type(s) explicitly in `schema_overrides`.'
        )
    return schema


def create_label_studio_project(
    t: Table,
    label_config: str,
    name: Optional[str] = None,
    title: Optional[str] = None,
    media_import_method: Literal['post', 'file', 'url'] = 'post',
    col_mapping: Optional[dict[str, str]] = None,
    sync_immediately: bool = True,
    s3_configuration: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> SyncStatus:
    """
    Create a new Label Studio project and link it to the specified [`Table`][pixeltable.Table].

    - A tutorial notebook with fully worked examples can be found here:
      [Using Label Studio for Annotations with Pixeltable](https://pixeltable.readme.io/docs/label-studio)

    The required parameter `label_config` specifies the Label Studio project configuration,
    in XML format, as described in the Label Studio documentation. The linked project will
    have one column for each data field in the configuration; for example, if the
    configuration has an entry
    ```
    <Image name="image_obj" value="$image"/>
    ```
    then the linked project will have a column named `image`. In addition, the linked project
    will always have a JSON-typed column `annotations` representing the output.

    By default, Pixeltable will link each of these columns to a column of the specified [`Table`][pixeltable.Table]
    with the same name. If any of the data fields are missing, an exception will be raised. If
    the `annotations` column is missing, it will be created. The default names can be overridden
    by specifying an optional `col_mapping`, with Pixeltable column names as keys and Label
    Studio field names as values. In all cases, the Pixeltable columns must have types that are
    consistent with their corresponding Label Studio fields; otherwise, an exception will be raised.

    The API key and URL for a valid Label Studio server must be specified in Pixeltable config. Either:

    * Set the `LABEL_STUDIO_API_KEY` and `LABEL_STUDIO_URL` environment variables; or
    * Specify `api_key` and `url` fields in the `label-studio` section of `$PIXELTABLE_HOME/config.toml`.

    __Requirements:__

    - `pip install label-studio-sdk`
    - `pip install boto3` (if using S3 import storage)

    Args:
        t: The table to link to.
        label_config: The Label Studio project configuration, in XML format.
        name: An optional name for the new project in Pixeltable. If specified, must be a valid
            Pixeltable identifier and must not be the name of any other external data store
            linked to `t`. If not specified, a default name will be used of the form
            `ls_project_0`, `ls_project_1`, etc.
        title: An optional title for the Label Studio project. This is the title that annotators
            will see inside Label Studio. Unlike `name`, it does not need to be an identifier and
            does not need to be unique. If not specified, the table name `t.name` will be used.
        media_import_method: The method to use when transferring media files to Label Studio:

            - `post`: Media will be sent to Label Studio via HTTP post. This should generally only be used for
                prototyping; due to restrictions in Label Studio, it can only be used with projects that have
                just one data field, and does not scale well.
            - `file`: Media will be sent to Label Studio as a file on the local filesystem. This method can be
                used if Pixeltable and Label Studio are running on the same host.
            - `url`: Media will be sent to Label Studio as externally accessible URLs. This method cannot be
                used with local media files or with media generated by computed columns.
            The default is `post`.
        col_mapping: An optional mapping of local column names to Label Studio fields.
        sync_immediately: If `True`, immediately perform an initial synchronization by
            exporting all rows of the table as Label Studio tasks.
        s3_configuration: If specified, S3 import storage will be configured for the new project. This can only
            be used with `media_import_method='url'`, and if `media_import_method='url'` and any of the media data is
            referenced by `s3://` URLs, then it must be specified in order for such media to display correctly
            in the Label Studio interface.

            The items in the `s3_configuration` dictionary correspond to kwarg
            parameters of the Label Studio `connect_s3_import_storage` method, as described in the
            [Label Studio connect_s3_import_storage docs](https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.connect_s3_import_storage).
            `bucket` must be specified; all other parameters are optional. If credentials are not specified explicitly,
            Pixeltable will attempt to retrieve them from the environment (such as from `~/.aws/credentials`). If a title is not
            specified, Pixeltable will use the default `'Pixeltable-S3-Import-Storage'`. All other parameters use their Label
            Studio defaults.
        kwargs: Additional keyword arguments are passed to the `start_project` method in the Label
            Studio SDK, as described in the
            [Label Studio start_project docs](https://labelstud.io/sdk/project.html#label_studio_sdk.project.Project.start_project).

    Returns:
        A `SyncStatus` representing the status of any synchronization operations that occurred.

    Examples:
        Create a Label Studio project whose tasks correspond to videos stored in the `video_col` column of the table `tbl`:

        >>> config = \"\"\"
            <View>
                <Video name="video_obj" value="$video_col"/>
                <Choices name="video-category" toName="video" showInLine="true">
                    <Choice value="city"/>
                    <Choice value="food"/>
                    <Choice value="sports"/>
                </Choices>
            </View>\"\"\"
            create_label_studio_project(tbl, config)

        Create a Label Studio project with the same configuration, using `media_import_method='url'`,
        whose media are stored in an S3 bucket:

        >>> create_label_studio_project(
                tbl,
                config,
                media_import_method='url',
                s3_configuration={'bucket': 'my-bucket', 'region_name': 'us-east-2'}
            )
    """
    Env.get().require_package('label_studio_sdk')

    from pixeltable.io.label_studio import LabelStudioProject

    ls_project = LabelStudioProject.create(
        t, label_config, name, title, media_import_method, col_mapping, s3_configuration, **kwargs
    )

    # Link the project to `t`, and sync if appropriate.
    t._link_external_store(ls_project)
    if sync_immediately:
        return t.sync()
    else:
        return SyncStatus.empty()


def import_rows(
    tbl_path: str,
    rows: list[dict[str, Any]],
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
) -> Table:
    """
    Creates a new base table from a list of dictionaries. The dictionaries must be of the
    form `{column_name: value, ...}`. Pixeltable will attempt to infer the schema of the table from the
    supplied data, using the most specific type that can represent all the values in a column.

    If `schema_overrides` is specified, then for each entry `(column_name, type)` in `schema_overrides`,
    Pixeltable will force the specified column to the specified type (and will not attempt any type inference
    for that column).

    All column types of the new table will be nullable unless explicitly specified as non-nullable in
    `schema_overrides`.

    Args:
        tbl_path: The qualified name of the table to create.
        rows: The list of dictionaries to import.
        schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
            as described above.
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
        num_retained_versions: The number of retained versions of the table (see [`create_table()`][pixeltable.create_table]).
        comment: A comment to attach to the table (see [`create_table()`][pixeltable.create_table]).

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    schema_overrides, primary_key = normalize_import_parameters(schema_overrides, primary_key)
    row_schema = _infer_schema_from_rows(rows, schema_overrides, primary_key)
    schema, pxt_pk, _ = normalize_schema_names(row_schema, primary_key, schema_overrides, True)

    table = find_or_create_table(
        tbl_path, schema, primary_key=pxt_pk, num_retained_versions=num_retained_versions, comment=comment
    )
    table.insert(rows)
    return table


def import_json(
    tbl_path: str,
    filepath_or_url: str,
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs: Any,
) -> Table:
    """
    Creates a new base table from a JSON file. This is a convenience method and is
    equivalent to calling `import_data(table_path, json.loads(file_contents, **kwargs), ...)`, where `file_contents`
    is the contents of the specified `filepath_or_url`.

    Args:
        tbl_path: The name of the table to create.
        filepath_or_url: The path or URL of the JSON file.
        schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
            (see [`import_rows()`][pixeltable.io.import_rows]).
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
        num_retained_versions: The number of retained versions of the table (see [`create_table()`][pixeltable.create_table]).
        comment: A comment to attach to the table (see [`create_table()`][pixeltable.create_table]).
        kwargs: Additional keyword arguments to pass to `json.loads`.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    import json
    import urllib.parse
    import urllib.request

    # TODO Consolidate this logic with other places where files/URLs are parsed
    parsed = urllib.parse.urlparse(filepath_or_url)
    if len(parsed.scheme) <= 1 or parsed.scheme == 'file':
        # local file path
        if len(parsed.scheme) <= 1:
            filepath = filepath_or_url
        else:
            filepath = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        with open(filepath) as fp:
            contents = fp.read()
    else:
        # URL
        contents = urllib.request.urlopen(filepath_or_url).read()
    rows = json.loads(contents, **kwargs)

    schema_overrides, primary_key = normalize_import_parameters(schema_overrides, primary_key)
    row_schema = _infer_schema_from_rows(rows, schema_overrides, primary_key)
    schema, pxt_pk, col_mapping = normalize_schema_names(row_schema, primary_key, schema_overrides, False)

    # Convert all rows to insertable format - not needed, misnamed columns and types are errors in the incoming row format
    if col_mapping is not None:
        tbl_rows = [
            {field if col_mapping is None else col_mapping[field]: val for field, val in row.items()} for row in rows
        ]
    else:
        tbl_rows = rows

    table = find_or_create_table(
        tbl_path, schema, primary_key=pxt_pk, num_retained_versions=num_retained_versions, comment=comment
    )

    table.insert(tbl_rows)
    return table


def export_images_as_fo_dataset(
    tbl: pxt.Table,
    images: exprs.Expr,
    image_format: str = 'webp',
    classifications: Union[exprs.Expr, list[exprs.Expr], dict[str, exprs.Expr], None] = None,
    detections: Union[exprs.Expr, list[exprs.Expr], dict[str, exprs.Expr], None] = None,
) -> 'fo.Dataset':
    """
    Export images from a Pixeltable table as a Voxel51 dataset. The data must consist of a single column
    (or expression) containing image data, along with optional additional columns containing labels. Currently, only
    classification and detection labels are supported.

    The [Working with Voxel51 in Pixeltable](https://docs.pixeltable.com/docs/working-with-voxel51) tutorial contains a
    fully worked example showing how to export data from a Pixeltable table and load it into Voxel51.

    Images in the dataset that already exist on disk will be exported directly, in whatever format they
    are stored in. Images that are not already on disk (such as frames extracted using a
    [`FrameIterator`][pixeltable.iterators.FrameIterator]) will first be written to disk in the specified
    `image_format`.

    The label parameters accept one or more sets of labels of each type. If a single `Expr` is provided, then it will
    be exported as a single set of labels with a default name such as `classifications`.
    (The single set of labels may still containing multiple individual labels; see below.)
    If a list of `Expr`s is provided, then each one will be exported as a separate set of labels with a default name
    such as `classifications`, `classifications_1`, etc. If a dictionary of `Expr`s is provided, then each entry will
    be exported as a set of labels with the specified name.

    __Requirements:__

    - `pip install fiftyone`

    Args:
        tbl: The table from which to export data.
        images: A column or expression that contains the images to export.
        image_format: The format to use when writing out images for export.
        classifications: Optional image classification labels. If a single `Expr` is provided, it must be a table
            column or an expression that evaluates to a list of dictionaries. Each dictionary in the list corresponds
            to an image class and must have the following structure:

            ```python
            {'label': 'zebra', 'confidence': 0.325}
            ```

            If multiple `Expr`s are provided, each one must evaluate to a list of such dictionaries.
        detections: Optional image detection labels. If a single `Expr` is provided, it must be a table column or an
            expression that evaluates to a list of dictionaries. Each dictionary in the list corresponds to an image
            detection, and must have the following structure:

            ```python
            {
                'label': 'giraffe',
                'confidence': 0.99,
                'bounding_box': [0.081, 0.836, 0.202, 0.136]  # [x, y, w, h], fractional coordinates
            }
            ```

            If multiple `Expr`s are provided, each one must evaluate to a list of such dictionaries.

    Returns:
        A Voxel51 dataset.

    Example:
        Export the images in the `image` column of the table `tbl` as a Voxel51 dataset, using classification
        labels from `tbl.classifications`:

        >>> export_as_fiftyone(
        ...     tbl,
        ...     tbl.image,
        ...     classifications=tbl.classifications
        ... )

        See the [Working with Voxel51 in Pixeltable](https://docs.pixeltable.com/docs/working-with-voxel51) tutorial
        for a fully worked example.
    """
    Env.get().require_package('fiftyone')

    import fiftyone as fo

    from pixeltable.io.fiftyone import PxtImageDatasetImporter

    if not images.col_type.is_image_type():
        raise excs.Error(f'`images` must be an expression of type Image (got {images.col_type._to_base_str()})')

    return fo.Dataset.from_importer(
        PxtImageDatasetImporter(tbl, images, image_format, classifications=classifications, detections=detections)
    )
