from typing import Any, Optional

import pixeltable as pxt
from pixeltable import Table


def create_label_studio_project(
        t: Table,
        label_config: str,
        col_mapping: Optional[dict[str, str]] = None,
        title: Optional[str] = None,
        sync_immediately: bool = True,
        **kwargs: Any
) -> None:
    """
    Creates a new Label Studio project and links it to the specified `Table`.

    The required parameter `label_config` specifies the Label Studio project configuration,
    in XML format, as described in the Label Studio documentation. The linked project will
    have one column for each data field in the configuration; for example, if the
    configuration has an entry
    ```
    <Image name="image_obj" value="$image"/>
    ```
    then the linked project will have a column named `image`. In addition, the linked project
    will always have a JSON-typed column `annotations` representing the output.

    By default, Pixeltable will link each of these columns to a column of the specified `Table`
    with the same name. If any of the data fields are missing, an exception will be thrown. If
    the `annotations` column is missing, it will be created. The default names can be overridden
    by specifying an optional `col_mapping`, with Pixeltable column names as keys and Label
    Studio field names as values.

    Args:
        t: The Table to link to.
        label_config: The Label Studio project configuration, in XML format.
        col_mapping: An optional mapping of local column names to remote column names.
        title: An optional title for the Label Studio project. If not specified, the
            name of the `Table` will be used as a default.
        sync_immediately: If `True`, immediately perform an initial synchronization by
            importing all rows of the `Table` as Label Studio tasks.
    """
    from pixeltable.datatransfer.label_studio import LabelStudioProject, ANNOTATIONS_COLUMN

    ls_project = LabelStudioProject.create(title or t.get_name(), label_config, **kwargs)

    # Create a column to hold the annotations, if one does not yet exist.
    if col_mapping is not None and ANNOTATIONS_COLUMN in col_mapping.values():
        local_annotations_column = next(k for k, v in col_mapping.items() if v == ANNOTATIONS_COLUMN)
    else:
        local_annotations_column = ANNOTATIONS_COLUMN
    if local_annotations_column not in t.column_names():
        t[local_annotations_column] = pxt.JsonType(nullable=True)

    # Link the project to `t`, and sync if appropriate.
    t._link(ls_project, col_mapping)
    if sync_immediately:
        t.sync()
