# View

Instances of class `View` are handles to Pixeltable views and snapshots (the latter require `is_snapshot=True` when
creating the view).

Use this handle to query and update the view and to add and drop columns.

`View` instances are created by calling [`pxt.create_view`][pixeltable.create_view]
or [`pxt.get_table`][pixeltable.get_table].

## Overview
| Column Operations                                |                                |
|--------------------------------------------------|--------------------------------|
| [`add_column`][pixeltable.View.add_column]       | Adds a column to the view      |
| [`drop_column`][pixeltable.View.drop_column]     | Removes a column from the view |
| [`rename_column`][pixeltable.View.rename_column] | Renames a column               |

| Data Operations                    |                         |
|------------------------------------|-------------------------|
| [`update`][pixeltable.View.update] | Update rows in the view |

| Versioning                         |                                    |
|------------------------------------|------------------------------------|
| [`revert`][pixeltable.View.update] | Revert the last change to the view |

## ::: pixeltable.View
    options:
      inherited_members: true
      members:
        - add_column
        - drop_column
        - rename_column
        - update
        - revert

