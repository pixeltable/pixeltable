# InsertableTable

Instances of class `InsertableTable` are handles to Pixeltable tables.

Use this handle to query and update the table and to add and drop columns.

`InsertableTable` instances are created by calling [`pxt.create_table`][pixeltable.create_table]
or [`pxt.get_table`][pixeltable.get_table].

## Overview
| Column Operations | |
|------------|-----------------------------------------------------|
| [`add_column`][pixeltable.InsertableTable.add_column] | Adds a column to the table |
| [`drop_column`][pixeltable.InsertableTable.drop_column] | Remove a column from the table |
| [`rename_column`][pixeltable.InsertableTable.rename_column] | Rename a column |

| Data Operations | |
|------------|-----------------------------------------------------|
| [`insert`][pixeltable.InsertableTable.insert] | Insert rows into table |
| [`update`][pixeltable.InsertableTable.update] | Upate rows in table |
| [`delete`][pixeltable.InsertableTable.delete] | Delete rows from table |

| Versioning | |
|------------|-----------------------------------------------------|
| [`revert`][pixeltable.InsertableTable.revert] | Reverts the last change |

## ::: pixeltable.InsertableTable
    options:
      inherited_members: true
      members:
        - add_column
        - drop_column
        - rename_column
        - insert
        - update
        - delete
        - revert

