# Table

Instances of class `Table` are handles to Pixeltable tables and views/snapshots.

Use this handle to query and update the table and to add and drop columns.

Tables are created by calling [`pxt.create_table`][pixeltable.create_table].
Views and snapshots are created by calling [`pxt.create_view`][pixeltable.create_view]
(snapshots require `is_snapshot=True`).

To get a handle to an existing table/view/snapshot, call [`pxt.get_table`][pixeltable.get_table].


## Overview
| Column Operations                                           |                                        |
|-------------------------------------------------------------|----------------------------------------|
| [`add_column`][pixeltable.Table.add_column]       | Add a column to the table or view      |
| [`drop_column`][pixeltable.Table.drop_column]     | Remove a column from the table or view |
| [`rename_column`][pixeltable.Table.rename_column] | Rename a column                        |

| Data Operations                               |                             |
|-----------------------------------------------|-----------------------------|
| [`insert`][pixeltable.Table.insert] | Insert rows into table      |
| [`update`][pixeltable.Table.update] | Upate rows in table or view |
| [`delete`][pixeltable.Table.delete] | Delete rows from table      |

| Versioning                                    |                        |
|-----------------------------------------------|------------------------|
| [`revert`][pixeltable.Table.revert] | Revert the last change |

## ::: pixeltable.Table
    options:
      inherited_members: true
      members:
      - __getattr__
      - __getitem__
      - collect
      - count
      - describe
      - df
      - head
      - link
      - unlink
      - sync
      - get_remotes
      - select
      - show
      - where
      - tail
      - add_column
      - delete
      - drop_column
      - insert
      - rename_column
      - revert
      - update