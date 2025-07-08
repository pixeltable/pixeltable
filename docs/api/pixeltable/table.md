# Table

Instances of class `Table` are handles to Pixeltable tables and views/snapshots.

Use this handle to query and update the table and to add and drop columns.

Tables are created by calling [`pxt.create_table`][pixeltable.create_table].
Views are created by calling [`pxt.create_view`][pixeltable.create_view], and snapshots by calling
[`pxt.create_snapshot`][pixeltable.create_snapshot].

To get a handle to an existing table/view/snapshot, call [`pxt.get_table`][pixeltable.get_table].

## Overview

| Column Operations                                         |                                        |
|-----------------------------------------------------------|----------------------------------------|
| [`add_column`][pixeltable.Table.add_column]               | Add a column to the table or view      |
| [`drop_column`][pixeltable.Table.drop_column]             | Remove a column from the table or view |
| [`rename_column`][pixeltable.Table.rename_column]         | Rename a column                        |
| [`recompute_columns`][pixeltable.Table.recompute_columns] | Recompute one or more computed columns |

| Data Operations                       |                              |
|---------------------------------------|------------------------------|
| [`insert`][pixeltable.Table.insert]   | Insert rows into table       |
| [`update`][pixeltable.Table.update]   | Update rows in table or view |
| [`delete`][pixeltable.Table.delete]   | Delete rows from table       |

| Indexing Operations                                             |                                  |
|-----------------------------------------------------------------|----------------------------------|
| [`add_embedding_index`][pixeltable.Table.add_embedding_index]   | Add embedding index on column    |
| [`drop_embedding_index`][pixeltable.Table.drop_embedding_index] | Drop embedding index from column |
| [`drop_index`][pixeltable.Table.drop_index]                     | Drop index from column           |

| Versioning                            |                        |
|---------------------------------------|------------------------|
| [`revert`][pixeltable.Table.revert]   | Revert the last change |

## ::: pixeltable.Table

    options:
      inherited_members: true
