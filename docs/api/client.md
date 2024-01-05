# Client

Use instances of pixeltable.Client to create and manage tables, snapshots, functions, and directories in the database.

Insertable tables, views and snapshots of these all have a tabular interface and are generically referred to as "tables"
below.

## Overview
| Table Operations | |
|------------|-----------------------------------------------------|
| [`create_table`][pixeltable.Client.create_table] | Create a new (insertable) table|
| [`create_view`][pixeltable.Client.create_view] | Create a new view |
| [`create_snapshot`][pixeltable.Client.create_snapshot] | Create a new snapshot of an insertable table or view |
| [`drop_table`][pixeltable.Client.drop_table] | Delete a table |
| [`get_table`][pixeltable.Client.get_table] | Get a handle to a table |
| [`list_tables`][pixeltable.Client.list_tables] | List the tables in a directory |

| Directory Operations | |
|------------|-----------------------------------------------------|
| [`create_dir`][pixeltable.Client.create_dir] | Create a directory |
| [`rm_dir`][pixeltable.Client.rm_dir] | Remove a directory |
| [`list_dirs`][pixeltable.Client.list_dirs] | List the directories in a directory |

| Misc | |
|------------|-----------------------------------------------------|
| [`move`][pixeltable.Client.move] | Move a schema object to a new directory and/or rename a schema object |
| [`logging`][pixeltable.Client.logging] | Configure logging |

## ::: pixeltable.Client
    options:
      members:
      - __init__
      - create_table
      - create_snapshot
      - create_view
      - get_table
      - move
      - list_tables
      - drop_table
      - create_dir
      - rm_dir
      - list_dirs
      - create_function
      - update_function
      - get_function
      - drop_function
      - logging

