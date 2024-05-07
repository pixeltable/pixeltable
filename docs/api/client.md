# Client

Insertable tables, views and snapshots of these all have a tabular interface and are generically referred to as "tables"
below.

## Overview
| Table Operations                               | |
|------------------------------------------------|-----------------------------------------------------|
| [`create_table`][pxt.create_table]             | Create a new (insertable) table|
| [`create_view`][pxt.create_view] | Create a new view |
| [`drop_table`][pxt.drop_table]   | Delete a table |
| [`get_table`][pxt.get_table]     | Get a handle to a table |
| [`list_tables`][pxt.list_tables] | List the tables in a directory |

| Directory Operations | |
|------------|-----------------------------------------------------|
| [`create_dir`][pxt.create_dir] | Create a directory |
| [`rm_dir`][pxt.rm_dir] | Remove a directory |
| [`list_dirs`][pxt.list_dirs] | List the directories in a directory |

| Misc                                         | |
|----------------------------------------------|-----------------------------------------------------|
| [`move`][pxt.move]                           | Move a schema object to a new directory and/or rename a schema object |
| [`configure_logging`][pxt.configure_logging] | Configure logging |

## ::: pixeltable
    options:
      members:
      - __init__
      - create_table
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

