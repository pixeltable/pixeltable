# Pixeltable

Insertable tables, views, and snapshots all have a tabular interface and are generically referred to as "tables"
below.

## Overview
| Table Operations                   |                                 |
|------------------------------------|---------------------------------|
| [`create_table`][pxt.create_table] | Create a new (insertable) table |
| [`create_view`][pxt.create_view]   | Create a new view               |
| [`drop_table`][pxt.drop_table]     | Delete a table                  |
| [`get_table`][pxt.get_table]       | Get a handle to a table         |
| [`list_tables`][pxt.list_tables]   | List the tables in a directory  |

| Directory Operations           |                                     |
|--------------------------------|-------------------------------------|
| [`create_dir`][pxt.create_dir] | Create a directory                  |
| [`list_dirs`][pxt.list_dirs]   | List the directories in a directory |
| [`rm_dir`][pxt.rm_dir]         | Remove a directory                  |

| Misc                                         |                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------|
| [`configure_logging`][pxt.configure_logging] | Configure logging                                                     |
| [`init`][pxt.init]                           | Initialize Pixeltable runtime now (if not already initialized)        |
| [`move`][pxt.move]                           | Move a schema object to a new directory and/or rename a schema object |

## ::: pixeltable
    options:
      members:
      - __init__
      - create_table
      - create_view
      - drop_table
      - get_table
      - list_tables
      - create_dir
      - list_dirs
      - rm_dir
      - configure_logging
      - init
      - move
