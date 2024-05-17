# Pixeltable

Import conventions:
```python
import pixeltable as pxt
```

Insertable tables, views, and snapshots all have a tabular interface and are generically referred to as "tables"
below.

## Overview
| Table Operations                              |                                 |
|-----------------------------------------------|---------------------------------|
| [`pxt.create_table`][pixeltable.create_table] | Create a new (insertable) table |
| [`pxt.create_view`][pixeltable.create_view]   | Create a new view               |
| [`pxt.drop_table`][pixeltable.drop_table]     | Delete a table                  |
| [`pxt.get_table`][pixeltable.get_table]       | Get a handle to a table         |
| [`pxt.list_tables`][pixeltable.list_tables]   | List the tables in a directory  |

| Directory Operations                      |                                     |
|-------------------------------------------|-------------------------------------|
| [`pxt.create_dir`][pixeltable.create_dir] | Create a directory                  |
| [`pxt.list_dirs`][pixeltable.list_dirs]   | List the directories in a directory |
| [`pxt.rm_dir`][pixeltable.rm_dir]         | Remove a directory                  |

| Misc                                                    |                                                                       |
|---------------------------------------------------------|-----------------------------------------------------------------------|
| [`pxt.configure_logging`][pixeltable.configure_logging] | Configure logging                                                     |
| [`pxt.init`][pixeltable.init]                           | Initialize Pixeltable runtime now (if not already initialized)        |
| [`pxt.move`][pixeltable.move]                           | Move a schema object to a new directory and/or rename a schema object |

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
