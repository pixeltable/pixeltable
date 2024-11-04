# Pixeltable

Import conventions:

```python
import pixeltable as pxt
```

Insertable tables, views, and snapshots all have a tabular interface and are generically referred to as "tables"
below.

## Overview

| Table Operations                                    |                                 |
|-----------------------------------------------------|---------------------------------|
| [`pxt.create_table`][pixeltable.create_table]       | Create a new (insertable) table |
| [`pxt.create_view`][pixeltable.create_view]         | Create a new view               |
| [`pxt.create_snapshot`][pixeltable.create_snapshot] | Create a new snapshot           |
| [`pxt.drop_table`][pixeltable.drop_table]           | Delete a table                  |
| [`pxt.get_table`][pixeltable.get_table]             | Get a handle to a table         |
| [`pxt.list_tables`][pixeltable.list_tables]         | List the tables in a directory  |

| Directory Operations                      |                                     |
|-------------------------------------------|-------------------------------------|
| [`pxt.create_dir`][pixeltable.create_dir] | Create a directory                  |
| [`pxt.list_dirs`][pixeltable.list_dirs]   | List the directories in a directory |
| [`pxt.drop_dir`][pixeltable.drop_dir]     | Remove a directory                  |

| Misc                                                    |                                                                       |
|---------------------------------------------------------|-----------------------------------------------------------------------|
| [`pxt.configure_logging`][pixeltable.configure_logging] | Configure logging                                                     |
| [`pxt.init`][pixeltable.init]                           | Initialize Pixeltable runtime now (if not already initialized)        |
| [`pxt.move`][pixeltable.move]                           | Move a schema object to a new directory and/or rename a schema object |

## ::: pixeltable

    options:
      members:
      - __init__
      - configure_logging
      - create_dir
      - create_snapshot
      - create_table
      - create_view
      - drop_table
      - get_table
      - init
      - list_tables
      - list_dirs
      - move
      - drop_dir
