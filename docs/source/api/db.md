# Db

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    Db
```

Instances of class {py:class}`pixeltable.Db` are handles to Pixeltable databases.

Use this handle to create and manage tables, snapshots, functions, and directories in the database.

{py:class}`pixeltable.Db` instances are created by calling {py:meth}`pixeltable.Client.create_db`
or {py:meth}`pixeltable.Client.get_db`.

## Table Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    Db.create_table
    Db.drop_table
    Db.get_table
    Db.list_tables
```

## Snapshot Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    Db.create_snapshot
```

## Function Operations

```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    Db.create_function
    Db.rename_function
    Db.update_function
    Db.get_function
    Db.drop_function
```

## Directory Operations
```{eval-rst}
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    Db.create_dir
    Db.rm_dir
    Db.list_dirs
```

