# MutableTable

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    MutableTable
```

Instances of class {py:class}`pixeltable.MutableTable` are handles to Pixeltable tables.

Use this handle to query and update the table and to add and drop columns.

{py:class}`pixeltable.MutableTable` instances are created by calling {py:meth}`pixeltable.Db.create_table`
or {py:meth}`pixeltable.Db.get_table`.

## Column Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    MutableTable.add_column
    MutableTable.drop_column
    MutableTable.rename_column
```

## Data Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    MutableTable.insert_rows
    MutableTable.insert_pandas
```

## Versioning

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    MutableTable.revert
```

