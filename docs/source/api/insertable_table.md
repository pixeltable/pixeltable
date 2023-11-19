# InsertableTable

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    InsertableTable
```

Instances of class {py:class}`pixeltable.InsertableTable` are handles to Pixeltable tables.

Use this handle to query and update the table and to add and drop columns.

{py:class}`pixeltable.InsertableTable` instances are created by calling {py:meth}`pixeltable.Client.create_table`
or {py:meth}`pixeltable.Client.get_table`.

## Column Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    InsertableTable.add_column
    InsertableTable.drop_column
    InsertableTable.rename_column
```

## Data Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    InsertableTable.insert
    InsertableTable.update
    InsertableTable.delete
```

## Versioning

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    InsertableTable.revert
```

