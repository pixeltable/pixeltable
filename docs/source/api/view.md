# View

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    View
```

Instances of class {py:class}`pixeltable.View` are handles to Pixeltable views.

Use this handle to query and update the view and to add and drop columns.

{py:class}`pixeltable.View` instances are created by calling {py:meth}`pixeltable.Db.create_view`
or {py:meth}`pixeltable.Db.get_table`.

## Column Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    View.add_column
    View.drop_column
    View.rename_column
```

## Data Operations

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    View.update
```

## Versioning

```{eval-rst}
.. currentmodule:: pixeltable
.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    View.revert
```

