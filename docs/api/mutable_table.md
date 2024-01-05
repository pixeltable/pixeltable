# MutableTable

`MutableTable` is the base class for Pixeltable (insertable) tables and views.

Use [`Client.create_table`][pixeltable.Client.create_table]/[`Client.create_view`][pixeltable.Client.create_view]
or [`Client.get_table`][pixeltable.Client.get_table] to create instances of this class.

## ::: pixeltable.MutableTable
    options:
      members:
      - add_column
      - drop_column
      - rename_column
      - update
      - revert
