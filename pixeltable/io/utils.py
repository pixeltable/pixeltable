from keyword import iskeyword as is_python_keyword
from typing import Any, Optional, Union

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog.globals import is_system_column_name


def normalize_pxt_col_name(name: str) -> str:
    """
    Normalizes an arbitrary DataFrame column name into a valid Pixeltable identifier by:
    - replacing any non-ascii or non-alphanumeric characters with an underscore _
    - prefixing the result with the letter 'c' if it starts with an underscore or a number
    """
    id = ''.join(ch if ch.isascii() and ch.isalnum() else '_' for ch in name)
    if id[0].isnumeric():
        id = f'c_{id}'
    elif id[0] == '_':
        id = f'c{id}'
    assert pxt.catalog.is_valid_identifier(id), id
    return id


def normalize_primary_key_parameter(primary_key: Optional[Union[str, list[str]]] = None) -> list[str]:
    if primary_key is None:
        primary_key = []
    elif isinstance(primary_key, str):
        primary_key = [primary_key]
    elif not isinstance(primary_key, list) or not all(isinstance(pk, str) for pk in primary_key):
        raise excs.Error('primary_key must be a single column name or a list of column names')
    return primary_key


def _is_usable_as_column_name(name: str, destination_schema: dict[str, Any]) -> bool:
    return not (is_system_column_name(name) or is_python_keyword(name) or name in destination_schema)


def normalize_schema_names(
    in_schema: dict[str, Any],
    primary_key: list[str],
    schema_overrides: dict[str, Any],
    require_valid_pxt_column_names: bool = False,
) -> tuple[dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Convert all names in the input schema from source names to valid Pixeltable identifiers
    - Ensure that all names are unique.
    - Report an error if any types are missing
    - If "require_valid_pxt_column_names", report an error if any column names are not valid Pixeltable column names
    - Report an error if any primary key columns are missing
    Returns
    - A new schema with normalized column names
    - The primary key columns, mapped to the normalized names
    - A mapping from the original names to the normalized names.
    """

    # Report any untyped columns as an error
    untyped_cols = [in_name for in_name, column_type in in_schema.items() if column_type is None]
    if len(untyped_cols) > 0:
        raise excs.Error(f'Could not infer pixeltable type for column(s): {", ".join(untyped_cols)}')

    # Report any columns in `schema_overrides` that are not in the source
    extraneous_overrides = schema_overrides.keys() - in_schema.keys()
    if len(extraneous_overrides) > 0:
        raise excs.Error(
            f'Some column(s) specified in `schema_overrides` are not present '
            f'in the source: {", ".join(extraneous_overrides)}'
        )

    schema: dict[str, Any] = {}
    col_mapping: dict[str, str] = {}  # Maps column names to Pixeltable column names if needed
    for in_name, pxt_type in in_schema.items():
        pxt_name = normalize_pxt_col_name(in_name)
        # Ensure that column names are unique by appending a distinguishing suffix
        # to any collisions
        pxt_fname = pxt_name
        n = 1
        while not _is_usable_as_column_name(pxt_fname, schema):
            pxt_fname = f'{pxt_name}_{n}'
            n += 1
        schema[pxt_fname] = pxt_type
        col_mapping[in_name] = pxt_fname

    # Determine if the col_mapping is the identity mapping
    non_identity_keys = [k for k, v in col_mapping.items() if k != v]
    if len(non_identity_keys) > 0:
        if require_valid_pxt_column_names:
            raise excs.Error(
                f'Column names must be valid pixeltable identifiers. Invalid names: {", ".join(non_identity_keys)}'
            )
    else:
        col_mapping = None

    # Report any primary key columns that are not in the source as an error
    missing_pk = [pk for pk in primary_key if pk not in in_schema]
    if len(missing_pk) > 0:
        raise excs.Error(f'Primary key column(s) are not found in the source: {", ".join(missing_pk)}')

    pxt_pk = [col_mapping[pk] for pk in primary_key] if col_mapping is not None else primary_key

    return schema, pxt_pk, col_mapping
