from typing import List
import dataclasses
import logging


_logger = logging.getLogger('pixeltable')

# name of the position column in a component view
POS_COLUMN_NAME = 'pos'

@dataclasses.dataclass
class UpdateStatus:
    num_rows: int = 0
    # TODO: disambiguate what this means: # of slots computed or # of columns computed?
    num_computed_values: int = 0
    num_excs: int = 0
    updated_cols: List[str] = dataclasses.field(default_factory=list)
    cols_with_excs: List[str] = dataclasses.field(default_factory=list)

def is_valid_identifier(name: str) -> bool:
    return name.isidentifier() and not name.startswith('_')

def is_valid_path(path: str, empty_is_valid : bool) -> bool:
    if path == '':
        return empty_is_valid

    for part in path.split('.'):
        if not is_valid_identifier(part):
            return False
    return True

def is_system_column_name(name: str) -> bool:
    return name == POS_COLUMN_NAME
