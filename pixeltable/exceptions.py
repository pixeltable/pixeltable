from typing import List, Any
from dataclasses import dataclass


class Error(Exception):
    pass


@dataclass
class ExprEvalError(Exception):
    expr: Any  # exprs.Expr, but we're not importing pixeltable.exprs to avoid circular imports
    exc: Exception
    dependency_vals: List[Any]
    row_num: int


class DuplicateNameError(Exception):
    pass


class UnknownEntityError(Exception):
    pass


class BadFormatError(Exception):
    pass


class DirectoryNotEmptyError(Exception):
    pass


class InsertError(Exception):
    pass


class RuntimeError(Exception):
    pass
