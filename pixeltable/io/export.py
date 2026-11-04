import copy

import pixeltable.exceptions as excs
from pixeltable.exprs.column_property_ref import ColumnPropertyRef
from pixeltable.exprs.column_ref import ColumnRef
from pixeltable.exprs.expr import Expr


def replace_media_with_fileurl(select_list_exprs: list[Expr]) -> list[Expr]:
    """Return a new select list where media ColumnRefs are replaced with their .fileurl property.

    This avoids Pixeltable's file caching pipeline and instead returns the authoritative
    URL stored in the database for each media column. Media-typed expressions that are not
    bare ColumnRefs (e.g. `.resize(...)` or other transformations) have no stable URL and
    are rejected.
    """
    result: list[Expr] = []
    for expr in copy.deepcopy(select_list_exprs):
        if isinstance(expr, ColumnRef) and expr.col_type.is_media_type():
            result.append(ColumnPropertyRef(expr, ColumnPropertyRef.Property.FILEURL))
        elif expr.col_type.is_media_type():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Cannot export media expression {expr!r}: only stored media columns can be serialized. '
                f'Materialize it as a computed column, or select its underlying '
                f'column via `.fileurl` instead.',
            )
        else:
            result.append(expr)
    return result
