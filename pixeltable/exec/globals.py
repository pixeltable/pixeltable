from dataclasses import dataclass

import pixeltable.type_system as ts
from pixeltable.exprs import ArrayMd

INLINED_OBJECT_MD_KEY = '__pxt_inlined_obj_md__'


@dataclass
class InlinedObjectMd:
    type: ts.ColumnType.Type
    url_idx: int
    start: int | None = None
    end: int | None = None
    array_md: ArrayMd | None = None
