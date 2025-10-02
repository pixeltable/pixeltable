from __future__ import annotations

import dataclasses

from pixeltable.exprs import ArrayMd
from pixeltable.utils.misc import non_none_dict_factory

INLINED_OBJECT_MD_KEY = '__pxtinlinedobjmd__'


@dataclasses.dataclass
class InlinedObjectMd:
    type: str  # corresponds to ts.ColumnType.Type
    url_idx: int
    img_start: int | None = None
    img_end: int | None = None
    array_md: ArrayMd | None = None

    @classmethod
    def from_dict(cls, d: dict) -> InlinedObjectMd:
        if 'array_md' in d:
            array_md = ArrayMd(**d['array_md'])
            del d['array_md']
            return cls(**d, array_md=array_md)
        else:
            return cls(**d)

    def as_dict(self) -> dict:
        result = dataclasses.asdict(self, dict_factory=non_none_dict_factory)
        if self.array_md is not None:
            result['array_md'] = self.array_md.as_dict()
        return result
