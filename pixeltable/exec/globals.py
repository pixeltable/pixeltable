from __future__ import annotations

import dataclasses

from pixeltable import exceptions as excs, exprs
from pixeltable.exprs import ArrayMd, BinaryMd
from pixeltable.utils.misc import non_none_dict_factory

INLINED_OBJECT_MD_KEY = '__pxtinlinedobjmd__'


@dataclasses.dataclass
class InlinedObjectMd:
    type: str  # corresponds to ts.ColumnType.Type
    url_idx: int
    img_start: int | None = None
    img_end: int | None = None
    array_md: ArrayMd | None = None
    binary_md: BinaryMd | None = None

    @classmethod
    def from_dict(cls, d: dict) -> InlinedObjectMd:
        d = d.copy()
        if 'array_md' in d:
            d['array_md'] = ArrayMd(**d['array_md'])
        if 'binary_md' in d:
            d['binary_md'] = BinaryMd(**d['binary_md'])
        return cls(**d)

    def as_dict(self) -> dict:
        result = dataclasses.asdict(self, dict_factory=non_none_dict_factory)
        if self.array_md is not None:
            result['array_md'] = self.array_md.as_dict()
        if self.binary_md is not None:
            result['binary_md'] = dataclasses.asdict(self.binary_md)
        return result

def resolve_int(e: exprs.Expr, args: dict[str, Any], role: str) -> int:
    """Resolve a limit/offset Expr to an int at iteration time.

    Accepts Literal (constant) or Variable (resolved from the bound args dict). Other Expr
    shapes are rejected because Python-side limit/offset (FilterNode, AggregationNode) needs
    a concrete int. SQL-side limit/offset on SqlNode supports the full Expr surface via
    bindparams.
    """
    if isinstance(e, exprs.Literal):
        return int(e.val)
    if isinstance(e, exprs.Variable):
        return int(args[e.name])
    raise excs.RequestError(
        excs.ErrorCode.UNSUPPORTED_OPERATION, f'{role}: unsupported expression for non-SQL limit/offset: {e}'
    )
