from __future__ import annotations
from typing import Optional, Dict, Any

from .signature import Signature


class FunctionMd:
    """Per-function metadata"""
    def __init__(self, signature: Signature, is_agg: bool, is_library_fn: bool):
        self.signature = signature
        self.is_agg = is_agg
        self.is_library_fn = is_library_fn
        # the following are set externally
        self.fqn: Optional[str] = None  # fully-qualified name
        self.src: str = ''  # source code shown in list()
        self.requires_order_by = False
        self.allows_std_agg = False
        self.allows_window = False

    def as_dict(self) -> Dict[str, Any]:
        # we leave out fqn, which is reconstructed externally
        return {
            'signature': self.signature.as_dict(),
            'is_agg': self.is_agg, 'is_library_fn': self.is_library_fn, 'src': self.src,
            'requires_order_by': self.requires_order_by, 'allows_std_agg': self.allows_std_agg,
            'allows_window': self.allows_window,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FunctionMd:
        result = cls(Signature.from_dict(d['signature']), d['is_agg'], d['is_library_fn'])
        result.requires_order_by = d['requires_order_by']
        result.allows_std_agg = d['allows_std_agg']
        result.allows_window = d['allows_window']
        if 'src' in d:
            result.src = d['src']
        return result
