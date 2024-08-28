from typing import Callable, Optional

from mypy.plugin import AnalyzeTypeContext, Plugin
from mypy.types import Type


class PxtPlugin(Plugin):
    def get_type_analyze_hook(self, fullname: str) -> Optional[Callable[[AnalyzeTypeContext], type]]:
        if fullname == 'pixeltable.type_system.ImageT':
            return lambda ctx: pxt_hook(ctx, 'PIL.Image.Image')
        if fullname == 'pixeltable.type_system.ArrayT':
            return lambda ctx: pxt_hook(ctx, 'numpy.ndarray')
        return None

def plugin(version: str):
    return PxtPlugin

def pxt_hook(ctx: AnalyzeTypeContext, subst_name: str) -> Type:
    return ctx.api.named_type(subst_name)
