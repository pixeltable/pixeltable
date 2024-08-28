from typing import Callable, Optional

from mypy.plugin import AnalyzeTypeContext, DynamicClassDefContext, FunctionSigContext, Plugin
from mypy.types import FunctionLike, Type


class PxtPlugin(Plugin):
    def get_type_analyze_hook(self, fullname: str) -> Optional[Callable[[AnalyzeTypeContext], type]]:
        if fullname in ('pixeltable.type_system.ImageT', 'pixeltable.type_system.ArrayT'):
            return pxt_hook
        return None

def plugin(version: str):
    return PxtPlugin

def pxt_hook(ctx: AnalyzeTypeContext) -> Type:
    print(ctx.type.name)
    return ctx.api.named_type('PIL.Image.Image')
