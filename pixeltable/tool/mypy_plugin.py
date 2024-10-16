from typing import Callable, Optional

from mypy.plugin import AnalyzeTypeContext, Plugin
from mypy.types import Type

import pixeltable as pxt


class PxtPlugin(Plugin):
    __TYPE_MAP = {
        pxt.Json: 'typing.Any',
        pxt.Array: 'numpy.ndarray',
        pxt.Image: 'PIL.Image.Image',
        pxt.Video: 'builtins.str',
        pxt.Audio: 'builtins.str',
        pxt.Document: 'builtins.str',
    }
    __FULLNAME_MAP = {
        f'{k.__module__}.{k.__name__}': v
        for k, v in __TYPE_MAP.items()
    }

    def get_type_analyze_hook(self, fullname: str) -> Optional[Callable[[AnalyzeTypeContext], type]]:
        if fullname in self.__FULLNAME_MAP:
            subst_name = self.__FULLNAME_MAP[fullname]
            return lambda ctx: pxt_hook(ctx, subst_name)

def plugin(version: str):
    return PxtPlugin

def pxt_hook(ctx: AnalyzeTypeContext, subst_name: str) -> Type:
    return ctx.api.named_type(subst_name)
