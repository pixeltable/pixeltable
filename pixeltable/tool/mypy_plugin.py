from typing import Callable, Optional

from mypy import nodes
from mypy.plugin import AnalyzeTypeContext, ClassDefContext, Plugin
from mypy.plugins.common import add_method_to_class
from mypy.types import AnyType, Type, TypeOfAny

import pixeltable as pxt


class PxtPlugin(Plugin):
    __UDA_FULLNAME = f'{pxt.uda.__module__}.{pxt.uda.__name__}'
    __TYPE_MAP = {
        pxt.Json: 'typing.Any',
        pxt.Array: 'numpy.ndarray',
        pxt.Image: 'PIL.Image.Image',
        pxt.Video: 'builtins.str',
        pxt.Audio: 'builtins.str',
        pxt.Document: 'builtins.str',
    }
    __FULLNAME_MAP = {f'{k.__module__}.{k.__name__}': v for k, v in __TYPE_MAP.items()}

    def get_type_analyze_hook(self, fullname: str) -> Optional[Callable[[AnalyzeTypeContext], Type]]:
        if fullname in self.__FULLNAME_MAP:
            subst_name = self.__FULLNAME_MAP[fullname]
            return lambda ctx: pxt_hook(ctx, subst_name)
        return None

    def get_class_decorator_hook_2(self, fullname: str) -> Optional[Callable[[ClassDefContext], bool]]:
        if fullname == self.__UDA_FULLNAME:
            return pxt_decorator_hook
        return None


def plugin(version: str) -> type:
    return PxtPlugin


def pxt_hook(ctx: AnalyzeTypeContext, subst_name: str) -> Type:
    if subst_name == 'typing.Any':
        return AnyType(TypeOfAny.special_form)
    return ctx.api.named_type(subst_name, [])


def pxt_decorator_hook(ctx: ClassDefContext) -> bool:
    arg = nodes.Argument(nodes.Var('fn'), AnyType(TypeOfAny.special_form), None, nodes.ARG_POS)
    add_method_to_class(
        ctx.api,
        ctx.cls,
        'to_sql',
        args=[arg],
        return_type=AnyType(TypeOfAny.special_form),
        is_staticmethod=True,
    )
    return True
