from typing import Callable, Optional

from mypy import nodes
from mypy.plugin import AnalyzeTypeContext, ClassDefContext, Plugin, MethodSigContext
from mypy.plugins.common import add_method_to_class
from mypy.types import AnyType, NoneType, Type, TypeOfAny, FunctionLike

import pixeltable as pxt


class PxtPlugin(Plugin):
    __UDA_FULLNAME = f'{pxt.uda.__module__}.{pxt.uda.__name__}'
    __ARRAY_GETITEM_FULLNAME = f'{pxt.Array.__module__}.{pxt.Array.__name__}.__class_getitem__'
    __ADD_COMPUTED_COLUMN_FULLNAME = (
        f'{pxt.Table.__module__}.{pxt.Table.__name__}.{pxt.Table.add_computed_column.__name__}'
    )
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

    def get_type_analyze_hook(self, fullname: str) -> Optional[Callable[[AnalyzeTypeContext], Type]]:
        if fullname in self.__FULLNAME_MAP:
            subst_name = self.__FULLNAME_MAP[fullname]
            return lambda ctx: pxt_hook(ctx, subst_name)
        return None

    def get_method_signature_hook(self, fullname: str) -> Optional[Callable[[MethodSigContext], FunctionLike]]:
        if fullname == self.__ADD_COMPUTED_COLUMN_FULLNAME:
            return pxt_method_hook
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

def pxt_method_hook(ctx: MethodSigContext) -> FunctionLike:
    sig = ctx.default_signature
    new_arg_names = sig.arg_names[-1:]
    new_arg_types = [AnyType(TypeOfAny.special_form)]
    new_arg_kinds = sig.arg_kinds[-1:]
    return sig.copy_modified(arg_names=new_arg_names, arg_types=new_arg_types, arg_kinds=new_arg_kinds)

def pxt_decorator_hook(ctx: ClassDefContext) -> bool:
    fn_arg = nodes.Argument(nodes.Var('fn'), AnyType(TypeOfAny.special_form), None, nodes.ARG_POS)
    args_arg = nodes.Argument(nodes.Var('args'), AnyType(TypeOfAny.special_form), None, nodes.ARG_STAR)
    kwargs_arg = nodes.Argument(nodes.Var('kwargs'), AnyType(TypeOfAny.special_form), None, nodes.ARG_STAR2)
    add_method_to_class(
        ctx.api,
        ctx.cls,
        "to_sql",
        args=[fn_arg],
        return_type=AnyType(TypeOfAny.special_form),
        is_staticmethod=True,
    )
    add_method_to_class(
        ctx.api,
        ctx.cls,
        "__init__",
        args=[args_arg, kwargs_arg],
        return_type=NoneType(),
    )
    return True
