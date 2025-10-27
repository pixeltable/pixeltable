from typing import Callable, ClassVar

from mypy import nodes
from mypy.plugin import AnalyzeTypeContext, ClassDefContext, FunctionContext, MethodSigContext, Plugin
from mypy.plugins.common import add_attribute_to_class, add_method_to_class
from mypy.types import AnyType, FunctionLike, Instance, NoneType, Type, TypeOfAny

import pixeltable as pxt
from pixeltable import exprs


class PxtPlugin(Plugin):
    __UDA_FULLNAME = f'{pxt.uda.__module__}.{pxt.uda.__name__}'
    __ARRAY_GETITEM_FULLNAME = f'{pxt.Array.__module__}.{pxt.Array.__name__}.__class_getitem__'
    __ADD_COLUMN_FULLNAME = f'{pxt.Table.__module__}.{pxt.Table.__name__}.{pxt.Table.add_column.__name__}'
    __ADD_COMPUTED_COLUMN_FULLNAME = (
        f'{pxt.Table.__module__}.{pxt.Table.__name__}.{pxt.Table.add_computed_column.__name__}'
    )
    __TYPE_MAP: ClassVar[dict] = {
        pxt.Json: 'typing.Any',
        pxt.Array: 'numpy.ndarray',
        pxt.Image: 'PIL.Image.Image',
        pxt.Video: 'builtins.str',
        pxt.Audio: 'builtins.str',
        pxt.Document: 'builtins.str',
    }
    __FULLNAME_MAP: ClassVar[dict] = {f'{k.__module__}.{k.__name__}': v for k, v in __TYPE_MAP.items()}

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        return adjust_uda_type

    def get_type_analyze_hook(self, fullname: str) -> Callable[[AnalyzeTypeContext], Type] | None:
        if fullname in self.__FULLNAME_MAP:
            subst_name = self.__FULLNAME_MAP[fullname]
            return lambda ctx: adjust_pxt_type(ctx, subst_name)
        return None

    def get_method_signature_hook(self, fullname: str) -> Callable[[MethodSigContext], FunctionLike] | None:
        if fullname in (self.__ADD_COLUMN_FULLNAME, self.__ADD_COMPUTED_COLUMN_FULLNAME):
            return adjust_kwargs
        return None

    def get_class_decorator_hook_2(self, fullname: str) -> Callable[[ClassDefContext], bool] | None:
        if fullname == self.__UDA_FULLNAME:
            return adjust_uda_methods
        return None


def plugin(version: str) -> type:
    return PxtPlugin


_AGGREGATOR_FULLNAME = f'{pxt.Aggregator.__module__}.{pxt.Aggregator.__name__}'
_FN_CALL_FULLNAME = f'{exprs.Expr.__module__}.{exprs.Expr.__name__}'


def adjust_uda_type(ctx: FunctionContext) -> Type:
    """
    Mypy doesn't understand that a class with a @uda decorator isn't actually a class, so it assumes
    that sum(expr), for example, actually returns an instance of sum. We correct this by changing the
    return type of any subclass of `Aggregator` to `FunctionCall`.
    """
    ret_type = ctx.default_return_type
    if isinstance(ret_type, Instance) and (
        ret_type.type.fullname == _AGGREGATOR_FULLNAME
        or any(base.type.fullname == _AGGREGATOR_FULLNAME for base in ret_type.type.bases)
    ):
        ret_type = AnyType(TypeOfAny.special_form)
    return ret_type


def adjust_pxt_type(ctx: AnalyzeTypeContext, subst_name: str) -> Type:
    """
    Replaces the special Pixeltable classes (such as pxt.Array) with their standard equivalents (such as np.ndarray).
    """
    if subst_name == 'typing.Any':
        return AnyType(TypeOfAny.special_form)
    return ctx.api.named_type(subst_name, [])


def adjust_kwargs(ctx: MethodSigContext) -> FunctionLike:
    """
    Mypy has a "feature" where it will spit out multiple warnings if a method with signature
    ```
    def my_func(*, arg1: int, arg2: str, **kwargs: Expr)
    ```
    (for example) is called with bare kwargs:
    ```
    my_func(my_kwarg=value)
    ```
    This is a disaster for type-checking of add_column and add_computed_column. Here we adjust the signature so
    that mypy thinks it is simply
    ```
    def my_func(**kwargs: Any)
    ```
    thereby avoiding any type-checking errors. For details, see: <https://github.com/python/mypy/issues/18481>
    """
    sig = ctx.default_signature
    new_arg_names = sig.arg_names[-1:]
    new_arg_types = [AnyType(TypeOfAny.special_form)]
    new_arg_kinds = sig.arg_kinds[-1:]
    return sig.copy_modified(arg_names=new_arg_names, arg_types=new_arg_types, arg_kinds=new_arg_kinds)


def adjust_uda_methods(ctx: ClassDefContext) -> bool:
    """
    Mypy does not handle the `@pxt.uda` aggregator well; it continues to treat the decorated class as a class,
    even though it has been replaced by an `AggregateFunction`. Here we add static methods to the class that
    imitate various (instance) methods of `AggregateFunction` so that they can be properly type-checked.
    """
    list_type = ctx.api.named_type('builtins.list', [AnyType(TypeOfAny.special_form)])
    fn_arg = nodes.Argument(nodes.Var('fn'), AnyType(TypeOfAny.special_form), None, nodes.ARG_POS)
    args_arg = nodes.Argument(nodes.Var('args'), AnyType(TypeOfAny.special_form), None, nodes.ARG_STAR)
    kwargs_arg = nodes.Argument(nodes.Var('kwargs'), AnyType(TypeOfAny.special_form), None, nodes.ARG_STAR2)
    add_method_to_class(ctx.api, ctx.cls, '__init__', args=[args_arg, kwargs_arg], return_type=NoneType())
    add_method_to_class(
        ctx.api, ctx.cls, 'to_sql', args=[fn_arg], return_type=AnyType(TypeOfAny.special_form), is_staticmethod=True
    )
    add_method_to_class(
        ctx.api, ctx.cls, 'overload', args=[fn_arg], return_type=AnyType(TypeOfAny.special_form), is_staticmethod=True
    )
    add_attribute_to_class(ctx.api, ctx.cls, 'signatures', typ=list_type, is_classvar=True)
    return True
