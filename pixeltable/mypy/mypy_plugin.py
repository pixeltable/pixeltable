from typing import Callable, ClassVar

from mypy import nodes
from mypy.nodes import GDEF, SymbolTableNode
from mypy.plugin import (
    AnalyzeTypeContext,
    ClassDefContext,
    DynamicClassDefContext,
    FunctionContext,
    MethodSigContext,
    Plugin,
)
from mypy.plugins.common import add_attribute_to_class, add_method_to_class
from mypy.types import AnyType, FunctionLike, Instance, NoneType, Type, TypeOfAny

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.catalog.model import TableModelMeta
from pixeltable.type_system import _PxtType


class PxtPlugin(Plugin):
    __UDA_FULLNAME = f'{pxt.uda.__module__}.{pxt.uda.__name__}'
    __ITERATOR_FULLNAME = f'{pxt.iterator.__module__}.{pxt.iterator.__name__}'
    __ARRAY_GETITEM_FULLNAME = f'{pxt.Array.__module__}.{pxt.Array.__name__}.__class_getitem__'
    __ADD_COLUMN_FULLNAME = f'{pxt.Table.__module__}.{pxt.Table.__name__}.{pxt.Table.add_column.__name__}'
    __ADD_COMPUTED_COLUMN_FULLNAME = (
        f'{pxt.Table.__module__}.{pxt.Table.__name__}.{pxt.Table.add_computed_column.__name__}'
    )
    __MODEL_BASE_FULLNAME = f'{pxt.model_base.__module__}.{pxt.model_base.__name__}'
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
        if fullname == _REQUIRED_FULLNAME:
            return adjust_required_type
        return None

    def get_base_class_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        if fullname == _PXT_TYPE_FULLNAME:
            return make_pxt_type_subscriptable
        return None

    def get_method_signature_hook(self, fullname: str) -> Callable[[MethodSigContext], FunctionLike] | None:
        if fullname in (self.__ADD_COLUMN_FULLNAME, self.__ADD_COMPUTED_COLUMN_FULLNAME):
            return adjust_kwargs
        return None

    def get_dynamic_class_hook(self, fullname: str) -> Callable[[DynamicClassDefContext], None] | None:
        if fullname == self.__MODEL_BASE_FULLNAME:
            return create_model_base_class
        return None

    def get_class_decorator_hook_2(self, fullname: str) -> Callable[[ClassDefContext], bool] | None:
        if fullname == self.__UDA_FULLNAME:
            return adjust_uda_methods
        if fullname == self.__ITERATOR_FULLNAME:
            return adjust_iterator_methods
        return None


def plugin(version: str) -> type:
    return PxtPlugin


_AGGREGATOR_FULLNAME = f'{pxt.Aggregator.__module__}.{pxt.Aggregator.__name__}'
_PXTITERATOR_FULLNAME = f'{pxt.PxtIterator.__module__}.{pxt.PxtIterator.__name__}'
_FN_CALL_FULLNAME = f'{exprs.Expr.__module__}.{exprs.Expr.__name__}'
_PXT_TYPE_FULLNAME = f'{_PxtType.__module__}.{_PxtType.__name__}'
_REQUIRED_FULLNAME = f'{pxt.Required.__module__}.{pxt.Required.__name__}'
_TABLE_MODEL_META_FULLNAME = f'{TableModelMeta.__module__}.{TableModelMeta.__name__}'


def make_pxt_type_subscriptable(ctx: ClassDefContext) -> None:
    """
    mypy treats `SomeClass[...]` as a generic type application and rejects it for a non-generic class, ignoring
    `__class_getitem__`. The Pixeltable type-hint family (`_PxtType` subclasses such as `Image`, `Array`, `Json`)
    is parameterizable at runtime via `__class_getitem__` (e.g. `Image[(300, 300), 'RGB']`). To let those subscripts
    type-check (as `Any`) in value positions such as schema dicts, we synthesize a metaclass whose
    `__getitem__(item) -> Any` mypy resolves the subscript through (only a *metaclass* `__getitem__` is honored for
    `Class[...]`; `__class_getitem__` and instance/classmethod `__getitem__` are not).

    `Required` additionally must appear non-generic: while generic, mypy parses its argument as a type, so
    `Required[Image[(300, 300)]]` fails ("Type expected within [...]") on the tuple before any hook runs. We clear
    its type variables here; `adjust_required_type` keeps `Required[...]` working in annotation positions.
    """
    info = ctx.cls.info
    meta = ctx.api.basic_new_typeinfo('_PxtSubscriptMeta', ctx.api.named_type('builtins.type'), ctx.cls.line)
    any_type = AnyType(TypeOfAny.special_form)
    add_method_to_class(
        ctx.api,
        meta.defn,
        '__getitem__',
        args=[nodes.Argument(nodes.Var('item'), any_type, None, nodes.ARG_POS)],
        return_type=any_type,
    )
    # Register the synthesized metaclass under the class. Its fullname is qualified under this class
    # (e.g. `...Image._PxtSubscriptMeta`), so it must be reachable from the class's symbol table or mypy's
    # incremental-cache fixup fails to resolve it on deserialization.
    meta_sym = nodes.SymbolTableNode(nodes.MDEF, meta)
    meta_sym.plugin_generated = True
    info.names[meta.name] = meta_sym

    meta_inst = Instance(meta, [])
    info.metaclass_type = meta_inst
    info.declared_metaclass = meta_inst
    if info.fullname == _REQUIRED_FULLNAME:
        info.type_vars = []
        info.defn.type_vars = []
        info.add_type_vars()


def adjust_required_type(ctx: AnalyzeTypeContext) -> Type:
    """
    In an annotation position, resolve `Required[T]` to `T` (the marker only signifies non-nullability, which has no
    bearing on mypy's view); a bare `Required` resolves to `Any`. This complements `make_pxt_type_subscriptable`,
    which makes `Required` non-generic (so mypy no longer interprets `Required[...]` natively).
    """
    if ctx.type.args:
        return ctx.api.analyze_type(ctx.type.args[0])
    return AnyType(TypeOfAny.special_form)


def create_model_base_class(ctx: DynamicClassDefContext) -> None:
    """
    Mypy cannot use the result of a function call as a base class, so `TableModel = pxt.model_base()` followed by
    `class MyModel(TableModel, ...)` produces "Variable is not valid as a type" / "Invalid base class" errors. Here we
    intercept the `model_base()` call and synthesize a real class for the assignment target, carrying
    `TableModelMeta` as its metaclass. That makes the name usable as a base class and lets mypy type-check the
    metaclass keyword arguments (`name=`, `base=`, ...) and the forwarded table methods on subclasses.
    """
    api = ctx.api
    metaclass = api.named_type_or_none(_TABLE_MODEL_META_FULLNAME)
    if metaclass is None:
        # `TableModelMeta` isn't ready yet; try again on a later pass.
        api.defer()
        return
    info = api.basic_new_typeinfo(ctx.name, api.named_type('builtins.object'), ctx.call.line)
    info.declared_metaclass = metaclass
    info.metaclass_type = metaclass
    api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, info))


def adjust_uda_type(ctx: FunctionContext) -> Type:
    """
    Mypy doesn't understand that a class with a @uda decorator isn't actually a class, so it assumes
    that sum(expr), for example, actually returns an instance of sum. We correct this by changing the
    return type of any subclass of `Aggregator` to `FunctionCall`.
    """
    ret_type = ctx.default_return_type
    if isinstance(ret_type, Instance) and (
        ret_type.type.fullname in (_AGGREGATOR_FULLNAME, _PXTITERATOR_FULLNAME)
        or any(base.type.fullname in (_AGGREGATOR_FULLNAME, _PXTITERATOR_FULLNAME) for base in ret_type.type.bases)
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


def adjust_iterator_methods(ctx: ClassDefContext) -> bool:
    """
    Same idea as `adjust_uda_methods`, but for the `@pxt.iterator` decorator.
    """
    args_arg = nodes.Argument(nodes.Var('args'), AnyType(TypeOfAny.special_form), None, nodes.ARG_STAR)
    kwargs_arg = nodes.Argument(nodes.Var('kwargs'), AnyType(TypeOfAny.special_form), None, nodes.ARG_STAR2)
    add_method_to_class(ctx.api, ctx.cls, '__init__', args=[args_arg, kwargs_arg], return_type=NoneType())
    add_method_to_class(
        ctx.api,
        ctx.cls,
        'conditional_output_schema',
        args=[args_arg, kwargs_arg],
        return_type=AnyType(TypeOfAny.special_form),
        is_classmethod=True,
    )
    add_method_to_class(
        ctx.api,
        ctx.cls,
        'validate',
        args=[args_arg, kwargs_arg],
        return_type=AnyType(TypeOfAny.special_form),
        is_classmethod=True,
    )
    return True
