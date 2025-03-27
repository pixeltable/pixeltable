from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, overload

from pixeltable import exprs, type_system as ts

from .function import Function
from .signature import Signature

if TYPE_CHECKING:
    from pixeltable import DataFrame


class QueryTemplateFunction(Function):
    """A parameterized query/DataFrame from which an executable DataFrame is created with a function call."""

    template_df: Optional['DataFrame']
    self_name: Optional[str]
    # conn: Optional[sql.engine.Connection]
    defaults: dict[str, exprs.Literal]

    @classmethod
    def create(
        cls, template_callable: Callable, param_types: Optional[list[ts.ColumnType]], path: str, name: str
    ) -> QueryTemplateFunction:
        # we need to construct a template df and a signature
        py_sig = inspect.signature(template_callable)
        py_params = list(py_sig.parameters.values())
        params = Signature.create_parameters(py_params=py_params, param_types=param_types)
        # invoke template_callable with parameter expressions to construct a DataFrame with parameters
        var_exprs = [exprs.Variable(param.name, param.col_type) for param in params]
        template_df = template_callable(*var_exprs)
        from pixeltable import DataFrame

        assert isinstance(template_df, DataFrame)
        # we take params and return json
        sig = Signature(return_type=ts.JsonType(), parameters=params)
        return QueryTemplateFunction(template_df, sig, path=path, name=name)

    def __init__(
        self, template_df: Optional['DataFrame'], sig: Signature, path: Optional[str] = None, name: Optional[str] = None
    ):
        assert sig is not None
        super().__init__([sig], self_path=path)
        self.self_name = name
        self.template_df = template_df

        # if we're running as part of an ongoing update operation, we need to use the same connection, otherwise
        # we end up with a deadlock
        # TODO: figure out a more general way to make execution state available
        # self.conn = None

        # convert defaults to Literals
        self.defaults = {}  # key: param name, value: default value converted to a Literal
        param_types = self.template_df.parameters()
        for param in [p for p in sig.parameters.values() if p.has_default()]:
            assert param.name in param_types
            param_type = param_types[param.name]
            literal_default = exprs.Literal(param.default, col_type=param_type)
            self.defaults[param.name] = literal_default

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        pass  # only one signature supported for QueryTemplateFunction

    @property
    def is_async(self) -> bool:
        return True

    async def aexec(self, *args: Any, **kwargs: Any) -> Any:
        # assert not self.is_polymorphic
        bound_args = self.signature.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {param_name: default for param_name, default in self.defaults.items() if param_name not in bound_args}
        )
        bound_df = self.template_df.bind(bound_args)
        result = await bound_df._acollect()
        return list(result)

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def _as_dict(self) -> dict:
        return {'name': self.name, 'signature': self.signatures[0].as_dict(), 'df': self.template_df.as_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        from pixeltable.dataframe import DataFrame

        return cls(DataFrame.from_dict(d['df']), Signature.from_dict(d['signature']), name=d['name'])


@overload
def query(py_fn: Callable) -> QueryTemplateFunction: ...


@overload
def query(*, param_types: Optional[list[ts.ColumnType]] = None) -> Callable[[Callable], QueryTemplateFunction]: ...


def query(*args: Any, **kwargs: Any) -> Any:
    def make_query_template(py_fn: Callable, param_types: Optional[list[ts.ColumnType]]) -> QueryTemplateFunction:
        if py_fn.__module__ != '__main__' and py_fn.__name__.isidentifier():
            # this is a named function in a module
            function_path = f'{py_fn.__module__}.{py_fn.__qualname__}'
        else:
            function_path = None
        query_name = py_fn.__name__
        query_fn = QueryTemplateFunction.create(py_fn, param_types=param_types, path=function_path, name=query_name)
        return query_fn

        # TODO: verify that the inferred return type matches that of the template
        # TODO: verify that the signature doesn't contain batched parameters

    if len(args) == 1:
        assert len(kwargs) == 0 and callable(args[0])
        return make_query_template(args[0], None)
    else:
        assert len(args) == 0 and len(kwargs) == 1 and 'param_types' in kwargs
        return lambda py_fn: make_query_template(py_fn, kwargs['param_types'])
