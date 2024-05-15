from __future__ import annotations
import inspect
from typing import Dict, Optional, Any, Callable

import sqlalchemy as sql

import pixeltable
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .function import Function
from .signature import Signature, Parameter


class QueryTemplate:
    def __init__(self, template_callable: Callable, param_types: Optional[list[ts.ColumnType]], path: str, name: str):
        self.template_callable = template_callable
        sig = inspect.signature(template_callable)
        # we exclude the first parameter (= the Table) from the Function signature
        py_params = list(sig.parameters.values())[1:]
        self.params = Signature.create_parameters(py_params=py_params, param_types=param_types)
        self.params_by_name = {p.name: p for p in self.params}
        self.path = path
        self.name = name

    def bind(self, t: pixeltable.Table) -> QueryTemplateFunction:
        import pixeltable.exprs as exprs
        var_exprs = [exprs.Variable(param.name, param.col_type) for param in self.params]
        # call the function with the parameter expressions to construct a DataFrame with parameters
        template = self.template_callable(t, *var_exprs)
        from pixeltable import DataFrame
        assert isinstance(template, DataFrame)
        sig = Signature(return_type=ts.JsonType(), parameters=self.params)
        return QueryTemplateFunction(template, sig, self_path=self.path, name=self.name)


class QueryTemplateFunction(Function):
    """A parameterized query/DataFrame from which an executable DataFrame is created with a function call."""

    def __init__(
            self, df: 'pixeltable.DataFrame', sig: Signature, self_path: Optional[str] = None,
            name: Optional[str] = None):
        self.df = df
        self.self_name = name
        self.param_types = df.parameters()
        # if we're running as part of an ongoing update operation, we need to use the same connection, otherwise
        # we end up with a deadlock
        # TODO: figure out a more general way to make execution state available
        self.conn: Optional[sql.engine.Connection] = None

        # convert defaults to Literals
        import pixeltable.exprs as exprs
        self.defaults: Dict[str, exprs.Literal] = {}  # key: param name, value: default value converted to a Literal
        for param in [p for p in sig.parameters.values() if p.has_default()]:
            assert param.name in self.param_types
            param_type = self.param_types[param.name]
            literal_default = exprs.Literal(param.default, col_type=param_type)
            self.defaults[param.name] = literal_default

        super().__init__(sig, self_path=self_path)

    def set_conn(self, conn: Optional[sql.engine.Connection]) -> None:
        self.conn = conn

    def exec(self, *args: Any, **kwargs: Any) -> Any:
        bound_args = self.signature.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {param_name: default for param_name, default in self.defaults.items() if param_name not in bound_args})
        bound_df = self.df.bind(bound_args)
        result = bound_df._collect(self.conn)
        return list(result)

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def _as_dict(self) -> Dict:
        return {'name': self.name, 'signature': self.signature.as_dict(), 'df': self.df.as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict) -> Function:
        from pixeltable.dataframe import DataFrame
        return cls(DataFrame.from_dict(d['df']), Signature.from_dict(d['signature']), name=d['name'])
