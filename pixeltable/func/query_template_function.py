from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable import exprs

from .function import Function
from .signature import Signature


class QueryTemplateFunction(Function):
    """A parameterized query/DataFrame from which an executable DataFrame is created with a function call."""

    @classmethod
    def create(
            cls, template_callable: Callable, param_types: Optional[list[pxt.ColumnType]], path: str, name: str
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
        sig = Signature(return_type=pxt.JsonType(), parameters=params)
        return QueryTemplateFunction(template_df, sig, path=path, name=name)

    def __init__(
            self, template_df: Optional['pxt.DataFrame'], sig: Optional[Signature], path: Optional[str] = None,
            name: Optional[str] = None,
    ):
        super().__init__(sig, self_path=path)
        self.self_name = name
        self.template_df = template_df

        # if we're running as part of an ongoing update operation, we need to use the same connection, otherwise
        # we end up with a deadlock
        # TODO: figure out a more general way to make execution state available
        self.conn: Optional[sql.engine.Connection] = None

        # convert defaults to Literals
        self.defaults: dict[str, exprs.Literal] = {}  # key: param name, value: default value converted to a Literal
        param_types = self.template_df.parameters()
        for param in [p for p in self.signature.parameters.values() if p.has_default()]:
            assert param.name in param_types
            param_type = param_types[param.name]
            literal_default = exprs.Literal(param.default, col_type=param_type)
            self.defaults[param.name] = literal_default

    def set_conn(self, conn: Optional[sql.engine.Connection]) -> None:
        self.conn = conn

    def exec(self, *args: Any, **kwargs: Any) -> Any:
        bound_args = self.signature.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {param_name: default for param_name, default in self.defaults.items() if param_name not in bound_args})
        bound_df = self.template_df.bind(bound_args)
        result = bound_df._collect(self.conn)
        return list(result)

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def _as_dict(self) -> dict:
        return {'name': self.name, 'signature': self.signature.as_dict(), 'df': self.template_df.as_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        from pixeltable.dataframe import DataFrame
        return cls(DataFrame.from_dict(d['df']), Signature.from_dict(d['signature']), name=d['name'])
