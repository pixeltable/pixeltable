"""Query template functions (`@pxt.query`, `pxt.retrieval_udf()`), upstream `pixeltable.func.query_template_function`.

The template is a `Query` built by calling the decorated function with `Variable` parameters; the backend binds the
arguments and runs the query for each call.
"""

from __future__ import annotations

import inspect
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Iterable

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import _core

from ._function import Function, InvalidFunction
from ._signature import Parameter, Signature

if TYPE_CHECKING:
    from pixeltable._query import Query


def _query_return_type(template_query: Query, return_scalar: bool) -> ts.ColumnType:
    schema = template_query.schema
    assert not return_scalar or len(schema) == 1
    row_schema: ts.ColumnType
    if return_scalar:
        row_schema = next(iter(schema.values()))
    else:
        row_schema = ts.JsonType(ts.JsonType.TypeSchema(type_spec=dict(schema)))
    return ts.JsonType(ts.JsonType.TypeSchema(type_spec=[], variadic_type=row_schema))


class QueryTemplateFunction(Function):
    """A parameterized query from which an executable query is created with a function call."""

    template_query: Query
    self_name: str | None
    return_scalar: bool
    _comment: str | None

    @classmethod
    def create(
        cls,
        template_callable: Callable,
        param_types: list[ts.ColumnType] | None,
        path: str | None,
        name: str,
        return_scalar: bool,
    ) -> QueryTemplateFunction:
        py_sig = inspect.signature(template_callable, eval_str=True)
        params = Signature.create_parameters(py_params=list(py_sig.parameters.values()), param_types=param_types)
        var_exprs = [_core.variable(param.name, param.col_type) for param in params]
        template_query = template_callable(*var_exprs)
        from pixeltable._query import Query

        assert isinstance(template_query, Query)
        return QueryTemplateFunction(
            template_query,
            params,
            return_scalar=return_scalar,
            path=path,
            name=name,
            comment=inspect.getdoc(template_callable),
        )

    def __init__(
        self,
        template_query: Query,
        params: list[Parameter],
        return_scalar: bool = False,
        path: str | None = None,
        name: str | None = None,
        comment: str | None = None,
        return_type: ts.ColumnType | None = None,
    ):
        if return_type is None:
            return_type = _query_return_type(template_query, return_scalar)
        sig = Signature(return_type=return_type, parameters=params)
        super().__init__([sig], self_path=path)
        self.self_name = name
        self.template_query = template_query
        self.return_scalar = return_scalar
        self._comment = comment

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        pass  # a query template has a single signature

    def _core_spec(self) -> dict[str, Any]:
        return {
            'kind': 'query',
            'name': self.name,
            'display_name': self.display_name,
            'self_path': self.self_path,
            'signatures': [self.signature._to_core()],
            'query': self.template_query._spec,
            'return_scalar': self.return_scalar,
        }

    @property
    def queries_tables(self) -> bool:
        return True

    @property
    def is_async(self) -> bool:
        return True

    @property
    def display_name(self) -> str:
        return self.self_name  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return self.self_name  # type: ignore[return-value]

    def comment(self) -> str | None:
        return self._comment

    def _as_dict(self) -> dict:
        return {
            'name': self.name,
            'signature': self.signature.as_dict(),
            'df': self.template_query.as_dict(),
            'return_scalar': self.return_scalar,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        from pixeltable._query import Query

        sig = Signature.from_dict(d['signature'])
        try:
            # called by the backend under its catalog lock: no catalog access here
            template_query = Query(_core.QuerySpec._from_dict_unchecked(d['df']))
        except excs.NotFoundError as e:
            return InvalidFunction(d['name'], d, f'the @pxt.query UDF {d["name"]!r} cannot be loaded: {e}')
        return cls(
            template_query,
            list(sig.parameters.values()),
            return_scalar=d.get('return_scalar', False),
            name=d['name'],
            return_type=sig.return_type,
        )


def query(*args: Any, **kwargs: Any) -> Any:
    """Decorator for query template functions."""

    def make_query_template(
        py_fn: Callable, param_types: list[ts.ColumnType] | None, return_scalar: bool
    ) -> QueryTemplateFunction:
        if py_fn.__module__ != '__main__' and py_fn.__name__.isidentifier():
            function_path: str | None = f'{py_fn.__module__}.{py_fn.__qualname__}'
        else:
            function_path = None
        return QueryTemplateFunction.create(
            py_fn, return_scalar=return_scalar, param_types=param_types, path=function_path, name=py_fn.__name__
        )

    if len(args) == 1:
        assert len(kwargs) == 0 and callable(args[0])
        return make_query_template(args[0], None, False)
    param_types = kwargs.pop('param_types', None)
    return_scalar = kwargs.pop('return_scalar', False)
    if len(kwargs) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'@pxt.query(): unknown argument(s): {", ".join(kwargs)}'
        )
    return lambda py_fn: make_query_template(py_fn, param_types, return_scalar)


def retrieval_udf(
    table: Any,
    name: str | None = None,
    description: str | None = None,
    parameters: Iterable[Any] | None = None,
    limit: int | None = 10,
) -> QueryTemplateFunction:
    """
    Constructs a retrieval UDF for the given table: its parameters are columns of the table and it returns the list
    of rows of the table that match the arguments.

    Args:
        table: The table to use as the dataset for the retrieval tool.
        name: The name of the tool. If not specified, then the name of the table will be used by default.
        description: The description of the tool. If not specified, then a default description will be generated.
        parameters: The columns of the table to use as parameters. If not specified, all data columns
            (non-computed columns) will be used as parameters.
        limit: The maximum number of rows to return. If not specified, then all matching rows will be returned.
    """
    md = table.get_metadata()
    columns = md['columns']
    col_names = list(columns.keys())
    param_names: list[str]
    if parameters is None:
        param_names = [n for n in col_names if not columns[n]['is_computed']]
    else:
        param_names = []
        for param in parameters:
            if isinstance(param, str):
                if param not in col_names:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_CONFIGURATION,
                        f'The specified parameter {param!r} is not a column of the table {md["path"]!r}',
                    )
                param_names.append(param)
            else:
                param_names.append(param.default_column_name())
    if len(param_names) == 0:
        raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, 'Parameter list cannot be empty.')

    col_refs = [table[n] for n in param_names]
    predicates = [ref == _core.variable(n, ref.col_type) for n, ref in zip(param_names, col_refs)]
    where_clause = reduce(lambda c1, c2: c1 & c2, predicates)
    q = table.select().where(where_clause)
    if limit is not None:
        q = q.limit(limit)

    query_params = [
        Parameter(n, ref.col_type, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n, ref in zip(param_names, col_refs)
    ]
    if name is None:
        name = md['name']
    if description is None:
        description = f'Retrieves an entry from the dataset {name!r} that matches the given parameters.\n\nParameters:\n'
        description += '\n'.join(f'    {n}: of type `{ref.col_type._to_base_str()}`' for n, ref in zip(param_names, col_refs))
    return QueryTemplateFunction(q, query_params, name=name, comment=description)
