from __future__ import annotations

import inspect
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Iterable, overload

from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts

from .function import Function
from .signature import Signature

if TYPE_CHECKING:
    from pixeltable import Query


class QueryTemplateFunction(Function):
    """A parameterized query from which an executable Query is created with a function call."""

    template_query: 'Query' | None
    self_name: str | None
    return_scalar: bool
    _comment: str | None

    @classmethod
    def create(
        cls,
        template_callable: Callable,
        param_types: list[ts.ColumnType] | None,
        path: str,
        name: str,
        return_scalar: bool,
    ) -> QueryTemplateFunction:
        # we need to construct a template df and a signature
        py_sig = inspect.signature(template_callable)
        py_params = list(py_sig.parameters.values())
        params = Signature.create_parameters(py_params=py_params, param_types=param_types)
        # invoke template_callable with parameter expressions to construct a Query with parameters
        var_exprs = [exprs.Variable(param.name, param.col_type) for param in params]
        template_query = template_callable(*var_exprs)
        from pixeltable import Query

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
        template_query: 'Query' | None,
        params: list[func.Parameter],
        return_scalar: bool = False,
        path: str | None = None,
        name: str | None = None,
        comment: str | None = None,
    ):
        schema = template_query.schema
        # Single-column queries return a variadic list of that column's type directly, rather
        # than wrapping each row in a single-field dict.
        row_schema: ts.ColumnType
        assert not return_scalar or len(schema) == 1
        if return_scalar:
            row_schema = next(iter(schema.values()))
        else:
            row_schema = ts.JsonType(ts.JsonType.TypeSchema(type_spec=dict(schema)))
        return_type = ts.JsonType(ts.JsonType.TypeSchema(type_spec=[], variadic_type=row_schema))
        sig = Signature(return_type=return_type, parameters=params)
        super().__init__([sig], self_path=path)
        self.self_name = name
        self.template_query = template_query
        self.return_scalar = return_scalar
        self._comment = comment

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
            {
                param.name: param.default
                for param in self.signature.parameters.values()
                if param.has_default() and param.name not in bound_args
            }
        )
        bound_df = self.template_query.bind(bound_args)
        result = await bound_df._acollect()
        if self.return_scalar:
            col_name = next(iter(self.template_query.schema))
            return [row[col_name] for row in result]
        return list(result)

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

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
        return cls(
            Query.from_dict(d['df']),
            list(sig.parameters.values()),
            return_scalar=d.get('return_scalar', False),
            name=d['name'],
        )


@overload
def query(py_fn: Callable) -> QueryTemplateFunction: ...


@overload
def query(
    *, param_types: list[ts.ColumnType] | None = None, return_scalar: bool = False
) -> Callable[[Callable], QueryTemplateFunction]: ...


def query(*args: Any, **kwargs: Any) -> Any:
    def make_query_template(
        py_fn: Callable, param_types: list[ts.ColumnType] | None, return_scalar: bool
    ) -> QueryTemplateFunction:
        if py_fn.__module__ != '__main__' and py_fn.__name__.isidentifier():
            # this is a named function in a module
            function_path = f'{py_fn.__module__}.{py_fn.__qualname__}'
        else:
            function_path = None
        query_name = py_fn.__name__
        query_fn = QueryTemplateFunction.create(
            py_fn, return_scalar=return_scalar, param_types=param_types, path=function_path, name=query_name
        )
        return query_fn

        # TODO: verify that the inferred return type matches that of the template
        # TODO: verify that the signature doesn't contain batched parameters

    if len(args) == 1:
        assert len(kwargs) == 0 and callable(args[0])
        return make_query_template(args[0], None, False)
    else:
        param_types = kwargs.pop('param_types', None)
        return_scalar = kwargs.pop('return_scalar', False)
        if len(kwargs) > 0:
            raise excs.Error(f'@pxt.query(): unknown argument(s): {", ".join(kwargs)}')
        return lambda py_fn: make_query_template(py_fn, param_types, return_scalar)


def retrieval_udf(
    table: catalog.Table,
    name: str | None = None,
    description: str | None = None,
    parameters: Iterable[str | exprs.ColumnRef] | None = None,
    limit: int | None = 10,
) -> func.QueryTemplateFunction:
    """
    Constructs a retrieval UDF for the given table. The retrieval UDF is a UDF whose parameters are
    columns of the table and whose return value is a list of rows from the table. The return value of
    ```python
    f(col1=x, col2=y, ...)
    ```
    will be a list of all rows from the table that match the specified arguments.

    Args:
        table: The table to use as the dataset for the retrieval tool.
        name: The name of the tool. If not specified, then the name of the table will be used by default.
        description: The description of the tool. If not specified, then a default description will be generated.
        parameters: The columns of the table to use as parameters. If not specified, all data columns
            (non-computed columns) will be used as parameters.

            All of the specified parameters will be required parameters of the tool, regardless of their status
            as columns.
        limit: The maximum number of rows to return. If not specified, then all matching rows will be returned.

    Returns:
        A list of dictionaries containing data from the table, one per row that matches the input arguments.
        If there are no matching rows, an empty list will be returned.
    """
    # Argument validation
    col_refs: list[exprs.ColumnRef]
    # TODO: get rid of references to ColumnRef internals and replace instead with a public interface
    col_names = table.columns()
    if parameters is None:
        col_refs = [table[col_name] for col_name in col_names if not table[col_name].col.is_computed]
    else:
        for param in parameters:
            if isinstance(param, str) and param not in col_names:
                raise excs.Error(f'The specified parameter {param!r} is not a column of the table {table._path()!r}')
        col_refs = [table[param] if isinstance(param, str) else param for param in parameters]

    if len(col_refs) == 0:
        raise excs.Error('Parameter list cannot be empty.')

    # Construct the Query
    predicates = [col_ref == exprs.Variable(col_ref.col.name, col_ref.col.col_type) for col_ref in col_refs]
    where_clause = reduce(lambda c1, c2: c1 & c2, predicates)
    df = table.select().where(where_clause)
    if limit is not None:
        df = df.limit(limit)

    # Construct the signature
    query_params = [
        func.Parameter(col_ref.col.name, col_ref.col.col_type, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for col_ref in col_refs
    ]

    # Construct a name and/or description if not provided
    if name is None:
        name = table._name
    if description is None:
        description = (
            f'Retrieves an entry from the dataset {name!r} that matches the given parameters.\n\nParameters:\n'
        )
        description += '\n'.join(
            [f'    {col_ref.col.name}: of type `{col_ref.col.col_type._to_base_str()}`' for col_ref in col_refs]
        )

    fn = func.QueryTemplateFunction(df, query_params, name=name, comment=description)
    return fn
