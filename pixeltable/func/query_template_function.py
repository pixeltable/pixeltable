from __future__ import annotations

import inspect
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Iterable, overload

from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts

from .function import Function
from .signature import Signature

if TYPE_CHECKING:
    from pixeltable._query_base import QueryBase


class QueryTemplateFunction(Function):
    """A parameterized query from which an executable Query is created with a function call."""

    template_query: 'QueryBase' | None
    self_name: str | None
    return_scalar: bool
    single_row: bool
    _comment: str | None

    @classmethod
    def create(
        cls, template_callable: Callable, param_types: list[ts.ColumnType] | None, path: str, name: str
    ) -> QueryTemplateFunction:
        # we need to construct a template df and a signature
        py_sig = inspect.signature(template_callable, eval_str=True)
        py_params = list(py_sig.parameters.values())
        params = Signature.create_parameters(py_params=py_params, param_types=param_types)
        # invoke template_callable with parameter expressions to construct a Query with parameters
        var_exprs = [exprs.Variable(param.name, param.col_type) for param in params]
        template_query = template_callable(*var_exprs)
        from pixeltable._query_base import QueryBase

        assert isinstance(template_query, QueryBase)
        return QueryTemplateFunction(
            template_query, params, path=path, name=name, comment=inspect.getdoc(template_callable)
        )

    def __init__(
        self,
        template_query: 'QueryBase' | None,
        params: list[func.Parameter],
        return_type: ts.ColumnType | None = None,
        path: str | None = None,
        name: str | None = None,
        comment: str | None = None,
    ):
        # TODO:
        # - re-resolve schema at execution time if this is a SELECT *
        self.template_query = template_query
        if return_type is None:
            select_list = template_query.select_list
            limit = template_query.limit_val
            # select_list[0][1] is None: no explicitly assigned name
            return_scalar = select_list is not None and len(select_list) == 1 and select_list[0][1] is None
            single_row = template_query._is_ungrouped_aggregate or (isinstance(limit, exprs.Literal) and limit.val == 1)
        else:
            shapes = [
                (single_row, return_scalar)
                for single_row in (False, True)
                for return_scalar in (False, True)
                if not return_scalar or len(template_query.schema) == 1
            ]
            # the untyped Json of older functions matches no shape; those returned lists of dicts
            single_row, return_scalar = next(
                (shape for shape in shapes if self._result_type(*shape) == return_type), (False, False)
            )
        sig = Signature(return_type=self._result_type(single_row, return_scalar), parameters=params)
        super().__init__([sig], self_path=path)
        self.self_name = name
        self.return_scalar = return_scalar
        self.single_row = single_row
        self._comment = comment

    def _result_type(self, single_row: bool, return_scalar: bool) -> ts.ColumnType:
        schema = self.template_query.schema
        row_type: ts.ColumnType
        if return_scalar:
            row_type = next(iter(schema.values()))
        else:
            row_type = ts.JsonType(ts.JsonType.TypeSchema(type_spec=dict(schema)))
        if single_row:
            # nullable: the query may return no rows
            return row_type.copy(nullable=True)
        return ts.JsonType(ts.JsonType.TypeSchema(type_spec=[], variadic_type=row_type))

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        pass  # only one signature supported for QueryTemplateFunction

    @property
    def is_storable(self) -> bool:
        # the query is serialized inline by value, so it is storable only if every function embedded in its
        # clauses is itself storable
        if self.template_query is None:
            return True
        return all(
            fn_call.fn.is_storable
            for query_expr in self.template_query._component_exprs()
            for fn_call in query_expr.subexprs(exprs.FunctionCall)
        )

    @property
    def queries_tables(self) -> bool:
        """Whether this template queries tables, rather than a model that is not bound to one yet."""
        from pixeltable._query import Query

        return isinstance(self.template_query, Query)

    @property
    def is_async(self) -> bool:
        return True

    async def aexec(self, *args: Any, **kwargs: Any) -> Any:
        # assert not self.is_polymorphic
        bound_args = self.signature.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {
                param.name: param.default_value
                for param in self.signature.parameters.values()
                if param.has_default() and param.name not in bound_args
            }
        )
        from pixeltable._query import Query

        # a route binds a template declared over a model to its table before serving it; nothing else does,
        # so a template that is still declared over a model here has no table to read
        if not isinstance(self.template_query, Query):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Query UDF {self.name!r} is declared over a model rather than a table, so it cannot be evaluated.',
            )
        result = await self.template_query._acollect(args=bound_args)
        rows: list[Any] = list(result)
        if self.return_scalar:
            col_name = next(iter(self.template_query.schema))
            rows = [row[col_name] for row in rows]
        if self.single_row:
            return rows[0] if len(rows) > 0 else None
        return rows

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def comment(self) -> str | None:
        return self._comment

    def _as_dict(self) -> dict:
        return {'name': self.name, 'signature': self.signature.as_dict(), 'df': self.template_query.as_dict()}

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        from pixeltable._query_base import QueryBase

        sig = Signature.from_dict(d['signature'])
        try:
            template_query = QueryBase.from_dict(d['df'])
        except excs.NotFoundError as e:
            # template_query references a table or column that no longer exists
            return func.InvalidFunction(d['name'], d, f'the @pxt.query UDF {d["name"]!r} cannot be loaded: {e}')
        return cls(template_query, list(sig.parameters.values()), return_type=sig.return_type, name=d['name'])


@overload
def query(py_fn: Callable) -> QueryTemplateFunction: ...


@overload
def query(*, param_types: list[ts.ColumnType] | None = None) -> Callable[[Callable], QueryTemplateFunction]: ...


def query(*args: Any, **kwargs: Any) -> Any:
    """Turn a function that returns a parameterized query into a function that runs the query.

    The decorated function builds the query from its parameters. A call to the resulting function in an expression,
    such as a computed column, runs the query for each row, with that row's values as the arguments.

    Args:
        param_types: The Pixeltable types of the parameters, for parameters without type annotations.

    Returns:
        The query function. A call returns a list of rows, each a dictionary from the selected column names to their
        values. If the query selects one unnamed item, as in `select(t.text)`, each row is that item's value instead.
        If the query aggregates without `group_by()` or has `limit(1)`, a call returns the one row instead of a list,
        or None if there is no row.

    Examples:
        The texts of the five chunks most similar to a question, given a table `chunks` with an embedding index on
        `text`:

        >>> @pxt.query
        ... def top_chunks(question: str) -> pxt.Query:
        ...     sim = chunks.text.similarity(string=question)
        ...     return chunks.order_by(sim, asc=False).select(chunks.text).limit(5)

        The total of a customer's orders, as a single value in a computed column of the table `customers`:

        >>> @pxt.query
        ... def customer_total(customer: str) -> pxt.Query:
        ...     return orders.where(orders.customer == customer).select(
        ...         pxtf.sum(orders.amount)
        ...     )
        >>> customers.add_computed_column(total=customer_total(customers.customer))
    """

    def make_query_template(py_fn: Callable, param_types: list[ts.ColumnType] | None) -> QueryTemplateFunction:
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
        param_types = kwargs.pop('param_types', None)
        if len(kwargs) > 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'@pxt.query(): unknown argument(s): {", ".join(kwargs)}'
            )
        return lambda py_fn: make_query_template(py_fn, param_types)


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
    >>> f(col1=x, col2=y, ...)
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
        If there are no matching rows, an empty list will be returned. With `limit=1`, the UDF returns the one
        matching row, or None.
    """
    # Argument validation
    col_refs: list[exprs.ColumnRef]
    col_names = table.columns()
    if parameters is None:
        col_refs = [table[col_name] for col_name in col_names if not table[col_name].column_md.is_computed]
    else:
        for param in parameters:
            if isinstance(param, str) and param not in col_names:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'The specified parameter {param!r} is not a column of the table {table._path()!r}',
                )
        col_refs = [table[param] if isinstance(param, str) else param for param in parameters]

    if len(col_refs) == 0:
        raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, 'Parameter list cannot be empty.')

    # Construct the Query
    predicates = [col_ref == exprs.Variable(col_ref.column_md.name, col_ref.column_md.col_type) for col_ref in col_refs]
    where_clause = reduce(lambda c1, c2: c1 & c2, predicates)
    df = table.select().where(where_clause)
    if limit is not None:
        df = df.limit(limit)

    # Construct the signature
    query_params = [
        func.Parameter(col_ref.column_md.name, col_ref.column_md.col_type, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for col_ref in col_refs
    ]

    # Construct a name and/or description if not provided
    if name is None:
        name = table._name()
    if description is None:
        description = (
            f'Retrieves an entry from the dataset {name!r} that matches the given parameters.\n\nParameters:\n'
        )
        description += '\n'.join(
            [
                f'    {col_ref.column_md.name}: of type `{col_ref.column_md.col_type._to_base_str()}`'
                for col_ref in col_refs
            ]
        )

    fn = func.QueryTemplateFunction(df, query_params, name=name, comment=description)
    return fn
