from __future__ import annotations

import inspect
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Union, overload

from pixeltable import catalog, exceptions as excs, exprs, func, type_system as ts

from .function import Function
from .signature import Signature

if TYPE_CHECKING:
    from pixeltable import DataFrame


class QueryTemplateFunction(Function):
    """A parameterized query/DataFrame from which an executable DataFrame is created with a function call."""

    template_df: Optional['DataFrame']
    self_name: Optional[str]
    _comment: Optional[str]

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
        return QueryTemplateFunction(template_df, sig, path=path, name=name, comment=inspect.getdoc(template_callable))

    def __init__(
        self,
        template_df: Optional['DataFrame'],
        sig: Signature,
        path: Optional[str] = None,
        name: Optional[str] = None,
        comment: Optional[str] = None,
    ):
        assert sig is not None
        super().__init__([sig], self_path=path)
        self.self_name = name
        self.template_df = template_df
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
        bound_df = self.template_df.bind(bound_args)
        result = await bound_df._acollect()
        return list(result)

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def comment(self) -> Optional[str]:
        return self._comment

    def _as_dict(self) -> dict:
        return {'name': self.name, 'signature': self.signature.as_dict(), 'df': self.template_df.as_dict()}

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


def retrieval_udf(
    table: catalog.Table,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Iterable[Union[str, exprs.ColumnRef]]] = None,
    limit: Optional[int] = 10,
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

    # Construct the dataframe
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
    query_signature = func.Signature(return_type=ts.JsonType(), parameters=query_params)

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

    fn = func.QueryTemplateFunction(df, query_signature, name=name, comment=description)
    return fn
