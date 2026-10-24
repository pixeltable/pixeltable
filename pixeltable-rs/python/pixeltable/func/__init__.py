"""Functions, UDFs, UDAs and iterators (upstream `pixeltable.func`).

Python introspects user functions (signatures from type hints, callables, templates); binding calls, typing them and
evaluating them happen in the backend (`pixeltable._core.Udf`, see ARCHITECTURE.md).
"""

# ruff: noqa: F401

from ._aggregate import AggregateFunction, Aggregator, uda
from ._callable import CallableFunction
from ._expr_template import ExprTemplate, ExprTemplateFunction
from ._function import Function, InvalidFunction, _resolve_function, resolve_symbol
from ._iterator import GeneratingFunction, GeneratingFunctionCall, PxtIterator, iterator
from ._query_template import QueryTemplateFunction, query, retrieval_udf
from ._signature import Batch, Parameter, Signature
from ._tools import Tool, ToolChoice, Tools, mcp_udfs
from ._udf import FunctionRegistry, expr_udf, make_function, udf
