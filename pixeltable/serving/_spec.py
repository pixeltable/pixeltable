from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict  # pydantic requires this spelling on Python < 3.12


class RouteSpec(TypedDict):
    """Complete metadata of a service route.

    Location-independent: a route declared against a model names the model, never the catalog path the model
    is bound to. Two routes with equal specifications serve the same contract.

    This is also used for display purposes and must never contain any cleartext secrets.
    """

    method: Literal['GET', 'POST']
    path: str
    route_type: Literal['insert', 'update', 'delete', 'compute', 'query']

    # at most one is set: model holds the table name a model declares, table the catalog path of a table
    model: str | None
    table: str | None

    # for a query route, inputs are its query's parameters and outputs the fields of its response
    inputs: list[str]
    uploadfile_inputs: list[str]
    outputs: list[str]
    match_columns: list[str]

    background: bool
    return_fileresponse: bool
    one_row: bool
    export_sql: dict[str, Any] | None  # the optional export target of a route, as produced by SqlExport.display_dict()
    query: str | None  # the symbol path of the query udf


class ServiceSpec(TypedDict):
    """Complete metadata of a service."""

    name: str
    prefix: str
    routes: list[RouteSpec]

    # paths served by a custom FastAPI instance that don't come from FastAPIRouter instances
    app_paths: list[str]
