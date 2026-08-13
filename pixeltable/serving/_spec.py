"""What an application file declares about a service, as a record that can be serialized, stored and compared."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class RouteSpec(TypedDict):
    """One route of a service definition.

    Location-independent: a route declared against a model names the model, never the catalog path the model
    is bound to. Two routes with equal specifications serve the same contract.
    """

    method: Literal['GET', 'POST']
    path: str
    route_type: Literal['insert', 'update', 'delete', 'compute', 'query']

    # the table name a model declares, for a route declared against a model; the catalog path of the table,
    # for one declared against a table. A query route written against tables has neither: the tables it runs
    # against are internal to the function named by `query`.
    model: str | None
    table: str | None

    # for a query route, `inputs` are its query's parameters and `outputs` the fields of its response
    inputs: list[str]
    uploadfile_inputs: list[str]
    outputs: list[str]
    match_columns: list[str]

    background: bool
    return_fileresponse: bool
    one_row: bool
    export_sql: dict[str, Any] | None
    query: str | None  # the symbol path of the function a query route calls


class ServiceSpec(TypedDict):
    """Everything an application file declares about one service.

    A deployment is this definition applied to a catalog directory, so nothing here names one.
    """

    name: str
    prefix: str
    routes: list[RouteSpec]
