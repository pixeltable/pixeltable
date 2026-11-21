from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pydantic

from pixeltable import exceptions as excs
from pixeltable.catalog.model import typescript
from pixeltable.serving import FastAPIRouter


class _InsertRoute(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid', strict=True)

    table: str
    service: str
    path: str = pydantic.Field(pattern=r'^/([A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+$')
    inputs: list[str]
    outputs: list[str]


def load_module(path: Path, root: Path, name: str) -> ModuleType:
    exported = typescript.export_models(path, root)
    base = typescript.models_from_export(exported, module=path.stem)
    try:
        routes = pydantic.TypeAdapter(list[_InsertRoute]).validate_python(exported.get('routes', []))
    except pydantic.ValidationError as e:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_SCHEMA, f'{path.name}: invalid TypeScript insert routes: {e}'
        ) from e

    module = ModuleType(name)
    vars(module)['TableModel'] = base
    models = {model.__table_spec__['name']: model for model in base.defined_models()}
    routers: dict[str, FastAPIRouter] = {}
    registered: set[tuple[str, str]] = set()
    for route in routes:
        model = models.get(route.table)
        if model is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'{path.name}: {route.path} refers to unknown table {route.table!r}'
            )
        key = (route.service, route.path)
        if key in registered:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'{path.name}: {route.service!r} declares POST {route.path} twice'
            )
        registered.add(key)
        if route.service not in routers:
            routers[route.service] = FastAPIRouter(name=route.service)
        routers[route.service].add_insert_route(model, path=route.path, inputs=route.inputs, outputs=route.outputs)
    for index, router in enumerate(routers.values()):
        setattr(module, f'router_{index}', router)
    return module
