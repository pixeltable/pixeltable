"""The service instances of a hosted database, managed through the cloud's management API.

The control plane owns their lifetime: this asks it to create, update, start, stop and delete an instance,
and reports the state it comes back with. Every mutation is followed by polling until the state settles,
since the control plane returns as soon as it has accepted the request.
"""

from __future__ import annotations

import time

from pixeltable import catalog, exceptions as excs
from pixeltable.service import management_client
from pixeltable.service.management_protocol import (
    CreateServiceInstanceRequest,
    DeleteServiceInstanceRequest,
    ListServiceInstancesRequest,
    ListServiceInstancesResponse,
    StartServiceInstanceRequest,
    StopServiceInstanceRequest,
    UpdateServiceInstanceRequest,
)
from pixeltable.utils.app_module import module_name

from ._app import service_router
from .service_instance import ServiceInstance, ServiceInstanceRecord, ServiceInstanceState
from .service_manager import ServiceManagerBase


class ServiceManagerProxy(ServiceManagerBase):
    """The manager of service instances of one hosted database."""

    _POLL_INTERVAL = 5.0
    _POLL_TIMEOUT = 300.0

    catalog_uri: catalog.Path

    def __init__(self, catalog_uri: catalog.Path) -> None:
        assert catalog_uri.org is not None and catalog_uri.db is not None
        self.catalog_uri = catalog_uri

    @property
    def _org(self):
        return self.catalog_uri.org

    @property
    def _db(self):
        return self.catalog_uri.db

    def get(self, name: str, base_path: str = '') -> ServiceInstance | None:
        # read the listing rather than get one instance: the management API reports a name it does not hold
        # as an error response, which would put a status code in this method's control flow
        return next((i for i in self.list(base_path) if i.service_name == name), None)

    def list(self, base_path: str = '', recursive: bool = False) -> list[ServiceInstance]:
        response = ListServiceInstancesResponse.model_validate(
            management_client.api_call(ListServiceInstancesRequest(org=self._org, db=self._db))
        )
        return [ServiceInstance(r, self) for r in response.instances if self._serves(r, base_path, recursive)]

    def start(self, app_file: str, name: str, base_path: str = '', *, otel: bool = False) -> ServiceInstance:
        spec = service_router(app_file, name).service_spec(name)
        app_module = module_name(app_file, subject='application file')
        instance = self.get(name, base_path)

        if instance is None:
            management_client.api_call(
                CreateServiceInstanceRequest(
                    org=self._org,
                    db=self._db,
                    service_name=name,
                    base_path=base_path,
                    spec=spec,
                    app_module=app_module,
                    otel=otel,
                )
            )
        else:
            if (instance.spec, instance.record.app_module, instance.otel) != (spec, app_module, otel):
                management_client.api_call(
                    UpdateServiceInstanceRequest(
                        org=self._org, db=self._db, service_name=name, spec=spec, app_module=app_module, otel=otel
                    )
                )
                instance = self._settled(name, base_path)
            if instance.state is ServiceInstanceState.AVAILABLE:
                return instance
            management_client.api_call(StartServiceInstanceRequest(org=self._org, db=self._db, service_name=name))

        started = self._settled(name, base_path)
        if started.state is not ServiceInstanceState.AVAILABLE:
            detail = '' if started.record.error is None else f': {started.record.error}'
            raise excs.Error(
                excs.ErrorCode.INTERNAL_ERROR, f'Service {name!r} did not start; it is {started.state.value}{detail}'
            )
        return started

    def stop(self, instance: ServiceInstance) -> None:
        management_client.api_call(
            StopServiceInstanceRequest(org=self._org, db=self._db, service_name=instance.service_name)
        )
        self._settled(instance.service_name, instance.base_path)

    def delete(self, instance: ServiceInstance) -> None:
        management_client.api_call(
            DeleteServiceInstanceRequest(org=self._org, db=self._db, service_name=instance.service_name)
        )

    def _serves(self, record: ServiceInstanceRecord, base_path: str, recursive: bool) -> bool:
        if record.base_path == base_path:
            return True
        return recursive and (base_path == '' or record.base_path.startswith(f'{base_path}/'))

    def _settled(self, name: str, base_path: str) -> ServiceInstance:
        """Poll the named instance until its state settles, and return it."""
        deadline = time.monotonic() + self._POLL_TIMEOUT
        while True:
            instance = self.get(name, base_path)
            if instance is None:
                raise excs.Error(
                    excs.ErrorCode.INTERNAL_ERROR, f'Service {name!r} is no longer in {self.catalog_uri.uri}'
                )
            if instance.state is not ServiceInstanceState.STARTING:
                return instance
            if time.monotonic() >= deadline:
                raise excs.Error(
                    excs.ErrorCode.INTERNAL_ERROR,
                    f'Service {name!r} is still {instance.state.value} after {self._POLL_TIMEOUT:.0f}s',
                )
            time.sleep(self._POLL_INTERVAL)
