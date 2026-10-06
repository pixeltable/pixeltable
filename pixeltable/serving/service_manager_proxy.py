"""The service instances of a hosted database, managed through the cloud's management API.

The control plane owns their lifetime: this module submits an instance's desired spec, or asks the control
plane to stop, restart or delete one. Each request becomes a generation of the instance, which the control
plane applies on its own; this module waits on the generation's receipt until it settles.
"""

from __future__ import annotations

import time
from typing import Sequence

import httpx

from pixeltable import catalog, exceptions as excs
from pixeltable.service import management_client, receipts
from pixeltable.service.management_protocol import (
    DeleteServiceInstanceRequest,
    DeleteServiceInstanceResponse,
    ExpectedGenerations,
    GetLogsRequest,
    GetLogsResponse,
    ListServiceInstancesRequest,
    ListServiceInstancesResponse,
    LogRecord,
    PrepareUpdateRequest,
    PrepareUpdateResponse,
    RestartServiceInstanceRequest,
    RestartServiceInstanceResponse,
    ServiceGeneration,
    ServiceMutation,
    StopServiceInstanceRequest,
    StopServiceInstanceResponse,
    SubmitUpdateRequest,
)
from pixeltable.utils.app_module import load_app_module, module_name, module_routers, service_spec, services_by_name
from pixeltable_cli.types import GenerationReceipt

from .service_instance import ServiceInstance, ServiceInstanceRecord, ServiceState
from .service_manager import ServiceManagerBase


class ServiceManagerProxy(ServiceManagerBase):
    """The manager of service instances of one hosted database."""

    _POLL_INTERVAL = 5.0
    _ENDPOINT_TIMEOUT = 60.0
    _ENDPOINT_PROBE_TIMEOUT = 10.0

    catalog_uri: catalog.Path

    def __init__(self, catalog_uri: catalog.Path) -> None:
        assert catalog_uri.org is not None and catalog_uri.db is not None
        self.catalog_uri = catalog_uri

    @property
    def _org(self) -> str:
        assert self.catalog_uri.org is not None
        return self.catalog_uri.org

    @property
    def _db(self) -> str:
        assert self.catalog_uri.db is not None
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

    def start(
        self,
        app_file: str,
        name: str,
        base_path: str = '',
        *,
        otel: bool = False,
        port: int | None = None,
        keep_release: bool = False,
        expected_generation: int | None = None,
        wait: bool = True,
    ) -> ServiceInstance:
        """Submit the named service in app_file as the desired spec of its instance at base_path.

        A new instance runs the release of the database's current desired generation, and so does an existing one
        unless keep_release is set, which keeps its current release. An available instance is not stopped: the
        control plane replaces its pods in place, and keeps the old ones serving until the new ones are ready.

        The file sets the routes, the module and tracing; resources and description are kept from the instance's
        desired spec, as read with the generation the submission is made against.

        expected_generation: the generation of the instance the caller's plan was computed against, 0 for an absent
            one; a submission against an older one is refused. None submits against the generation read here.
        wait: wait until the generation finishes, and raise if it fails or is superseded.
            Otherwise the returned instance carries the generation's receipt.
        """
        if port is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'A hosted service is reached at its own hostname, not a port; --port applies to a local target',
            )
        module = load_app_module(app_file, subject='application file')
        services = services_by_name(module, app_file)
        if name not in services:
            defined = ', '.join(sorted(services))
            raise excs.NotFoundError(
                excs.ErrorCode.SERVICE_NOT_FOUND, f'{app_file} defines no service named {name!r}; it defines: {defined}'
            )
        exists = self.get(name, base_path) is not None if expected_generation is None else expected_generation > 0
        mutation = ServiceMutation(
            service_name=name,
            base_path=base_path,
            spec=service_spec(name, services[name], module_routers(module)),
            app_module=module_name(app_file, subject='application file'),
            otel=otel,
            pin=('keep' if keep_release else 'latest') if exists else None,
        )
        prepared = PrepareUpdateResponse.model_validate(
            management_client.api_call(PrepareUpdateRequest(org=self._org, db=self._db, service_mutations=[mutation]))
        )
        generation = prepared.generations.service(name, base_path)
        if (generation > 0) != exists:
            raise excs.ConcurrencyError(
                excs.ErrorCode.CONCURRENT_MODIFICATION,
                f'Service {name!r} was {"created" if generation > 0 else "deleted"} in '
                f'{self.catalog_uri.uri_str} since this command read it; run the command again',
            )
        desired = next((m for m in prepared.services if (m.service_name, m.base_path) == (name, base_path)), None)
        if generation > 0 and desired is None:
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR,
                f'Pixeltable Cloud reported service {name!r} at generation {generation} without its desired spec',
            )
        if desired is not None:
            mutation = mutation.model_copy(
                update={
                    'workers': desired.workers,
                    'cpu': desired.cpu,
                    'memory_mb': desired.memory_mb,
                    'disk_gb': desired.disk_gb,
                    'description': desired.description,
                }
            )
        expected = ServiceGeneration(
            service_name=name,
            base_path=base_path,
            generation=generation if expected_generation is None else expected_generation,
        )
        request = SubmitUpdateRequest(
            org=self._org,
            db=self._db,
            service_mutations=[mutation],
            expected_generations=ExpectedGenerations(services=[expected]),
        )
        accepted = receipts.submit(self.catalog_uri, request)
        if wait:
            return self._settle(name, base_path, accepted)
        submitted = self.get(name, base_path)
        if submitted is None:
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR,
                f'Service {name!r} is not in {self.catalog_uri.uri_str} after its submission',
            )
        submitted.record = submitted.record.model_copy(
            update={'receipt': self._required(name, next(iter(accepted), None))}
        )
        return submitted

    def stop(self, instance: ServiceInstance) -> None:
        stopped = StopServiceInstanceResponse.model_validate(
            management_client.api_call(
                StopServiceInstanceRequest(
                    org=self._org, db=self._db, service_name=instance.service_name, base_path=instance.base_path
                )
            )
        )
        self._await([self._required(instance.service_name, stopped.receipt)])

    def restart(self, instance: ServiceInstance) -> None:
        """Restart instance's pods on their current release."""
        restarting = RestartServiceInstanceResponse.model_validate(
            management_client.api_call(
                RestartServiceInstanceRequest(
                    org=self._org, db=self._db, service_name=instance.service_name, base_path=instance.base_path
                )
            )
        )
        receipt = self._required(instance.service_name, restarting.receipt)
        self._settle(instance.service_name, instance.base_path, [receipt])

    def retry(self, instance: ServiceInstance) -> ServiceInstance | None:
        """Start a new attempt of instance's current generation, which must have failed.

        Returns the instance afterwards, or None if the generation was a deletion.
        """
        receipt = instance.record.receipt
        if receipt is None or not receipt.failed:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE,
                f'the current generation of service {instance.service_name!r} has not failed; nothing to retry',
            )
        settled = self._await([receipts.retry(self.catalog_uri, receipt)])
        return self._instance_after(instance.service_name, instance.base_path, settled)

    def delete(self, instance: ServiceInstance) -> None:
        deleted = DeleteServiceInstanceResponse.model_validate(
            management_client.api_call(
                DeleteServiceInstanceRequest(
                    org=self._org, db=self._db, service_name=instance.service_name, base_path=instance.base_path
                )
            )
        )
        self._await([self._required(instance.service_name, deleted.receipt)])

    def logs(
        self, instance: ServiceInstance, *, since_seconds: int, limit: int, include_health: bool
    ) -> Sequence[LogRecord]:
        response = GetLogsResponse.model_validate(
            management_client.api_call(
                GetLogsRequest(
                    org=self._org,
                    db=self._db,
                    service_name=instance.service_name,
                    base_path=instance.base_path,
                    since_seconds=since_seconds,
                    limit=limit,
                    include_health=include_health,
                )
            )
        )
        return response.records

    def _serves(self, record: ServiceInstanceRecord, base_path: str, recursive: bool) -> bool:
        if record.base_path == base_path:
            return True
        return recursive and (base_path == '' or record.base_path.startswith(f'{base_path}/'))

    def _required(self, name: str, receipt: GenerationReceipt | None) -> GenerationReceipt:
        if receipt is None:
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR,
                f'Pixeltable Cloud accepted a change to service {name!r} without a receipt',
            )
        return receipt

    def _await(self, accepted: Sequence[GenerationReceipt]) -> Sequence[GenerationReceipt]:
        return receipts.await_receipts(self.catalog_uri, accepted)

    def _settle(self, name: str, base_path: str, accepted: Sequence[GenerationReceipt]) -> ServiceInstance:
        """Wait on the receipts of a change to the named instance, and return the instance afterwards."""
        instance = self._instance_after(name, base_path, self._await(accepted))
        if instance is None:
            raise excs.InternalError(
                excs.ErrorCode.INTERNAL_ERROR, f'Service {name!r} is no longer in {self.catalog_uri.uri_str}'
            )
        return instance

    def _instance_after(
        self, name: str, base_path: str, settled: Sequence[GenerationReceipt]
    ) -> ServiceInstance | None:
        """The named instance after a settled change, carrying that change's receipt; None if it is gone.

        The listing may already carry a later generation's receipt, which is not this change's.
        """
        instance = self.get(name, base_path)
        if instance is None:
            return None
        if len(settled) == 1:
            instance.record = instance.record.model_copy(update={'receipt': settled[0]})
        if instance.state is ServiceState.AVAILABLE:
            self._wait_for_endpoint(instance)
        return instance

    def _wait_for_endpoint(self, instance: ServiceInstance) -> None:
        """Poll an available instance's endpoint until a request reaches the pod behind it.

        AVAILABLE says the pod is ready, not that the gateway routes to it: the instance replaces its pod
        in place, so there is a window where the route resolves to no ready pod and the gateway answers
        502. Any status the pod itself produced, 404 included, means the route is through.
        """
        endpoint = instance.record.endpoint
        if endpoint is None:
            return
        deadline = time.monotonic() + self._ENDPOINT_TIMEOUT
        while True:
            try:
                # /health needs no credential: the gateway authenticates every other path
                status = httpx.get(f'{endpoint}/health', timeout=self._ENDPOINT_PROBE_TIMEOUT).status_code
                if status not in (502, 503, 504):
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise excs.InternalError(
                    excs.ErrorCode.INTERNAL_ERROR,
                    f'Service {instance.service_name!r} is available, but {endpoint} did not answer within '
                    f'{self._ENDPOINT_TIMEOUT:.0f}s',
                )
            time.sleep(self._POLL_INTERVAL)
