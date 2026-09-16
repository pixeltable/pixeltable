"""The durable shape of a service instance's metadata.

The control plane stores these and the management protocol sends them, so the wire carries what is stored.
"""

from __future__ import annotations

import pydantic

from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli import types
from pixeltable_cli.types import ServiceSpec, ServiceState
from pixeltable_cli.utils import PxtPath


class ServiceResources(pydantic.BaseModel):
    """What one of a service instance's pods gets, and how many of them serve it."""

    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10
    workers_min: int = 1


class ServiceInstanceRecord(pydantic.BaseModel):
    """Metadata of a service instance."""

    model_config = pydantic.ConfigDict(extra='ignore')

    service_name: str

    # the path within the instance's catalog (excludes catalog uri)
    base_path: str

    # where the instance answers; derived on both sides, so a stored one is never read back
    endpoint: str = ''

    # the app file's module path, relative to the project root
    app_module: str

    spec: ServiceSpec

    # whether the instance emits OpenTelemetry traces
    otel: bool = False

    state: ServiceState = ServiceState.AVAILABLE

    # set once, when the record is created
    created_at: float | None = None

    # set on every update of the record
    updated_at: float | None = None

    # the reason for a FAILED state
    error: str | None = None

    # the project the instance serves; None until its pod reports one
    fingerprint: ProjectFingerprint | None = None

    # whether its database has moved past the project this instance serves; derived on read, so a stored
    # one is never read back. A restart moves the instance onto the new project.
    update_pending: bool = False

    resources: ServiceResources = pydantic.Field(default_factory=ServiceResources)
    description: str | None = None

    def to_cli_instance(self, catalog_uri: str = '') -> types.ServiceInstance:
        return types.ServiceInstance(
            name=self.service_name,
            catalog_path=PxtPath('/'.join(part for part in (catalog_uri, self.base_path) if part != '')),
            endpoint=self.endpoint,
            port=None,
            state=self.state,
            error=self.error,
            app_module=self.app_module,
            spec=self.spec,
            pid=None,
            process_started_at=None,
            update_pending=self.update_pending,
        )


class LocalServiceInstanceRecord(ServiceInstanceRecord):
    """A service instance served by a process on this machine."""

    # the loopback port, kept across a restart so callers keep their address
    port: int | None = None

    # the process serving the instance
    pid: int | None = None

    # creation time of pid, None where the platform does not report one
    process_started_at: float | None = None

    def to_cli_instance(self, catalog_uri: str = '') -> types.ServiceInstance:
        instance = super().to_cli_instance(catalog_uri)
        return instance.model_copy(
            update={'port': self.port, 'pid': self.pid, 'process_started_at': self.process_started_at}
        )
