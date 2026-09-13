"""The durable shape of a service instance's metadata.

The control plane stores these and the management protocol sends them, so the wire carries what is stored.
"""

from __future__ import annotations

import pydantic

from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli import types
from pixeltable_cli.types import ServiceSpec, ServiceState
from pixeltable_cli.utils import PxtPath


class ServiceInstanceRecord(pydantic.BaseModel):
    """Metadata of a service instance."""

    model_config = pydantic.ConfigDict(extra='ignore')

    service_name: str

    # the path within the instance's catalog (excludes catalog uri)
    base_path: str

    endpoint: str

    # the loopback port, kept across a restart so callers keep their address;
    # None for a hosted instance, which is reached at its own hostname
    port: int | None = None

    # the app file's module path, relative to the project root
    app_module: str

    spec: ServiceSpec

    # whether the instance emits OpenTelemetry traces
    otel: bool = False

    state: ServiceState = ServiceState.AVAILABLE

    created_at: float | None = None

    # the reason for a FAILED state
    error: str | None = None

    # how many workers serve the instance; a local instance is always one process
    workers: int | None = None

    # the process serving the instance; set only for an instance running on this machine
    pid: int | None = None

    # creation time of pid, None where the platform does not report one
    process_started_at: float | None = None

    # the project fingerprint
    fingerprint: ProjectFingerprint

    def to_cli_instance(self, catalog_uri: str = '') -> types.ServiceInstance:
        return types.ServiceInstance(
            name=self.service_name,
            catalog_path=PxtPath('/'.join(part for part in (catalog_uri, self.base_path) if part != '')),
            endpoint=self.endpoint,
            port=self.port,
            state=self.state,
            error=self.error,
            app_module=self.app_module,
            spec=self.spec,
            pid=self.pid,
            process_started_at=self.process_started_at,
        )
