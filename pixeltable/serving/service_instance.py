from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import pydantic

from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli import types
from pixeltable_cli.types import ServiceSpec
from pixeltable_cli.utils import PxtPath

if TYPE_CHECKING:
    from .service_manager import ServiceManagerBase


class ServiceInstanceState(StrEnum):
    STARTING = 'STARTING'
    AVAILABLE = 'AVAILABLE'
    STOPPED = 'STOPPED'
    FAILED = 'FAILED'


class ServiceInstanceRecord(pydantic.BaseModel):
    """Metadata of a service instance."""

    model_config = pydantic.ConfigDict(extra='ignore')

    service_name: str

    # the path within the instance's catalog (excludes catalog uri)
    base_path: str

    endpoint: str

    # the loopback port the instance serves on, kept across a restart so callers keep their address;
    # None for a hosted instance, which is reached at its own hostname
    port: int | None = None

    # the app file's module path, relative to the project root
    app_module: str

    spec: ServiceSpec

    # whether the instance emits OpenTelemetry traces
    otel: bool = False

    state: ServiceInstanceState = ServiceInstanceState.AVAILABLE

    created_at: float | None = None

    # the reason for a FAILED state
    error: str | None = None

    # how many workers serve the instance; a local instance is always one process
    workers: int | None = None

    # the process serving the instance; set only for an instance running on this machine
    pid: int | None = None

    # creation time of pid, None where the platform does not report one
    process_started_at: float | None = None

    # the project files the instance is running, as its process fingerprinted them at start; None until
    # the instance's process reports them
    fingerprint: ProjectFingerprint | None = None

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


class ServiceInstance:
    """A running instance of a service."""

    record: ServiceInstanceRecord
    _manager: ServiceManagerBase

    def __init__(self, record: ServiceInstanceRecord, manager: ServiceManagerBase) -> None:
        self.record = record
        self._manager = manager

    @property
    def service_name(self) -> str:
        return self.record.service_name

    @property
    def base_path(self) -> str:
        return self.record.base_path

    @property
    def endpoint(self) -> str:
        return self.record.endpoint

    @property
    def app_module(self) -> str:
        return self.record.app_module

    @property
    def spec(self) -> ServiceSpec:
        return self.record.spec

    @property
    def otel(self) -> bool:
        return self.record.otel

    @property
    def state(self) -> ServiceInstanceState:
        """The state its manager last reported."""
        return self.record.state

    def stop(self) -> None:
        """Stop serving, leaving this instance startable again."""
        self._manager.stop(self)

    def delete(self) -> None:
        """Stop serving and forget this instance."""
        self._manager.delete(self)
