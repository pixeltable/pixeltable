from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pydantic

from ._spec import ServiceSpec

if TYPE_CHECKING:
    from .service_manager import ServiceManagerBase


# str-valued: the plan and listing types the daemon validates declare state as str
class ServiceInstanceState(str, Enum):
    DEPLOYING = 'DEPLOYING'
    AVAILABLE = 'AVAILABLE'
    STOPPED = 'STOPPED'
    UPDATING = 'UPDATING'
    FAILED = 'FAILED'


class ServiceInstanceRecord(pydantic.BaseModel):
    """Metadata of a service instance."""

    model_config = pydantic.ConfigDict(extra='ignore')

    service_name: str

    # the path within the instance's catalog (excludes catalog uri)
    base_path: str

    endpoint: str

    # the file path
    app_file: str

    spec: ServiceSpec

    # whether the instance emits OpenTelemetry traces
    otel: bool = False

    # a record read from disk takes the default, since the process table decides whether its instance serves
    state: ServiceInstanceState = ServiceInstanceState.AVAILABLE

    # the process serving the instance; set only for an instance running on this machine
    pid: int | None = None

    # creation time of pid, None where the platform does not report one
    process_started_at: float | None = None


class ServiceInstance:
    """A running instance of a service.

    The router executes the routes; this reports where it runs and what it serves, and takes it down. The
    manager that produced it carries out both, so a local and a hosted instance are the same class.
    """

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
    def app_file(self) -> str:
        return self.record.app_file

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
        """Stop serving and forget this instance."""
        self._manager.stop(self)
