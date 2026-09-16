from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from pixeltable.service.service_md import ServiceInstanceRecord
from pixeltable_cli.types import ServiceSpec, ServiceState

if TYPE_CHECKING:
    from pixeltable.service.management_protocol import LogRecord

    from .service_manager import ServiceManagerBase


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
    def state(self) -> ServiceState:
        """The state its manager last reported."""
        return self.record.state

    def stop(self) -> None:
        """Stop serving, leaving this instance startable again."""
        self._manager.stop(self)

    def restart(self) -> None:
        """Cycle the process or pods serving this instance onto what they already run."""
        self._manager.restart(self)

    def delete(self) -> None:
        """Stop serving and forget this instance."""
        self._manager.delete(self)

    def logs(self, *, since_seconds: int, limit: int, include_health: bool) -> Sequence[LogRecord]:
        return self._manager.logs(self, since_seconds=since_seconds, limit=limit, include_health=include_health)
