"""The durable shape of a hosted database's metadata.

The control plane stores these and the management protocol sends them, so the wire carries what is stored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli.types import DbState


class DatabaseResources(BaseModel):
    """The resources of a database, in the widest sense (everything available in the runtime environment)."""

    # None: the "base image" of the "default" database, created when the org was created
    # TODO: make that fingerprint accessible instead and record it here
    fingerprint: ProjectFingerprint | None = None

    # the metadata schema version of the Pixeltable that packaged the archive, which only that
    # Pixeltable can report
    pxt_md_version: int = 0
    default_bucket: str | None = None

    # None: take default
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None
    workers: int | None = None

    def capacity(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        if self.cpu is not None:
            result['cpu'] = self.cpu
        if self.memory_mb is not None:
            result['memory_mb'] = self.memory_mb
        if self.disk_gb is not None:
            result['disk_gb'] = self.disk_gb
        if self.workers is not None:
            result['workers'] = self.workers
        return result


class DatabaseStatus(BaseModel):
    """The control plane's view of a running database."""

    model_config = ConfigDict(extra='ignore')

    resources: DatabaseResources
    state: DbState

    last_build_outcome: str | None = None
    last_build_error: str | None = None

    # why the state is FAILED
    failure_reason: str | None = None
