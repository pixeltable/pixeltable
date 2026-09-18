"""The durable shape of a hosted database's metadata.

The control plane stores these and the management protocol sends them, so the wire carries what is stored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli.types import DbState


class DatabaseResources(BaseModel):
    """The resources of a database, in the widest sense (everything available in the runtime environment)."""

    # TODO: make that fingerprint accessible instead and record it here
    fingerprint: ProjectFingerprint | None = Field(
        default=None, description='the deployed project; null for the base image of a new org'
    )

    pxt_md_version: int = Field(
        default=0, description='metadata schema version of the Pixeltable that packaged the archive'
    )
    default_bucket: str | None = None

    cpu: float | None = Field(default=None, description='cores; null takes the default for the tier')
    memory_mb: int | None = Field(default=None, description='null takes the default for the tier')
    disk_gb: int | None = Field(default=None, description='null takes the default for the tier')
    workers: int | None = Field(default=None, description='pods serving the database; null takes the default')

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

    last_build_outcome: str | None = Field(
        default=None, description="the last image build's outcome; null when no build has run"
    )
    last_build_error: str | None = None
    failure_reason: str | None = Field(default=None, description='why the state is FAILED')
