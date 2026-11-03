"""The durable shape of a hosted database's metadata.

The control plane stores these and the management protocol sends them, so the wire carries what is stored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pixeltable.service.resources import DatabaseCapacity
from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli.types import DbState


class DatabaseResources(DatabaseCapacity):
    """A hosted database's deployment: its project and the size and number of its pods."""

    # TODO: make that fingerprint accessible instead and record it here
    fingerprint: ProjectFingerprint | None = Field(
        default=None, description='the deployed project; null for the base image of a new org'
    )

    pxt_md_version: int = Field(
        default=0, description='metadata schema version of the Pixeltable that packaged the archive'
    )
    default_bucket: str | None = None


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
    md_version: int | None = Field(
        default=None, description='metadata version of the running database; null if unknown'
    )
