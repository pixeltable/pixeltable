"""The resources of hosted databases and services, shared by the project config, the requests and the stored records."""

from __future__ import annotations

import pydantic
from typing_extensions import Self


class PodResources(pydantic.BaseModel):
    cpu: float | None = pydantic.Field(default=None, gt=0, description='vCPUs per pod')
    memory_mb: int | None = pydantic.Field(default=None, gt=0, description='memory per pod, in MB')
    workers: int | None = pydantic.Field(default=None, gt=0, description='number of pods')

    def changes(self, other: Self) -> Self:
        """Return the fields set in other whose values differ from these."""
        return type(self)(
            **{
                field: value
                for field, value in other.model_dump(exclude_none=True).items()
                if value != getattr(self, field)
            }
        )


class ServiceResources(PodResources):
    file_cache_gb: float | None = pydantic.Field(default=None, gt=0, description='file cache size per pod, in GB')


# Modal's proxy loses the response of a call that runs past a limit between 310 and 400 seconds
MAX_UDF_CALL_SECONDS = 300


class UdfResources(pydantic.BaseModel):
    """Resources of the Modal containers that run one UDF's calls."""

    model_config = pydantic.ConfigDict(extra='forbid', frozen=True)

    cpu: float | None = pydantic.Field(default=None, gt=0, description='vCPUs per container')
    memory_mb: int | None = pydantic.Field(default=None, gt=0, description='memory per container, in MB')
    gpu: str | None = pydantic.Field(default=None, min_length=1, description="Modal GPU spec, eg 'T4' or 'H100:2'")
    timeout_seconds: int = pydantic.Field(
        default=MAX_UDF_CALL_SECONDS, gt=0, le=MAX_UDF_CALL_SECONDS, description='time limit of one call'
    )


class DatabaseCapacity(PodResources):
    disk_gb: int | None = pydantic.Field(
        default=None, gt=0, description="size of each pod's PIXELTABLE_HOME volume, in GB"
    )

    def capacity(self) -> DatabaseCapacity:
        return DatabaseCapacity(**self.model_dump(include=set(DatabaseCapacity.model_fields)))
