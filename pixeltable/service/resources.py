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


class DatabaseCapacity(PodResources):
    disk_gb: int | None = pydantic.Field(
        default=None, gt=0, description="size of each pod's PIXELTABLE_HOME volume, in GB"
    )

    def capacity(self) -> DatabaseCapacity:
        return DatabaseCapacity(**self.model_dump(include=set(DatabaseCapacity.model_fields)))
