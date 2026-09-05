"""Management API protocol: request/response models shared between the pxt SDK and the Pixeltable cloud server."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pixeltable.serving import ServiceInstanceRecord
from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli.types import DbArtifact, DbPlan, ServiceSpec


class ManagementOperationType(str, Enum):
    GET_DB = 'get_db'
    LIST_DBS = 'list_dbs'
    DELETE_DB = 'delete_db'

    CREATE_SERVICE_INSTANCE = 'create_service'
    GET_SERVICE_INSTANCE = 'get_service'
    LIST_SERVICE_INSTANCES = 'list_services'
    UPDATE_SERVICE_INSTANCE = 'update_service'
    REPORT_SERVICE_INSTANCE = 'report_service_instance'
    START_SERVICE_INSTANCE = 'start_service'
    STOP_SERVICE_INSTANCE = 'stop_service'
    DELETE_SERVICE_INSTANCE = 'delete_service'

    START_DB = 'start_db'
    STOP_DB = 'stop_db'
    UPDATE_DB = 'update_db'
    GET_ARCHIVE = 'get_archive'

    LIST_ORGS = 'list_orgs'

    SET_SECRET = 'set_secret'
    DELETE_SECRET = 'delete_secret'
    LIST_SECRETS = 'list_secrets'


# Db operations

# A hosted database name: lowercase letters, digits, and hyphens, starting and ending with a letter
# or digit, at most 29 characters. This is the `db` identifier that appears in pxt://org:db URIs.
_HOSTED_NAME_RE = re.compile(r'[a-z0-9]([a-z0-9-]*[a-z0-9])?')
_HOSTED_NAME_MAX_LEN = 29


def _validate_hosted_name(value: str, kind: str) -> str:
    if len(value) > _HOSTED_NAME_MAX_LEN:
        raise ValueError(f'{kind} must be at most {_HOSTED_NAME_MAX_LEN} characters (got {len(value)})')
    # fullmatch anchors both ends; match() + `$` would let a trailing newline
    # through ('main\n'), which corrupts the URI we build from this downstream.
    if not _HOSTED_NAME_RE.fullmatch(value):
        raise ValueError(
            f'{kind} {value!r} is invalid: use only lowercase letters, digits, and hyphens, '
            'starting and ending with a letter or digit.'
        )
    return value


class StoredSecret(BaseModel):
    """What the secret store holds under one key."""

    # an opaque digest of the value, which only the control plane produces
    digest: str

    # which binding the value came from, so that a rebinding is noticed
    binding: str = ''


class DatabaseSpec(BaseModel):
    """The resources provided by a database, in the widest sense (everything available in the runtime environment)."""

    fingerprint: ProjectFingerprint | None = None

    # the metadata schema version of the Pixeltable that packaged the archive, which only that
    # Pixeltable can report
    pxt_md_version: int = 0

    # the project's secret bindings: each key names the source of its value, never the value
    secrets: dict[str, str] = Field(default_factory=dict)

    # what the secret store holds. SET_SECRET changes it, not this field: the control plane fills it when
    # reporting a spec and ignores it in a request
    stored_secrets: dict[str, StoredSecret] = Field(default_factory=dict)

    # None: take default
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None
    workers: int | None = None
    default_bucket: str | None = None


class DatabaseStatus(BaseModel):
    """Information about the running system."""

    model_config = ConfigDict(extra='ignore')

    state: str = ''

    fingerprint: ProjectFingerprint | None = None

    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None
    workers: int | None = None

    # one entry per pod serving the database; workers is how many are running
    worker_status: list[dict[str, Any]] = Field(default_factory=list)

    # the pods' secret digests
    secret_digests: dict[str, str] = Field(default_factory=dict)

    last_build_outcome: str | None = None
    last_build_error: str | None = None

    # why the database is FAILED
    failure_reason: str | None = None


class DatabaseState(BaseModel):
    """The state of a hosted db: its current spec and what is actually running."""

    model_config = ConfigDict(extra='ignore')

    db: str = ''
    spec: DatabaseSpec = Field(default_factory=DatabaseSpec)
    status: DatabaseStatus = Field(default_factory=DatabaseStatus)


class GetDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_DB] = ManagementOperationType.GET_DB
    org: str | None = None
    db: str


class GetDbResponse(BaseModel):
    database: DatabaseState


class ListDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_DBS] = ManagementOperationType.LIST_DBS
    org: str | None = None


class UpdateDbRequest(BaseModel):
    """Request for changing the running system to match the spec."""

    operation_type: Literal[ManagementOperationType.UPDATE_DB] = ManagementOperationType.UPDATE_DB
    org: str | None = None
    db: str
    spec: DatabaseSpec

    # compute the plan without recording the spec, acting on it, or handing out an upload url
    dry_run: bool = False

    # build the image even when one is already built for the spec's image digest
    force_image_build: bool = False

    @field_validator('db')
    @classmethod
    def _validate_db_name(cls, value: str) -> str:
        return _validate_hosted_name(value, 'Database name')


class ArtifactUpload(BaseModel):
    """An artifact the spec names and the control plane does not hold, and where to put it."""

    artifact: DbArtifact
    url: str


class UpdateDbResponse(BaseModel):
    plan: DbPlan
    state: DatabaseState

    # if non-empty: perform the uploads first, then retry the request
    uploads: list[ArtifactUpload] = Field(default_factory=list)


class DeleteDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_DB] = ManagementOperationType.DELETE_DB
    org: str | None = None
    db: str


class StartDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.START_DB] = ManagementOperationType.START_DB
    org: str | None = None
    db: str


class StopDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.STOP_DB] = ManagementOperationType.STOP_DB
    org: str | None = None
    db: str


class GetArchiveRequest(BaseModel):
    """Ask for a url serving the database's current project archive; a pod sends this as it starts."""

    operation_type: Literal[ManagementOperationType.GET_ARCHIVE] = ManagementOperationType.GET_ARCHIVE
    org: str | None = None
    db: str


class GetArchiveResponse(BaseModel):
    presigned_url: str
    # ProjectFingerprint.archive_digest() of the archive the url serves
    digest: str


# Secrets


class SetSecretRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.SET_SECRET] = ManagementOperationType.SET_SECRET
    org: str
    db: str | None = None
    key: str
    value: str


class SetSecretResponse(BaseModel):
    key: str


class DeleteSecretRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_SECRET] = ManagementOperationType.DELETE_SECRET
    org: str
    db: str | None = None
    key: str


class DeleteSecretResponse(BaseModel):
    key: str


class ListSecretsRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_SECRETS] = ManagementOperationType.LIST_SECRETS
    org: str
    db: str | None = None


class ListSecretsResponse(BaseModel):
    keys: list[str]


# Services


class CreateServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.CREATE_SERVICE_INSTANCE] = (
        ManagementOperationType.CREATE_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''  # the path within the database (excludes the catalog uri)
    spec: ServiceSpec
    app_module: str
    otel: bool = False
    workers: int = 1
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10
    description: str | None = None


class CreateServiceInstanceResponse(BaseModel):
    instance: ServiceInstanceRecord


class GetServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_SERVICE_INSTANCE] = ManagementOperationType.GET_SERVICE_INSTANCE
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''


class GetServiceInstanceResponse(BaseModel):
    instance: ServiceInstanceRecord


class ListServiceInstancesRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_SERVICE_INSTANCES] = (
        ManagementOperationType.LIST_SERVICE_INSTANCES
    )
    org: str | None = None
    db: str


class ListServiceInstancesResponse(BaseModel):
    instances: list[ServiceInstanceRecord]


class UpdateServiceInstanceRequest(BaseModel):
    """An omitted field is left as it is."""

    operation_type: Literal[ManagementOperationType.UPDATE_SERVICE_INSTANCE] = (
        ManagementOperationType.UPDATE_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''
    spec: ServiceSpec | None = None
    app_module: str | None = None
    otel: bool | None = None
    workers: int | None = None
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None
    description: str | None = None


class UpdateServiceInstanceResponse(BaseModel):
    instance: ServiceInstanceRecord


class ReportServiceInstanceRequest(BaseModel):
    """Record the project an instance loaded; a service pod sends this as it starts serving.

    Unlike UPDATE_SERVICE_INSTANCE this changes nothing the instance is asked to serve, so it must not
    restart the pod that sends it.
    """

    operation_type: Literal[ManagementOperationType.REPORT_SERVICE_INSTANCE] = (
        ManagementOperationType.REPORT_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''
    fingerprint: ProjectFingerprint


class StartServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.START_SERVICE_INSTANCE] = (
        ManagementOperationType.START_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''


class StartServiceInstanceResponse(BaseModel):
    instance: ServiceInstanceRecord


class StopServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.STOP_SERVICE_INSTANCE] = (
        ManagementOperationType.STOP_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''


class StopServiceInstanceResponse(BaseModel):
    instance: ServiceInstanceRecord


class DeleteServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_SERVICE_INSTANCE] = (
        ManagementOperationType.DELETE_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''


class DeleteServiceInstanceResponse(BaseModel):
    service_name: str


# Orgs


class OrgRecord(BaseModel):
    org_id: str
    org: str
    default_db: str | None = None
    created_at: float
    updated_at: float


class ListOrgsRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_ORGS] = ManagementOperationType.LIST_ORGS


class ListOrgsResponse(BaseModel):
    orgs: list[OrgRecord]
