"""Management API protocol: request/response models shared between the pxt SDK and the Pixeltable cloud server."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pixeltable.serving import ServiceInstanceRecord
from pixeltable.utils.project import DepsType, ProjectFingerprint
from pixeltable_cli.types import ServiceSpec


class ManagementOperationType(str, Enum):
    CREATE_DB = 'create_db'
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
    BUILD_IMAGE = 'build_image'
    SET_ARCHIVE = 'set_archive'
    GET_ARCHIVE = 'get_archive'
    GET_ARCHIVE_UPLOAD_URL = 'get_archive_upload_url'

    LIST_ORGS = 'list_orgs'

    SET_SECRET = 'set_secret'
    DELETE_SECRET = 'delete_secret'
    LIST_SECRETS = 'list_secrets'

    CREATE_API_KEY = 'create_api_key'
    GET_API_KEY = 'get_api_key'
    LIST_API_KEYS = 'list_api_keys'
    DELETE_API_KEY = 'delete_api_key'

    CREATE_RUNTIME_KEY = 'create_runtime_key'
    LIST_RUNTIME_KEYS = 'list_runtime_keys'
    UPDATE_RUNTIME_KEY = 'update_runtime_key'
    DELETE_RUNTIME_KEY = 'delete_runtime_key'


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


class CreateDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.CREATE_DB] = ManagementOperationType.CREATE_DB
    org: str | None = None
    db: str
    db_name: str | None = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10
    workers: int = 1

    @field_validator('db')
    @classmethod
    def _validate_db_name(cls, value: str) -> str:
        return _validate_hosted_name(value, 'Database name')


class GetDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_DB] = ManagementOperationType.GET_DB
    org: str | None = None
    db: str


class ListDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_DBS] = ManagementOperationType.LIST_DBS
    org: str | None = None


class UpdateDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.UPDATE_DB] = ManagementOperationType.UPDATE_DB
    org: str | None = None
    db: str
    db_name: str | None = None
    default_bucket: str | None = None
    workers: int | None = None
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None


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


class BuildImageRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.BUILD_IMAGE] = ManagementOperationType.BUILD_IMAGE
    org: str | None = None
    db: str
    archive_key: str
    # ProjectFingerprint.image_digest()
    image_digest: str
    python_version: str
    system_dependencies: list[str] = []
    # how the build installs the project's packages, and the options it passes to uv sync
    deps_type: DepsType
    uv_options: str | None = None
    # the metadata schema version of the Pixeltable that packaged the archive
    pxt_md_version: int


class SetArchiveRequest(BaseModel):
    """Point the database's pods at a stored project archive, restarting them to fetch it."""

    operation_type: Literal[ManagementOperationType.SET_ARCHIVE] = ManagementOperationType.SET_ARCHIVE
    org: str | None = None
    db: str
    archive_key: str
    # what the archive holds and the environment it runs in; GET_DB reports it back, and a diff compares
    # the project here against it
    fingerprint: ProjectFingerprint


class GetArchiveRequest(BaseModel):
    """Ask for a url serving the database's current project archive; a pod sends this as it starts."""

    operation_type: Literal[ManagementOperationType.GET_ARCHIVE] = ManagementOperationType.GET_ARCHIVE
    org: str | None = None
    db: str


class GetArchiveResponse(BaseModel):
    presigned_url: str
    archive_key: str
    # ProjectFingerprint.archive_digest() of the archive the url serves
    digest: str


class GetArchiveUploadUrlRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_ARCHIVE_UPLOAD_URL] = (
        ManagementOperationType.GET_ARCHIVE_UPLOAD_URL
    )
    org: str | None = None
    db: str
    # ProjectFingerprint.archive_digest(); the control plane keys the stored archive by it
    digest: str


class GetArchiveUploadUrlResponse(BaseModel):
    archive_key: str
    # None when the digest names a stored archive: nothing left to upload
    presigned_url: str | None = None


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


# API keys
#
# A person's own key, keyed by user: it acts as whoever created it and carries no grants, which is
# the whole difference from a runtime key below.


class CreateApiKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.CREATE_API_KEY] = ManagementOperationType.CREATE_API_KEY
    name: str


class CreateApiKeyResponse(BaseModel):
    name: str
    api_key: str | None  # List/Get api key will not use this field since api keys follow show-once pattern.
    created_at: datetime


class GetApiKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_API_KEY] = ManagementOperationType.GET_API_KEY
    name: str


class GetApiKeyResponse(BaseModel):
    api_key: CreateApiKeyResponse


class ListApiKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_API_KEYS] = ManagementOperationType.LIST_API_KEYS


class ListApiKeyResponse(BaseModel):
    api_keys: list[CreateApiKeyResponse]


class DeleteApiKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_API_KEY] = ManagementOperationType.DELETE_API_KEY
    name: str


class DeleteApiKeyResponse(BaseModel):
    name: str


# Runtime keys
#
# A runtime key is an API key held by something that is not a person — an agent, a job, another
# service. It belongs to an organization, like the WorkOS key it is, and reaches only what it is
# granted.
#
# A grant is a verb on a pxt:// resource, never a bare resource: naming a service does not say
# whether the holder may call it or reconfigure it, and those are not the same permission. The split
# follows the one every IAM makes — Cloud Run's run.invoker vs run.admin.
#
#     access:pxt://org:db                     reach the data: every service in the database, and its storage
#     access:pxt://org:db/services            call any service in the database
#     access:pxt://org:db/services/name       call that service: every route under it
#     manage:pxt://org:db/services            create, list, and manage any service in the database
#     manage:pxt://org:db/services/name       start, stop, update, delete that service
#
# `access` and `manage` are independent: neither implies the other, so a key that can call a service
# cannot reconfigure it, and one that can stop it cannot read what flows through it.
#
# The org segment must be the organization the caller's own key belongs to. It is not how the control
# plane decides whose keys these are — the credential settles that — so a grant naming another
# organization is refused rather than quietly reinterpreted.
#
# No request here names an organization on its own, for the same reason: a field for it would
# suggest a caller could act on another one, which no credential permits.


class RuntimeKeyRecord(BaseModel):
    name: str
    grants: list[str]
    created_at: datetime
    # Set only in a create response: the secret is shown once and never stored in retrievable form.
    api_key: str | None = None


class CreateRuntimeKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.CREATE_RUNTIME_KEY] = ManagementOperationType.CREATE_RUNTIME_KEY
    name: str
    # A key granted nothing could never be used, so an empty list is a mistake rather than a default.
    grants: list[str] = Field(min_length=1)


class CreateRuntimeKeyResponse(BaseModel):
    runtime_key: RuntimeKeyRecord


class ListRuntimeKeysRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_RUNTIME_KEYS] = ManagementOperationType.LIST_RUNTIME_KEYS


class ListRuntimeKeysResponse(BaseModel):
    runtime_keys: list[RuntimeKeyRecord]


class UpdateRuntimeKeyRequest(BaseModel):
    """Add and remove grants on an existing key, leaving the rest alone.

    A delta rather than a replacement list, so granting one more resource does not depend on the
    caller first knowing — and faithfully resending — everything the key already had.
    """

    operation_type: Literal[ManagementOperationType.UPDATE_RUNTIME_KEY] = ManagementOperationType.UPDATE_RUNTIME_KEY
    name: str
    allow: list[str] = Field(default_factory=list)
    revoke: list[str] = Field(default_factory=list)


class UpdateRuntimeKeyResponse(BaseModel):
    runtime_key: RuntimeKeyRecord


class DeleteRuntimeKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_RUNTIME_KEY] = ManagementOperationType.DELETE_RUNTIME_KEY
    name: str


class DeleteRuntimeKeyResponse(BaseModel):
    name: str
