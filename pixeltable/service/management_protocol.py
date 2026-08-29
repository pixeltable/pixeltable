"""Management API protocol: request/response models shared between the pxt SDK and the Pixeltable cloud server."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator

from pixeltable.serving import ServiceInstanceRecord
from pixeltable_cli.types import ServiceSpec


class ManagementOperationType(str, Enum):
    CREATE_DB = 'create_db'
    GET_DB = 'get_db'
    LIST_DBS = 'list_dbs'
    DELETE_DB = 'delete_db'

    CREATE_SERVICE_INSTANCE = 'create_service_instance'
    GET_SERVICE_INSTANCE = 'get_service_instance'
    LIST_SERVICE_INSTANCES = 'list_service_instances'
    UPDATE_SERVICE_INSTANCE = 'update_service_instance'
    START_SERVICE_INSTANCE = 'start_service_instance'
    STOP_SERVICE_INSTANCE = 'stop_service_instance'
    DELETE_SERVICE_INSTANCE = 'delete_service_instance'

    START_DB = 'start_db'
    STOP_DB = 'stop_db'
    UPDATE_DB = 'update_db'
    BUILD_IMAGE = 'build_image'
    SET_PROJECT = 'set_project'
    GET_PROJECT = 'get_project'
    GET_PROJECT_UPLOAD_URL = 'get_project_upload_url'

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


class CreateDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.CREATE_DB] = ManagementOperationType.CREATE_DB
    org: str | None = None
    db: str
    db_name: str | None = None
    location: str | None = None
    region: str | None = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10

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
    """Build the image the database's pods run on.

    The image holds the environment and no project code, so image_digest identifies it and any database
    declaring that environment can run it. The build reads the lockfile out of the archive project_key
    names, and nothing else out of it.
    """

    operation_type: Literal[ManagementOperationType.BUILD_IMAGE] = ManagementOperationType.BUILD_IMAGE
    org: str | None = None
    db: str
    project_key: str
    # ProjectFingerprint.image_digest()
    image_digest: str
    python_version: str
    system_dependencies: list[str] = []
    # the metadata schema version of the Pixeltable that packaged the archive
    pxt_md_version: int


class SetProjectRequest(BaseModel):
    """Point the database's pods at a stored project archive, restarting them to fetch it."""

    operation_type: Literal[ManagementOperationType.SET_PROJECT] = ManagementOperationType.SET_PROJECT
    org: str | None = None
    db: str
    project_key: str


class GetProjectRequest(BaseModel):
    """Ask for a url serving the database's current project archive; a pod sends this as it starts."""

    operation_type: Literal[ManagementOperationType.GET_PROJECT] = ManagementOperationType.GET_PROJECT
    org: str | None = None
    db: str


class GetProjectResponse(BaseModel):
    presigned_url: str
    project_key: str
    # ProjectFingerprint.archive_digest() of the archive the url serves
    digest: str


class GetProjectUploadUrlRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_PROJECT_UPLOAD_URL] = (
        ManagementOperationType.GET_PROJECT_UPLOAD_URL
    )
    org: str | None = None
    db: str
    # ProjectFingerprint.archive_digest(); the control plane keys the stored archive by it
    digest: str


class GetProjectUploadUrlResponse(BaseModel):
    project_key: str
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


class StartServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.START_SERVICE_INSTANCE] = (
        ManagementOperationType.START_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str


class StartServiceInstanceResponse(BaseModel):
    instance: ServiceInstanceRecord


class StopServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.STOP_SERVICE_INSTANCE] = (
        ManagementOperationType.STOP_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str


class StopServiceInstanceResponse(BaseModel):
    instance: ServiceInstanceRecord


class DeleteServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_SERVICE_INSTANCE] = (
        ManagementOperationType.DELETE_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str


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
