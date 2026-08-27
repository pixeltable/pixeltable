"""Management API protocol: request/response models shared between the pxt SDK and the Pixeltable cloud server."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator

from pixeltable.serving import ServiceSpec


class ServiceOperationType(str, Enum):
    CREATE_DB = 'create_db'
    GET_DB = 'get_db'
    LIST_DBS = 'list_dbs'
    DELETE_DB = 'delete_db'

    CREATE_SERVICE = 'create_service'
    GET_SERVICE = 'get_service'
    LIST_SERVICES = 'list_services'
    UPDATE_SERVICE = 'update_service'
    START_SERVICE = 'start_service'
    STOP_SERVICE = 'stop_service'
    DELETE_SERVICE = 'delete_service'
    LIST_SERVICE_RUNS = 'list_service_runs'
    GET_SERVICE_RUN = 'get_service_run'

    START_DB = 'start_db'
    STOP_DB = 'stop_db'
    UPDATE_DB = 'update_db'
    UPDATE_RUNTIME = 'update_runtime'
    GET_BUNDLE_UPLOAD_URL = 'get_bundle_upload_url'

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
    operation_type: Literal[ServiceOperationType.CREATE_DB] = ServiceOperationType.CREATE_DB
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
    operation_type: Literal[ServiceOperationType.GET_DB] = ServiceOperationType.GET_DB
    org: str | None = None
    db: str


class ListDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_DBS] = ServiceOperationType.LIST_DBS
    org: str | None = None


class UpdateDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_DB] = ServiceOperationType.UPDATE_DB
    org: str | None = None
    db: str
    db_name: str | None = None
    default_bucket: str | None = None
    workers: int | None = None
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None


class DeleteDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_DB] = ServiceOperationType.DELETE_DB
    org: str | None = None
    db: str


class StartDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.START_DB] = ServiceOperationType.START_DB
    org: str | None = None
    db: str


class StopDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.STOP_DB] = ServiceOperationType.STOP_DB
    org: str | None = None
    db: str


class UpdateRuntimeRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_RUNTIME] = ServiceOperationType.UPDATE_RUNTIME
    org: str | None = None
    db: str
    bundle_s3_key: str


class GetBundleUploadUrlRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_BUNDLE_UPLOAD_URL] = ServiceOperationType.GET_BUNDLE_UPLOAD_URL
    org: str | None = None
    db: str


class GetBundleUploadUrlResponse(BaseModel):
    presigned_url: str
    bundle_s3_key: str


# Secrets


class SetSecretRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.SET_SECRET] = ServiceOperationType.SET_SECRET
    org: str
    db: str | None = None
    key: str
    value: str


class SetSecretResponse(BaseModel):
    key: str


class DeleteSecretRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_SECRET] = ServiceOperationType.DELETE_SECRET
    org: str
    db: str | None = None
    key: str


class DeleteSecretResponse(BaseModel):
    key: str


class ListSecretsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SECRETS] = ServiceOperationType.LIST_SECRETS
    org: str
    db: str | None = None


class ListSecretsResponse(BaseModel):
    keys: list[str]


# Services


class ServiceRecord(BaseModel):
    service_id: str
    org_id: str
    db_id: str
    service_name: str
    base_path: str = ''
    workers_min: int = 1
    workers_max: int = 1
    state: str  # DEPLOYING | AVAILABLE | STOPPED | UPDATING | FAILED
    endpoint: str | None = None
    error: str | None = None
    created_at: float
    service_spec: str | None = None  # JSON-serialized ServiceSpec from latest run


class CreateServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.CREATE_SERVICE] = ServiceOperationType.CREATE_SERVICE
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''
    workers_min: int = 1
    description: str | None = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10
    service_spec: ServiceSpec | None = None

    @field_validator('service_spec', mode='before')
    @classmethod
    def _parse_service_spec(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v)
        return v


class CreateServiceResponse(BaseModel):
    service: ServiceRecord


class GetServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_SERVICE] = ServiceOperationType.GET_SERVICE
    org: str | None = None
    db: str
    service_name: str


class GetServiceResponse(BaseModel):
    service: ServiceRecord


class ListServicesRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SERVICES] = ServiceOperationType.LIST_SERVICES
    org: str | None = None
    db: str


class ListServicesResponse(BaseModel):
    services: list[ServiceRecord]


class StartServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.START_SERVICE] = ServiceOperationType.START_SERVICE
    org: str | None = None
    db: str
    service_name: str


class StartServiceResponse(BaseModel):
    service: ServiceRecord


class StopServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.STOP_SERVICE] = ServiceOperationType.STOP_SERVICE
    org: str | None = None
    db: str
    service_name: str


class StopServiceResponse(BaseModel):
    service: ServiceRecord


class UpdateServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_SERVICE] = ServiceOperationType.UPDATE_SERVICE
    org: str | None = None
    db: str
    service_name: str
    workers_min: int | None = None
    description: str | None = None
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None
    service_spec: ServiceSpec | None = None

    @field_validator('service_spec', mode='before')
    @classmethod
    def _parse_service_spec(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v)
        return v


class UpdateServiceResponse(BaseModel):
    service: ServiceRecord


class DeleteServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_SERVICE] = ServiceOperationType.DELETE_SERVICE
    org: str | None = None
    db: str
    service_name: str


class DeleteServiceResponse(BaseModel):
    service_name: str


class ListServiceRunsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SERVICE_RUNS] = ServiceOperationType.LIST_SERVICE_RUNS
    org: str | None = None
    db: str
    service_name: str


class GetServiceRunRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_SERVICE_RUN] = ServiceOperationType.GET_SERVICE_RUN
    org: str | None = None
    db: str
    service_name: str
    run_id: str


class ServiceRunRecord(BaseModel):
    run_id: str
    workers_min: int
    state: str  # AVAILABLE | STOPPED | FAILED
    started_at: float
    stopped_at: float | None = None
    runtime_build_id: str | None = None
    bundle_r2_path: str | None = None
    service_spec: str | None = None  # JSON-serialized ServiceSpec for this run
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10


class ListServiceRunsResponse(BaseModel):
    runs: list[ServiceRunRecord]


class GetServiceRunResponse(BaseModel):
    run: ServiceRunRecord


# Orgs


class OrgRecord(BaseModel):
    org_id: str
    org: str
    default_db: str | None = None
    created_at: float
    updated_at: float


class ListOrgsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_ORGS] = ServiceOperationType.LIST_ORGS


class ListOrgsResponse(BaseModel):
    orgs: list[OrgRecord]
