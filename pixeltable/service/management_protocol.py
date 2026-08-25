"""Management API protocol: request/response models shared between the pxt SDK and the Pixeltable cloud server."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from pixeltable.config import ServiceConfig


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
    org: Optional[str] = None
    db: str
    db_name: Optional[str] = None
    location: Optional[str] = None
    region: Optional[str] = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10

    @field_validator('db')
    @classmethod
    def _validate_db_name(cls, value: str) -> str:
        return _validate_hosted_name(value, 'Database name')


class GetDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_DB] = ServiceOperationType.GET_DB
    org: Optional[str] = None
    db: str


class ListDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_DBS] = ServiceOperationType.LIST_DBS
    org: Optional[str] = None


class UpdateDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_DB] = ServiceOperationType.UPDATE_DB
    org: Optional[str] = None
    db: str
    db_name: Optional[str] = None
    default_bucket: Optional[str] = None
    workers: Optional[int] = None
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None
    disk_gb: Optional[int] = None


class DeleteDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_DB] = ServiceOperationType.DELETE_DB
    org: Optional[str] = None
    db: str


class StartDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.START_DB] = ServiceOperationType.START_DB
    org: Optional[str] = None
    db: str


class StopDbRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.STOP_DB] = ServiceOperationType.STOP_DB
    org: Optional[str] = None
    db: str


class UpdateRuntimeRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_RUNTIME] = ServiceOperationType.UPDATE_RUNTIME
    org: Optional[str] = None
    db: str
    bundle_s3_key: str


class GetBundleUploadUrlRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_BUNDLE_UPLOAD_URL] = ServiceOperationType.GET_BUNDLE_UPLOAD_URL
    org: Optional[str] = None
    db: str


class GetBundleUploadUrlResponse(BaseModel):
    presigned_url: str
    bundle_s3_key: str


# Secrets


class SetSecretRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.SET_SECRET] = ServiceOperationType.SET_SECRET
    org: str
    db: Optional[str] = None
    key: str
    value: str


class SetSecretResponse(BaseModel):
    key: str


class DeleteSecretRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_SECRET] = ServiceOperationType.DELETE_SECRET
    org: str
    db: Optional[str] = None
    key: str


class DeleteSecretResponse(BaseModel):
    key: str


class ListSecretsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SECRETS] = ServiceOperationType.LIST_SECRETS
    org: str
    db: Optional[str] = None


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
    endpoint: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    service_config: Optional[str] = None  # JSON-serialized ServiceConfig from latest run


class CreateServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.CREATE_SERVICE] = ServiceOperationType.CREATE_SERVICE
    org: Optional[str] = None
    db: str
    service_name: str
    base_path: str = ''
    workers_min: int = 1
    description: Optional[str] = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10
    service_config: Optional[ServiceConfig] = None

    @field_validator('service_config', mode='before')
    @classmethod
    def _parse_service_config(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v)
        return v


class CreateServiceResponse(BaseModel):
    service: ServiceRecord


class GetServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_SERVICE] = ServiceOperationType.GET_SERVICE
    org: Optional[str] = None
    db: str
    service_name: str


class GetServiceResponse(BaseModel):
    service: ServiceRecord


class ListServicesRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SERVICES] = ServiceOperationType.LIST_SERVICES
    org: Optional[str] = None
    db: str


class ListServicesResponse(BaseModel):
    services: list[ServiceRecord]


class StartServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.START_SERVICE] = ServiceOperationType.START_SERVICE
    org: Optional[str] = None
    db: str
    service_name: str


class StartServiceResponse(BaseModel):
    service: ServiceRecord


class StopServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.STOP_SERVICE] = ServiceOperationType.STOP_SERVICE
    org: Optional[str] = None
    db: str
    service_name: str


class StopServiceResponse(BaseModel):
    service: ServiceRecord


class UpdateServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_SERVICE] = ServiceOperationType.UPDATE_SERVICE
    org: Optional[str] = None
    db: str
    service_name: str
    workers_min: Optional[int] = None
    description: Optional[str] = None
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None
    disk_gb: Optional[int] = None
    service_config: Optional[ServiceConfig] = None

    @field_validator('service_config', mode='before')
    @classmethod
    def _parse_service_config(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v)
        return v


class UpdateServiceResponse(BaseModel):
    service: ServiceRecord


class DeleteServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_SERVICE] = ServiceOperationType.DELETE_SERVICE
    org: Optional[str] = None
    db: str
    service_name: str


class DeleteServiceResponse(BaseModel):
    service_name: str


class ListServiceRunsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SERVICE_RUNS] = ServiceOperationType.LIST_SERVICE_RUNS
    org: Optional[str] = None
    db: str
    service_name: str


class GetServiceRunRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_SERVICE_RUN] = ServiceOperationType.GET_SERVICE_RUN
    org: Optional[str] = None
    db: str
    service_name: str
    run_id: str


class ServiceRunRecord(BaseModel):
    run_id: str
    workers_min: int
    state: str  # AVAILABLE | STOPPED | FAILED
    started_at: float
    stopped_at: Optional[float] = None
    runtime_build_id: Optional[str] = None
    bundle_r2_path: Optional[str] = None
    service_config: Optional[str] = None  # JSON-serialized ServiceConfig for this run
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
    default_db: Optional[str] = None
    created_at: float
    updated_at: float


class ListOrgsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_ORGS] = ServiceOperationType.LIST_ORGS


class ListOrgsResponse(BaseModel):
    orgs: list[OrgRecord]
