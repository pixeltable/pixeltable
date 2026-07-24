"""Cloud API protocol — shared request/response models between the pxt SDK and cloud server."""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from pixeltable.config import ServiceConfig


class PixeltableStoreOperationType(str, Enum):
    GET_BUCKET_CREDENTIALS = 'get_bucket_credentials'
    GET_PRESIGNED_URL = 'get_presigned_url'


class ServiceOperationType(str, Enum):
    CREATE_DATABASE = 'create_database'
    GET_DATABASE = 'get_database'
    LIST_DATABASES = 'list_databases'
    DELETE_DATABASE = 'delete_database'

    CREATE_SERVICE = 'create_service'
    GET_SERVICE = 'get_service'
    LIST_SERVICES = 'list_services'
    UPDATE_SERVICE = 'update_service'
    START_SERVICE = 'start_service'
    STOP_SERVICE = 'stop_service'
    DELETE_SERVICE = 'delete_service'
    LIST_SERVICE_RUNS = 'list_service_runs'
    GET_SERVICE_RUN = 'get_service_run'

    START_DATABASE = 'start_database'
    STOP_DATABASE = 'stop_database'
    UPDATE_DATABASE = 'update_database'
    UPDATE_RUNTIME = 'update_runtime'
    GET_BUNDLE_UPLOAD_URL = 'get_bundle_upload_url'

    LIST_ORGS = 'list_orgs'

    SET_SECRET = 'set_secret'
    DELETE_SECRET = 'delete_secret'
    LIST_SECRETS = 'list_secrets'


# Database operations


class CreateDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.CREATE_DATABASE] = ServiceOperationType.CREATE_DATABASE
    org_slug: Optional[str] = None
    db_slug: str
    db_name: Optional[str] = None
    location: Optional[str] = None
    region: Optional[str] = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10


class GetDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_DATABASE] = ServiceOperationType.GET_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class ListDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_DATABASES] = ServiceOperationType.LIST_DATABASES
    org_slug: Optional[str] = None


class UpdateDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_DATABASE] = ServiceOperationType.UPDATE_DATABASE
    org_slug: Optional[str] = None
    db_slug: str
    db_name: Optional[str] = None
    default_bucket: Optional[str] = None
    workers: Optional[int] = None
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None
    disk_gb: Optional[int] = None


class DeleteDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_DATABASE] = ServiceOperationType.DELETE_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StartDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.START_DATABASE] = ServiceOperationType.START_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StopDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.STOP_DATABASE] = ServiceOperationType.STOP_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class UpdateRuntimeRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_RUNTIME] = ServiceOperationType.UPDATE_RUNTIME
    org_slug: Optional[str] = None
    db_slug: str
    bundle_s3_key: str


class GetBundleUploadUrlRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_BUNDLE_UPLOAD_URL] = ServiceOperationType.GET_BUNDLE_UPLOAD_URL
    org_slug: Optional[str] = None
    db_slug: str


class GetBundleUploadUrlResponse(BaseModel):
    presigned_url: str
    bundle_s3_key: str


# Secrets


class SetSecretRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.SET_SECRET] = ServiceOperationType.SET_SECRET
    org_slug: str
    db_slug: Optional[str] = None
    key: str
    value: str


class SetSecretResponse(BaseModel):
    key: str


class DeleteSecretRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.DELETE_SECRET] = ServiceOperationType.DELETE_SECRET
    org_slug: str
    db_slug: Optional[str] = None
    key: str


class DeleteSecretResponse(BaseModel):
    key: str


class ListSecretsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SECRETS] = ServiceOperationType.LIST_SECRETS
    org_slug: str
    db_slug: Optional[str] = None


class ListSecretsResponse(BaseModel):
    keys: list[str]


# Services


class ServiceRecord(BaseModel):
    service_id: str
    org_id: str
    database_id: str
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
    org_slug: Optional[str] = None
    db_slug: str
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
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str


class GetServiceResponse(BaseModel):
    service: ServiceRecord


class ListServicesRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SERVICES] = ServiceOperationType.LIST_SERVICES
    org_slug: Optional[str] = None
    db_slug: str


class ListServicesResponse(BaseModel):
    services: list[ServiceRecord]


class StartServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.START_SERVICE] = ServiceOperationType.START_SERVICE
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str


class StartServiceResponse(BaseModel):
    service: ServiceRecord


class StopServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.STOP_SERVICE] = ServiceOperationType.STOP_SERVICE
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str


class StopServiceResponse(BaseModel):
    service: ServiceRecord


class UpdateServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_SERVICE] = ServiceOperationType.UPDATE_SERVICE
    org_slug: Optional[str] = None
    db_slug: str
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
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str


class DeleteServiceResponse(BaseModel):
    service_name: str


class ListServiceRunsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_SERVICE_RUNS] = ServiceOperationType.LIST_SERVICE_RUNS
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str


class GetServiceRunRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_SERVICE_RUN] = ServiceOperationType.GET_SERVICE_RUN
    org_slug: Optional[str] = None
    db_slug: str
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
    org_slug: str
    default_db_slug: Optional[str] = None
    created_at: float
    updated_at: float


class ListOrgsRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_ORGS] = ServiceOperationType.LIST_ORGS


class ListOrgsResponse(BaseModel):
    orgs: list[OrgRecord]


# Home bucket credentials and presigned URLs


class GetBucketCredentialsRequest(BaseModel):
    operation_type: Literal[PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS] = (
        PixeltableStoreOperationType.GET_BUCKET_CREDENTIALS
    )
    org_slug: str
    db_slug: str
    bucket_name: str
    prefix: Optional[str] = None


class GetBucketCredentialsResponse(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: str
    endpoint_url: str
    storage_provider: str
    resolved_bucket_name: str
    ttl_seconds: int
    prefix: Optional[str] = None
    no_space_left: bool = False


class GetPresignedUrlRequest(BaseModel):
    operation_type: Literal[PixeltableStoreOperationType.GET_PRESIGNED_URL] = (
        PixeltableStoreOperationType.GET_PRESIGNED_URL
    )
    org_slug: str
    db_slug: str
    bucket_name: str
    key: str
    method: str = 'get'
    expiration: int = 3600


class GetPresignedUrlResponse(BaseModel):
    url: str
    key: str
    expiration: int
