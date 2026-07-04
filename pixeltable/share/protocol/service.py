"""Protocol models for Pixeltable cloud database and service operations."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

from .operation_types import ServiceOperationType

# ── Database ──────────────────────────────────────────────────────────────────


class DatabaseRecord(BaseModel):
    database_id: str
    org_id: str
    db_name: str
    location: str
    region: str
    state: str  # PROVISIONING | ACTIVE | SLEEPING | DELETING
    endpoint: Optional[str] = None  # proxy endpoint host (for pxt:// connections)
    created_at: float


class CreateDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.CREATE_DATABASE] = ServiceOperationType.CREATE_DATABASE
    org_slug: str
    db_name: str
    location: str = 'aws'
    region: str = 'us-east-1'
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10


class CreateDatabaseResponse(BaseModel):
    database: DatabaseRecord


class GetDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_DATABASE] = ServiceOperationType.GET_DATABASE
    org_slug: str
    db_name: str


class GetDatabaseResponse(BaseModel):
    database: DatabaseRecord


class ListDatabasesRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.LIST_DATABASES] = ServiceOperationType.LIST_DATABASES
    org_slug: str


class ListDatabasesResponse(BaseModel):
    databases: list[DatabaseRecord]


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
    org_slug: str
    db_name: str


class DeleteDatabaseResponse(BaseModel):
    db_name: str


# Alias matching the cloud protocol's naming (no trailing 's').
ListDatabaseRequest = ListDatabasesRequest


# ── Service ───────────────────────────────────────────────────────────────────


class ServiceRecord(BaseModel):
    service_id: str
    org_id: str
    database_id: str
    service_name: str
    table_path: str  # in-db path to the table backing this service
    workers_min: int = 1
    workers_max: int = 1
    state: str  # DEPLOYING | STARTING | RUNNING | STOPPED | FAILED
    endpoint: Optional[str] = None
    error: Optional[str] = None
    created_at: float


class CreateServiceRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.CREATE_SERVICE] = ServiceOperationType.CREATE_SERVICE
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str
    table_path: str
    workers_min: int = 1
    description: Optional[str] = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10


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


# ── Start / Stop / UpdateRuntime ──────────────────────────────────────────────


class StartDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.START_DATABASE] = ServiceOperationType.START_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StartDatabaseResponse(BaseModel):
    message: str = 'started'


class StopDatabaseRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.STOP_DATABASE] = ServiceOperationType.STOP_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StopDatabaseResponse(BaseModel):
    message: str = 'stopped'


class UpdateRuntimeRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.UPDATE_RUNTIME] = ServiceOperationType.UPDATE_RUNTIME
    org_slug: Optional[str] = None
    db_slug: str
    bundle_s3_key: str


class UpdateRuntimeResponse(BaseModel):
    message: str = 'runtime update triggered'


class GetBundleUploadUrlRequest(BaseModel):
    operation_type: Literal[ServiceOperationType.GET_BUNDLE_UPLOAD_URL] = ServiceOperationType.GET_BUNDLE_UPLOAD_URL
    org_slug: Optional[str] = None
    db_slug: str


class GetBundleUploadUrlResponse(BaseModel):
    presigned_url: str
    bundle_s3_key: str


# ── Service Runs ──────────────────────────────────────────────────────────────


class ServiceRunRecord(BaseModel):
    run_id: str
    workers_min: int
    state: str  # STARTING | RUNNING | STOPPED | FAILED
    started_at: float
    stopped_at: Optional[float] = None
    runtime_build_id: Optional[str] = None
    bundle_r2_path: Optional[str] = None


class ListServiceRunsResponse(BaseModel):
    runs: list[ServiceRunRecord]


class GetServiceRunResponse(BaseModel):
    run: ServiceRunRecord


# ── Org ───────────────────────────────────────────────────────────────────────


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
