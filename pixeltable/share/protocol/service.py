"""Protocol models for Pixeltable cloud database and service operations."""

from __future__ import annotations

from typing import Any, Optional

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
    operation_type: str = ServiceOperationType.CREATE_DATABASE
    org_slug: str
    db_name: str
    location: str = 'aws'
    region: str = 'us-east-1'


class CreateDatabaseResponse(BaseModel):
    database: DatabaseRecord


class GetDatabaseRequest(BaseModel):
    operation_type: str = ServiceOperationType.GET_DATABASE
    org_slug: str
    db_name: str


class GetDatabaseResponse(BaseModel):
    database: DatabaseRecord


class ListDatabasesRequest(BaseModel):
    operation_type: str = ServiceOperationType.LIST_DATABASES
    org_slug: str


class ListDatabasesResponse(BaseModel):
    databases: list[DatabaseRecord]


class UpdateDatabaseRequest(BaseModel):
    operation_type: str = ServiceOperationType.UPDATE_DATABASE
    org_slug: Optional[str] = None
    db_slug: str
    db_name: Optional[str] = None
    default_bucket: Optional[str] = None
    workers: Optional[int] = None


class DeleteDatabaseRequest(BaseModel):
    operation_type: str = ServiceOperationType.DELETE_DATABASE
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
    operation_type: str = ServiceOperationType.CREATE_SERVICE
    org_slug: str
    db_name: str
    service_name: str
    table_path: str
    workers_min: int = 1
    workers_max: int = 1


class CreateServiceResponse(BaseModel):
    service: ServiceRecord


class GetServiceRequest(BaseModel):
    operation_type: str = ServiceOperationType.GET_SERVICE
    org_slug: str
    db_name: str
    service_name: str


class GetServiceResponse(BaseModel):
    service: ServiceRecord


class ListServicesRequest(BaseModel):
    operation_type: str = ServiceOperationType.LIST_SERVICES
    org_slug: str
    db_name: str


class ListServicesResponse(BaseModel):
    services: list[ServiceRecord]


class StartServiceRequest(BaseModel):
    operation_type: str = ServiceOperationType.START_SERVICE
    org_slug: str
    db_name: str
    service_name: str
    workers_min: Optional[int] = None
    workers_max: Optional[int] = None


class StartServiceResponse(BaseModel):
    service: ServiceRecord


class StopServiceRequest(BaseModel):
    operation_type: str = ServiceOperationType.STOP_SERVICE
    org_slug: str
    db_name: str
    service_name: str


class StopServiceResponse(BaseModel):
    service: ServiceRecord


class UpdateServiceRequest(BaseModel):
    operation_type: str = ServiceOperationType.UPDATE_SERVICE
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str
    workers_min: Optional[int] = None
    description: Optional[str] = None


class UpdateServiceResponse(BaseModel):
    service: ServiceRecord


class DeleteServiceRequest(BaseModel):
    operation_type: str = ServiceOperationType.DELETE_SERVICE
    org_slug: str
    db_name: str
    service_name: str


class DeleteServiceResponse(BaseModel):
    service_name: str


class ListServiceRunsRequest(BaseModel):
    operation_type: str = ServiceOperationType.LIST_SERVICE_RUNS
    org_slug: Optional[str] = None
    db_slug: str
    service_name: str


# ── Secrets ───────────────────────────────────────────────────────────────────


class SetSecretRequest(BaseModel):
    operation_type: str = ServiceOperationType.SET_SECRET
    org_slug: str
    db_name: str
    key: str
    value: str


class SetSecretResponse(BaseModel):
    key: str


class DeleteSecretRequest(BaseModel):
    operation_type: str = ServiceOperationType.DELETE_SECRET
    org_slug: str
    db_name: str
    key: str


class DeleteSecretResponse(BaseModel):
    key: str


class ListSecretsRequest(BaseModel):
    operation_type: str = ServiceOperationType.LIST_SECRETS
    org_slug: str
    db_name: str


class ListSecretsResponse(BaseModel):
    keys: list[str]


# ── Start / Stop / UpdateRuntime ──────────────────────────────────────────────


class StartDatabaseRequest(BaseModel):
    operation_type: str = ServiceOperationType.START_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StartDatabaseResponse(BaseModel):
    message: str = 'started'


class StopDatabaseRequest(BaseModel):
    operation_type: str = ServiceOperationType.STOP_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StopDatabaseResponse(BaseModel):
    message: str = 'stopped'


class UpdateRuntimeRequest(BaseModel):
    operation_type: str = ServiceOperationType.UPDATE_RUNTIME
    org_slug: Optional[str] = None
    db_slug: str
    runtime_image: Optional[str] = None


class UpdateRuntimeResponse(BaseModel):
    message: str = 'runtime update triggered'


# ── Org ───────────────────────────────────────────────────────────────────────


class OrgRecord(BaseModel):
    org_id: str
    org_slug: str
    default_db_slug: Optional[str] = None
    created_at: float
    updated_at: float


class ListOrgsRequest(BaseModel):
    operation_type: str = ServiceOperationType.LIST_ORGS


class ListOrgsResponse(BaseModel):
    orgs: list[OrgRecord]
