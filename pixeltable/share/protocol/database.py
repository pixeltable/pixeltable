"""Protocol models for cloud-hosted database operations.

Defines request/response types shared between the pixeltable CLI and the cloud server.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from .operation_types import DatabaseOperationType


class DatabaseRecord(BaseModel):
    org_id: str
    db_slug: str
    db_name: str
    state: str = 'ACTIVE'
    location: str = ''
    created_at: float
    updated_at: float


class CreateDatabaseRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.CREATE_DATABASE] = DatabaseOperationType.CREATE_DATABASE
    org_slug: Optional[str] = None
    db_slug: str
    db_name: Optional[str] = None
    location: Optional[str] = None
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10


class GetDatabaseRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.GET_DATABASE] = DatabaseOperationType.GET_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class ListDatabaseRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.LIST_DATABASES] = DatabaseOperationType.LIST_DATABASES
    org_slug: Optional[str] = None


class UpdateDatabaseRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.UPDATE_DATABASE] = DatabaseOperationType.UPDATE_DATABASE
    org_slug: Optional[str] = None
    db_slug: str
    db_name: Optional[str] = None
    default_bucket: Optional[str] = None
    workers: Optional[int] = None


class DeleteDatabaseRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.DELETE_DATABASE] = DatabaseOperationType.DELETE_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StopDatabaseRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.STOP_DATABASE] = DatabaseOperationType.STOP_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class StartDatabaseRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.START_DATABASE] = DatabaseOperationType.START_DATABASE
    org_slug: Optional[str] = None
    db_slug: str


class UpdateRuntimeRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.UPDATE_RUNTIME] = DatabaseOperationType.UPDATE_RUNTIME
    org_slug: Optional[str] = None
    db_slug: str
    bundle_s3_key: str


# db_slug=None → org-scoped secret; db_slug=<slug> → DB-scoped secret.
# list_secrets returns key names only, never values.


class SetSecretRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.SET_SECRET] = DatabaseOperationType.SET_SECRET
    org_slug: str
    db_slug: Optional[str] = None
    key: str
    value: str


class SetSecretResponse(BaseModel):
    key: str


class DeleteSecretRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.DELETE_SECRET] = DatabaseOperationType.DELETE_SECRET
    org_slug: str
    db_slug: Optional[str] = None
    key: str


class DeleteSecretResponse(BaseModel):
    key: str


class ListSecretsRequest(BaseModel):
    operation_type: Literal[DatabaseOperationType.LIST_SECRETS] = DatabaseOperationType.LIST_SECRETS
    org_slug: str
    db_slug: Optional[str] = None


class ListSecretsResponse(BaseModel):
    keys: list[str]
