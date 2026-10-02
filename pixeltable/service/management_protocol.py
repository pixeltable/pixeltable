"""Management API protocol: request/response models shared between the Pixeltable SDK and the Pixeltable cloud server.

Each request is org-scoped and requires an API key for authorization. Some requests include an optional org parameter.
If an org is set, the server will verify that the API key belongs to that org. This is to avoid scenarios in which
the client thinks that it acts on one org whereas its API key actually points to the other.

The pixeltable-cloud repo imports this module, so care must be taken when making backwards-incompatible changes."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pixeltable.service.db_md import DatabaseResources, DatabaseStatus
from pixeltable.service.service_md import ServiceInstanceRecord
from pixeltable.utils.project import ProjectFingerprint
from pixeltable_cli.types import DbArtifact, DbPlan, GenerationReceipt, ServiceSpec
from pixeltable_cli.utils import hosted_name_error


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
    RESTART_SERVICE_INSTANCE = 'restart_service'
    DELETE_SERVICE_INSTANCE = 'delete_service'

    START_DB = 'start_db'
    STOP_DB = 'stop_db'
    RESTART_DB = 'restart_db'
    UPDATE_DB = 'update_db'
    PREPARE_UPDATE = 'prepare_update'
    SUBMIT_UPDATE = 'submit_update'
    GET_RECEIPTS = 'get_receipts'
    RETRY = 'retry'
    GET_ARCHIVE = 'get_archive'
    GET_LOGS = 'get_logs'

    CREATE_ORG = 'create_org'
    LIST_ORGS = 'list_orgs'

    SET_SECRET = 'set_secret'
    DELETE_SECRET = 'delete_secret'
    # TODO(PXT-1438): delete this when we no longer need to support older pxt cli
    LIST_SECRETS = 'list_secrets'
    LIST_ALL_SECRETS = 'list_all_secrets'

    CREATE_KEY = 'create_key'
    LIST_KEYS = 'list_keys'
    UPDATE_KEY = 'update_key'
    DELETE_KEY = 'delete_key'


# Db operations


def _validate_hosted_name(value: str, kind: str) -> str:
    error = hosted_name_error(value, kind)
    if error is not None:
        raise ValueError(error)
    return value


class DatabaseReport(BaseModel):
    """A hosted db's requested resources and the ones it provides."""

    model_config = ConfigDict(extra='ignore')

    db: str = ''

    target_resources: DatabaseResources | None = Field(
        default=None, description='what the project configuration asks for; an update moves the database to this'
    )

    current: DatabaseStatus | None = Field(
        default=None, description='what the database provides now; null when the database does not exist'
    )

    receipt: GenerationReceipt | None = Field(
        default=None, description="the receipt of the database's current desired generation"
    )


class GetDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_DB] = ManagementOperationType.GET_DB
    org: str | None = None
    db: str


class GetDbResponse(BaseModel):
    report: DatabaseReport

    # one entry per pod serving the database; DatabaseResources.workers is how many should run
    worker_status: list[dict[str, Any]] = Field(default_factory=list)


class ListDbRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_DBS] = ManagementOperationType.LIST_DBS
    org: str | None = None


class UpdateDbRequest(BaseModel):
    """Request for changing the running system to match the spec."""

    operation_type: Literal[ManagementOperationType.UPDATE_DB] = ManagementOperationType.UPDATE_DB
    org: str | None = None
    db: str
    target: DatabaseResources | None = None

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
    report: DatabaseReport

    # if non-empty: perform the uploads first, then retry the request
    uploads: list[ArtifactUpload] = Field(default_factory=list)


# Submissions


_SHA256_HEX = re.compile(r'[0-9a-f]{64}')


class BlobRef(BaseModel):
    """A project archive, named by the SHA-256 and size of its bytes."""

    sha256: str
    size: int = Field(ge=0)

    @field_validator('sha256')
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if _SHA256_HEX.fullmatch(value) is None:
            raise ValueError(f'sha256 must be 64 lowercase hex digits, got {value!r}')
        return value


class BlobUpload(BaseModel):
    """Where to put a blob the control plane does not hold, and the headers its url was signed with."""

    model_config = ConfigDict(extra='ignore')

    url: str
    headers: dict[str, str] = Field(default_factory=dict)


class DatabaseTarget(DatabaseResources):
    """What a project asks of its hosted database; the control plane keeps the rest of its desired spec."""

    default_bucket: str | None = Field(default=None, exclude=True)

    force_build_nonce: str | None = Field(
        default=None, description='changed only to rebuild the image of a project that did not change'
    )


ReleasePin = Literal['latest', 'keep', 'submission']


class ServiceMutation(BaseModel):
    """The full desired spec of one service."""

    service_name: str
    base_path: str = ''
    lifecycle: Literal['RUNNING', 'STOPPED', 'DELETED'] = 'RUNNING'
    spec: ServiceSpec
    app_module: str
    otel: bool = False
    workers: int = 1
    cpu: float = 0.5
    memory_mb: int = 512
    disk_gb: int = 10
    description: str | None = None
    pin: ReleasePin | None = Field(
        default=None,
        description="the release intent: 'latest' runs the release of the database's current desired generation, "
        "'keep' the release the service is pinned to, 'submission' the release of the database generation created "
        'by the same submission; null creates a service on the latest release',
    )


class ServiceGeneration(BaseModel):
    service_name: str
    base_path: str = ''
    generation: int = Field(ge=0)


class ExpectedGenerations(BaseModel):
    """The generations a submission was prepared against: 0 for a resource that does not exist."""

    model_config = ConfigDict(extra='ignore')

    database: int | None = Field(default=None, ge=0)
    services: list[ServiceGeneration] = Field(default_factory=list)

    def service(self, service_name: str, base_path: str = '') -> int:
        return next(
            (s.generation for s in self.services if (s.service_name, s.base_path) == (service_name, base_path)), 0
        )


class PrepareUpdateRequest(BaseModel):
    """Read the generations a submission would replace, its plan, and where to upload its archive."""

    operation_type: Literal[ManagementOperationType.PREPARE_UPDATE] = ManagementOperationType.PREPARE_UPDATE
    org: str | None = None
    db: str
    database_target: DatabaseTarget | None = None
    service_mutations: list[ServiceMutation] = Field(default_factory=list)
    blob: BlobRef | None = None

    @field_validator('db')
    @classmethod
    def _validate_db_name(cls, value: str) -> str:
        return _validate_hosted_name(value, 'Database name')


class PrepareUpdateResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    generations: ExpectedGenerations
    report: DatabaseReport
    plan: DbPlan | None = Field(default=None, description="the database's dry-run plan; null without a target")
    services: list[ServiceMutation] = Field(
        default_factory=list, description='the current desired spec of each touched service that exists'
    )
    upload: BlobUpload | None = Field(default=None, description='null when the control plane holds the blob')


class SubmitUpdateRequest(BaseModel):
    """Accept the next desired spec of every touched resource, all or none.

    A repeat of an accepted submission returns the receipts it was accepted as, so resending one is safe.
    """

    operation_type: Literal[ManagementOperationType.SUBMIT_UPDATE] = ManagementOperationType.SUBMIT_UPDATE
    org: str | None = None
    db: str
    database_target: DatabaseTarget | None = None
    service_mutations: list[ServiceMutation] = Field(default_factory=list)
    blob: BlobRef | None = None
    expected_generations: ExpectedGenerations

    @field_validator('db')
    @classmethod
    def _validate_db_name(cls, value: str) -> str:
        return _validate_hosted_name(value, 'Database name')

    @model_validator(mode='after')
    def _validate_mutation(self) -> SubmitUpdateRequest:
        if self.database_target is None and len(self.service_mutations) == 0:
            raise ValueError('a submission needs a database target or a service mutation')
        if self.database_target is not None and self.expected_generations.database is None:
            raise ValueError('a database target needs the expected database generation')
        if self.database_target is None and any(m.pin == 'submission' for m in self.service_mutations):
            raise ValueError("a service pinned to 'submission' needs a database target in the same submission")
        return self


class SubmitUpdateResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    receipts: list[GenerationReceipt]


class ReceiptRef(BaseModel):
    kind: str
    resource_id: str
    generation: int

    @classmethod
    def of(cls, receipt: GenerationReceipt) -> ReceiptRef:
        return cls(kind=receipt.kind, resource_id=receipt.resource_id, generation=receipt.generation)


class GetReceiptsRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.GET_RECEIPTS] = ManagementOperationType.GET_RECEIPTS
    org: str | None = None
    db: str
    receipts: list[ReceiptRef]


class GetReceiptsResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    receipts: list[GenerationReceipt]


class RetryRequest(BaseModel):
    """Start a new attempt of a current generation that failed."""

    operation_type: Literal[ManagementOperationType.RETRY] = ManagementOperationType.RETRY
    org: str | None = None
    db: str
    kind: Literal['database', 'service']
    service_name: str | None = None
    base_path: str = ''
    generation: int

    @model_validator(mode='after')
    def _validate_service(self) -> RetryRequest:
        if (self.kind == 'service') != (self.service_name is not None):
            raise ValueError('a service_name is required for a service retry, and only for one')
        return self


class RetryResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    receipt: GenerationReceipt


class DbReceiptResponse(BaseModel):
    """What a database start, stop, restart or delete was accepted as."""

    model_config = ConfigDict(extra='ignore')

    receipt: GenerationReceipt


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


class RestartDbRequest(BaseModel):
    """Cycle the database's pods onto the image and archive it already runs."""

    operation_type: Literal[ManagementOperationType.RESTART_DB] = ManagementOperationType.RESTART_DB
    org: str | None = None
    db: str


class GetArchiveRequest(BaseModel):
    """Ask for a url serving a project archive of the database; a pod sends this as it starts.

    A pod pinned to a release names its archive digest and build id, and gets that release's archive; without
    them it gets the archive of the database's current release.
    """

    operation_type: Literal[ManagementOperationType.GET_ARCHIVE] = ManagementOperationType.GET_ARCHIVE
    org: str | None = None
    db: str
    archive_digest: str | None = None
    build_id: str | None = None


class GetArchiveResponse(BaseModel):
    presigned_url: str
    # ProjectFingerprint.archive_digest() of the archive the url serves
    digest: str

    # the fingerprint the archive was published under. A pod reports this rather than one it computes:
    # loading the application file writes bytecode into the unpacked project, so a pod that walked its own
    # directory would report files the published project never held.
    fingerprint: ProjectFingerprint | None = None


class GetLogsRequest(BaseModel):
    """Read the log of the database pod, or of one service when service_name is given.

    The log merges the process's log records with its console output, ordered by time.
    """

    operation_type: Literal[ManagementOperationType.GET_LOGS] = ManagementOperationType.GET_LOGS
    org: str | None = None
    db: str
    service_name: str | None = None
    base_path: str = ''
    since_seconds: int = Field(default=3600, ge=1)
    # only the newest limit lines of the window are returned
    limit: int = Field(default=200, ge=1, le=10000)
    # the readiness and liveness probes are nearly the whole log, so they are left out by default
    include_health: bool = False


class LogRecord(BaseModel):
    # the time the line was written, in milliseconds since the epoch
    ts_ms: int
    line: str


class GetLogsResponse(BaseModel):
    # oldest first
    records: list[LogRecord]


# Secrets


class SetSecretRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.SET_SECRET] = ManagementOperationType.SET_SECRET
    org: str
    db: str | None = None
    key: str
    value: str

    @field_validator('key')
    @classmethod
    def _validate_key(cls, key: str) -> str:
        # TODO(PXT-1418): pxt secret operations can fail partially
        if key.upper().startswith('PIXELTABLE_'):
            raise ValueError(f'Invalid secret name {key!r}: the PIXELTABLE_ prefix is reserved.')
        return key


class SetSecretResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    key: str
    revision: int | None = None


class DeleteSecretRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_SECRET] = ManagementOperationType.DELETE_SECRET
    org: str
    db: str | None = None
    key: str


class DeleteSecretResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    key: str
    revision: int | None = None


class ListSecretsRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_SECRETS] = ManagementOperationType.LIST_SECRETS
    org: str | None = None
    db: str | None = None


class ListSecretsResponse(BaseModel):
    keys: list[str]


class ListAllSecretsRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_ALL_SECRETS] = ManagementOperationType.LIST_ALL_SECRETS
    org: str | None = None
    # If db is set, the server returns that database's secrets and org-wide secrets. Otherwise it returns all secrets in
    # the org.
    db: str | None = None


class SecretListItem(BaseModel):
    model_config = ConfigDict(extra='ignore')

    key: str
    # None for an org-level secret
    db: str | None


class ListAllSecretsResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    org: str
    secrets: list[SecretListItem]


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
    model_config = ConfigDict(extra='ignore')

    instance: ServiceInstanceRecord
    receipt: GenerationReceipt | None = None


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
    model_config = ConfigDict(extra='ignore')

    instance: ServiceInstanceRecord
    receipt: GenerationReceipt | None = None


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
    model_config = ConfigDict(extra='ignore')

    instance: ServiceInstanceRecord
    receipt: GenerationReceipt | None = None


class StopServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.STOP_SERVICE_INSTANCE] = (
        ManagementOperationType.STOP_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''


class StopServiceInstanceResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    instance: ServiceInstanceRecord
    receipt: GenerationReceipt | None = None


class RestartServiceInstanceRequest(BaseModel):
    """Cycle the service's pods onto the image and archive they already run."""

    operation_type: Literal[ManagementOperationType.RESTART_SERVICE_INSTANCE] = (
        ManagementOperationType.RESTART_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''


class RestartServiceInstanceResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    instance: ServiceInstanceRecord
    receipt: GenerationReceipt | None = None


class DeleteServiceInstanceRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_SERVICE_INSTANCE] = (
        ManagementOperationType.DELETE_SERVICE_INSTANCE
    )
    org: str | None = None
    db: str
    service_name: str
    base_path: str = ''


class DeleteServiceInstanceResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')

    service_name: str
    receipt: GenerationReceipt | None = None


# Orgs


class OrgRecord(BaseModel):
    org_id: str
    org: str
    default_db: str | None = None
    created_at: float
    updated_at: float


class CreateOrgRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.CREATE_ORG] = ManagementOperationType.CREATE_ORG
    org: str  # the namespace in pxt://org:db; unique across Pixeltable

    # the backing WorkOS organization. A dashboard has already created one and sets this; with no
    # WorkOS credentials, a CLI leaves it empty and the control plane creates the organization.
    org_id: str | None = None

    display_name: str | None = None  # what people see; defaults to org
    location: str | None = None  # e.g. 'aws/us-east-1'; defaults to the primary region

    @field_validator('org')
    @classmethod
    def _validate_org(cls, value: str) -> str:
        return _validate_hosted_name(value, 'Organization name')


class CreateOrgResponse(BaseModel):
    org_id: str
    org: str
    default_db: str | None = None
    created_at: datetime
    updated_at: datetime


class ListOrgsRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_ORGS] = ManagementOperationType.LIST_ORGS


class ListOrgsResponse(BaseModel):
    orgs: list[OrgRecord]


# API keys
#
# The caller's credential decides the organization: no request here has an organization field, and a grant in
# another organization is refused.


class KeyRecord(BaseModel):
    name: str
    key_type: Literal['user', 'runtime']  # the Principal.type the control plane records for it
    grants: list[str] = Field(default_factory=list)  # empty for a key that acts as its creator
    created_at: datetime
    # Who created it: an email when the control plane knows one, else a user id; empty when unknown.
    created_by: str = ''
    # Set only in a create response: the secret is shown once and never stored in retrievable form.
    api_key: str | None = None


class CreateKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.CREATE_KEY] = ManagementOperationType.CREATE_KEY
    name: str
    # Each a verb on a pxt:// resource, such as 'access:pxt://acme:main/services/ingest'. Empty asks for
    # a key that acts as you; any grant asks for one that acts as nobody.
    grants: list[str] = Field(default_factory=list)


class CreateKeyResponse(BaseModel):
    key: KeyRecord


class ListKeysRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.LIST_KEYS] = ManagementOperationType.LIST_KEYS


class ListKeysResponse(BaseModel):
    keys: list[KeyRecord]


class UpdateKeyRequest(BaseModel):
    """Add and remove grants on an existing key, leaving the rest alone.

    A delta rather than a replacement list, so granting one more resource does not depend on the
    caller first knowing -- and faithfully resending -- everything the key already had.
    """

    operation_type: Literal[ManagementOperationType.UPDATE_KEY] = ManagementOperationType.UPDATE_KEY
    name: str
    # grants in the form CreateKeyRequest.grants takes
    allow: list[str] = Field(default_factory=list)
    revoke: list[str] = Field(default_factory=list)


class UpdateKeyResponse(BaseModel):
    key: KeyRecord


class DeleteKeyRequest(BaseModel):
    operation_type: Literal[ManagementOperationType.DELETE_KEY] = ManagementOperationType.DELETE_KEY
    name: str


class DeleteKeyResponse(BaseModel):
    name: str
