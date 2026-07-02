"""Operation type enums for the Pixeltable cloud protocol."""

from __future__ import annotations

from enum import Enum


class ServiceOperationType(str, Enum):
    """Operation types for cloud database and service management."""

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

    SET_SECRET = 'set_secret'
    DELETE_SECRET = 'delete_secret'
    LIST_SECRETS = 'list_secrets'

    START_DATABASE = 'start_database'
    STOP_DATABASE = 'stop_database'
    UPDATE_DATABASE = 'update_database'
    UPDATE_RUNTIME = 'update_runtime'

    LIST_ORGS = 'list_orgs'


# DatabaseOperationType mirrors the database-related entries above for use by database.py.
class DatabaseOperationType(str, Enum):
    CREATE_DATABASE = 'create_database'
    GET_DATABASE = 'get_database'
    LIST_DATABASES = 'list_databases'
    UPDATE_DATABASE = 'update_database'
    DELETE_DATABASE = 'delete_database'
    STOP_DATABASE = 'stop_database'
    START_DATABASE = 'start_database'
    UPDATE_RUNTIME = 'update_runtime'


SERVICE_OPERATIONS = frozenset(ServiceOperationType)
DATABASE_OPERATIONS = frozenset(DatabaseOperationType)
