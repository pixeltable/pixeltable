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
    START_SERVICE = 'start_service'
    STOP_SERVICE = 'stop_service'
    DELETE_SERVICE = 'delete_service'

    SET_SECRET = 'set_secret'
    DELETE_SECRET = 'delete_secret'
    LIST_SECRETS = 'list_secrets'


SERVICE_OPERATIONS = frozenset(ServiceOperationType)
