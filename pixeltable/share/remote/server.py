from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError, create_model

from pixeltable import catalog

# from .remote_schema_objects import RemoteDir, RemoteTable  # Imported locally when needed
from .utils import convert_local_path_to_remote

# Configure logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('pixeltable.local_server')


# Removed unused TypeAdapter code - using simplified conversion instead


def _serialize_dir_contents(dir_contents: Any) -> dict:
    """Convert DirContents (NamedTuple) to remote format"""
    if hasattr(dir_contents, 'tables') and hasattr(dir_contents, 'dirs'):
        return {
            'tables': [convert_local_path_to_remote(path) for path in dir_contents.tables],
            'dirs': [convert_local_path_to_remote(path) for path in dir_contents.dirs],
        }
    return dir_contents


def _convert_local_table_to_remote(obj: catalog.Table) -> dict[str, str]:
    """Convert local Table to remote format."""
    remote_path = convert_local_path_to_remote(obj.path)
    return {'remote_table_path': remote_path}


def _convert_local_dir_to_remote(obj: catalog.Dir) -> dict[str, str]:
    """Convert local Dir to remote format."""
    remote_path = convert_local_path_to_remote(obj._path())
    return {'remote_dir_path': remote_path}


def _convert_local_pandas_dataframe_to_remote(obj: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert local pandas DataFrame to remote format."""
    return obj.to_dict('records')  # type: ignore


def _serialize_path_list(path_list: list) -> list:
    """Convert list of paths to remote format"""
    if isinstance(path_list, list):
        return [convert_local_path_to_remote(path) if isinstance(path, str) else path for path in path_list]
    return path_list


# Removed unused TypeAdapter functions - using simplified conversion instead


def convert_local_result_to_remote(result: Any, func: Any) -> Any:
    """Simplified conversion using singledispatch."""
    if result is None:
        return None

    # Handle pandas DataFrame (from ls function)
    if isinstance(result, pd.DataFrame):
        return result.to_dict('records')

    # Use the consolidated dispatcher
    if isinstance(result, (catalog.Dir, catalog.Table)):
        from .utils import serialize_to_remote

        return serialize_to_remote(result)
    elif isinstance(result, list):
        # Check if this is a list of path strings (from list_dirs, list_tables, etc.)
        if result and isinstance(result[0], str) and not result[0].startswith('pxt://'):
            # Convert list of local paths to remote paths
            from .utils import convert_local_path_to_remote

            return [convert_local_path_to_remote(path) for path in result]
        else:
            return [convert_local_result_to_remote(item, func) for item in result]
    elif hasattr(result, 'tables') and hasattr(result, 'dirs'):  # DirContents
        from .utils import convert_local_path_to_remote

        return {
            'tables': [convert_local_path_to_remote(path) for path in result.tables],
            'dirs': [convert_local_path_to_remote(path) for path in result.dirs],
        }
    return result


def _convert_local_result_to_remote_fallback(result: Any, func_name: str) -> Any:
    """Fallback to function-name-based conversion."""
    # Handle specific function return types
    if func_name in ['create_table', 'create_view', 'create_snapshot', 'get_table']:
        return _convert_local_table_to_remote(result)
    elif func_name == 'create_dir':
        return _convert_local_dir_to_remote(result)
    elif func_name in ['list_tables', 'list_dirs']:
        return _convert_local_path_list_to_remote(result)
    elif func_name == 'get_dir_contents':
        return _convert_local_dir_contents_to_remote(result)
    elif func_name == 'ls':
        return _convert_local_pandas_dataframe_to_remote(result)
    else:
        # For functions that return None or other simple types
        return result


def _convert_local_path_list_to_remote(result: list[str]) -> list[str]:
    """Convert list of local paths to remote paths."""
    if isinstance(result, list):
        return [convert_local_path_to_remote(path) for path in result]
    return result


def _convert_local_dir_contents_to_remote(result: Any) -> Any:
    """Convert DirContents to remote format."""
    # DirContents is a NamedTuple with tables and dirs lists
    if hasattr(result, 'tables') and hasattr(result, 'dirs'):
        # Convert table paths
        remote_tables = [convert_local_path_to_remote(path) for path in result.tables]
        # Convert dir paths
        remote_dirs = [convert_local_path_to_remote(path) for path in result.dirs]
        # Return as dict to preserve structure
        return {'tables': remote_tables, 'dirs': remote_dirs}
    return result


def _convert_remote_json_to_dir_contents(json_result: Any) -> Any:
    """Convert JSON result to DirContents format."""
    # Server converts DirContents to dict with 'tables' and 'dirs' keys
    # Both lists contain remote paths already
    if isinstance(json_result, dict) and 'tables' in json_result and 'dirs' in json_result:
        return json_result
    # Fallback for other formats
    return json_result


app = FastAPI(title='Pixeltable Local Server', version='1.0.0')


class RemoteRequest(BaseModel):
    """Request model for remote function calls."""

    operation_type: str
    args: dict


def create_remote_response_model(return_type: Any = Any) -> type[BaseModel]:
    """Create a dynamic RemoteResponse model with proper return type handling."""
    return create_model(
        'RemoteResponse',
        success=(bool, ...),
        result=(return_type, None),
        error=(Optional[str], None),  # Fix: Use Optional[str] instead of str
    )


# Default response model for when we don't know the return type
RemoteResponse = create_remote_response_model()


class RemoteServer:
    """Server that handles remote function execution using the pixeltable module."""

    def __init__(self) -> None:
        from .utils import ModelCache

        self.model_cache = ModelCache()
        _logger.info('RemoteServer initialized - assumes pixeltable module is already configured')

    def execute_function(self, request: RemoteRequest) -> Any:
        """
        Execute a registered function with the provided arguments.

        Args:
            request: The remote request containing operation_type and args

        Returns:
            RemoteResponse with success status and result or error
        """
        operation_type = request.operation_type
        args = request.args

        # Check if we have a model for this function (means it's a @remote function)
        if operation_type not in self.model_cache._cache:
            return RemoteResponse(
                success=False, error=f"Function '{operation_type}' not found or not a @remote function"
            )

        # Get the function from the preloaded cache
        func = self.model_cache.get_function(operation_type)

        try:
            # Convert remote paths to local paths in arguments
            converted_args = self._convert_remote_paths_in_args(args)

            # Get the field mapping and map original field names to model field names
            field_mapping = self.model_cache.get_field_mapping(operation_type)

            # Create reverse mapping from original param names to model field names
            reverse_field_mapping = {v: k for k, v in field_mapping.items()}

            # Map arguments to model field names for validation
            model_args = {}
            for original_param, value in converted_args.items():
                model_field = reverse_field_mapping.get(original_param, original_param)
                model_args[model_field] = value

            # Get the preloaded model for validation
            arg_model = self.model_cache._cache[operation_type]
            validated_args = arg_model(**model_args)

            # Instead of calling model_dump(), directly access the validated data
            # model_dump() would re-serialize the data, undoing the deserialization
            validated_kwargs = {}
            for field_name in validated_args.__class__.model_fields:
                validated_kwargs[field_name] = getattr(validated_args, field_name)

            # Map field names back to original parameter names for function call
            original_kwargs = {}
            for model_field, value in validated_kwargs.items():
                original_param = field_mapping.get(model_field, model_field)
                original_kwargs[original_param] = value

            _logger.info(f'Executing {operation_type} with args: {original_kwargs}')

            # Execute the function
            result = func(**original_kwargs)

            # Convert result to remote format
            converted_result = convert_local_result_to_remote(result, func)

            # Use the response model (which now always uses Any for result)
            try:
                response_model = self.model_cache.get_response_model(operation_type)
                return response_model(success=True, result=converted_result)
            except Exception as e:
                _logger.warning(f'Could not use response model for {operation_type}: {e}')
                # Fallback to basic response
                return RemoteResponse(success=True, result=converted_result)

        except ValidationError as e:
            _logger.error(f'Validation error for {operation_type}: {e}')
            return RemoteResponse(success=False, error=f'Validation error: {e!s}')
        except Exception as e:
            _logger.error(f'Execution error for {operation_type}: {e}')
            return RemoteResponse(success=False, error=f'Execution error: {e!s}')

    def _convert_remote_paths_in_args(self, args: dict) -> dict:
        """Convert remote paths in arguments to local paths. RemoteTable/RemoteDir objects are passed through."""
        from pixeltable import get_table

        from .remote_schema_objects import RemoteDir, RemoteTable
        from .utils import _deserialize_table_or_dataframe, convert_remote_path_to_local

        # Import DataFrame only when needed to avoid circular imports
        try:
            from pixeltable.dataframe import DataFrame as pixeltable_dataframe  # noqa: N813
        except ImportError:
            pixeltable_dataframe = type(None)  # type: ignore  # Safe fallback

        # Use a wide type to accommodate strings, Tables, and other objects after conversion
        converted_args: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith('pxt://'):
                # Convert string paths to local paths
                converted_args[key] = convert_remote_path_to_local(value)
            elif isinstance(value, dict) and 'remote_table_path' in value:
                # Deserialize dict to RemoteTable, then convert to local Table
                remote_table = _deserialize_table_or_dataframe(value)
                local_path = convert_remote_path_to_local(remote_table.path)
                converted_args[key] = get_table(local_path)
            elif isinstance(value, dict) and 'remote_dir_path' in value:
                # Deserialize dict to RemoteDir, then convert to local path
                remote_dir = RemoteDir(path=value['remote_dir_path'])
                local_path = convert_remote_path_to_local(remote_dir.path)
                converted_args[key] = local_path
            elif isinstance(value, RemoteTable):
                # Convert RemoteTable to local Table object
                local_path = convert_remote_path_to_local(value.path)
                converted_args[key] = get_table(local_path)
            elif isinstance(value, RemoteDir):
                # Convert RemoteDir to local path string (most functions just need the path)
                local_path = convert_remote_path_to_local(value.path)
                converted_args[key] = local_path
            elif pixeltable_dataframe is not type(None) and isinstance(value, pixeltable_dataframe):
                # Pixeltable DataFrames are not supported in remote calls
                raise ValueError(
                    'Pixeltable DataFrames are not supported in remote calls. Use RemoteTable or catalog.Table instead.'
                )
            else:
                converted_args[key] = value

        return converted_args


# Global server instance - created lazily to ensure pixeltable module is initialized
_remote_server: RemoteServer | None = None


def get_remote_server() -> RemoteServer:
    """Get the global remote server instance, creating it if necessary."""
    global _remote_server  # noqa: PLW0603
    if _remote_server is None:
        _remote_server = RemoteServer()
    return _remote_server


@app.get('/health')
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {'status': 'healthy', 'message': 'Pixeltable Remote Server is running'}


@app.post('/api/execute')
async def execute_remote_function(request: RemoteRequest) -> Any:
    """Execute a remote function call."""
    remote_server = get_remote_server()
    return remote_server.execute_function(request)
