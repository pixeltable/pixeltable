from __future__ import annotations

import logging
from typing import Any, Callable

import requests

from . import utils as remote_utils
from .remote_schema_objects import RemoteDir, RemoteTable

_logger = logging.getLogger('pixeltable.remote_client')


class RemoteClient:
    """HTTP client for making remote calls to Pixeltable functions."""

    _base_url: str
    _timeout: int
    _model_cache: remote_utils.ModelCache

    def __init__(self, base_url: str = 'http://localhost:8000', timeout: int = 30):
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout
        self._model_cache = remote_utils.ModelCache()

    def make_remote_call(self, func: Callable, **kwargs: Any) -> Any:
        """
        Make a remote call by serializing arguments using Pydantic.

        Args:
            func: The function to call remotely
            **kwargs: Function arguments

        Returns:
            Result from remote server

        Raises:
            ConnectionError: If remote call fails
            RuntimeError: If remote operation fails
        """
        _logger.debug(f'Making remote call to {func.__name__} with {len(kwargs)} arguments')

        try:
            # Get the model for this function to use its custom serializers
            arg_model = self._model_cache.get_model(func)

            # Get field mapping to convert original param names to model field names
            field_mapping = self._model_cache.get_field_mapping(func.__name__)
            reverse_field_mapping = {v: k for k, v in field_mapping.items()}

            # Map arguments to model field names for validation
            model_args = {}
            for original_param, value in kwargs.items():
                # Check for unsupported Pixeltable DataFrames
                try:
                    from pixeltable.dataframe import DataFrame as PixeltableDataFrame

                    if isinstance(value, PixeltableDataFrame):
                        raise ValueError(
                            'Pixeltable DataFrames are not supported in remote calls. '
                            'Use RemoteTable or catalog.Table instead.'
                        )
                except ImportError:
                    pass  # DataFrame class not available
                model_field = reverse_field_mapping.get(original_param, original_param)
                model_args[model_field] = value

            # Validate and serialize the arguments using the model
            validated_args = arg_model(**model_args)
            serialized_args = validated_args.model_dump()

            # Debug logging
            _logger.debug(f'Model validation successful for {func.__name__}')
            _logger.debug(f'Serialized args: {serialized_args}')

            # Map field names back to original parameter names for server
            original_args = {}
            for model_field, value in serialized_args.items():
                original_param = field_mapping.get(model_field, model_field)
                original_args[original_param] = value

            # Create request
            request_data = {'operation_type': func.__name__, 'args': original_args}

            # Make HTTP request
            response = requests.post(
                f'{self._base_url}/api/execute',
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=self._timeout,
            )
            response.raise_for_status()

            # Parse the response
            response_data = response.json()

            if not response_data.get('success', False):
                raise RuntimeError(f'Remote call failed: {response_data.get("error", "Unknown error")}')

            result = response_data.get('result')

            # **NEW: Deserialize the result using registry-based conversion**
            deserialized_result = self._deserialize_result(result, func)

            return deserialized_result

        except requests.RequestException as e:
            _logger.error(f'Connection error for {func.__name__}: {e}')
            raise ConnectionError(f'Failed to connect to remote endpoint: {e}') from e
        except Exception as e:
            _logger.error(f'Model validation failed for {func.__name__}: {e}')
            raise ValueError(f'Invalid arguments for {func.__name__}: {e}') from e

    def _deserialize_result(self, result: Any, func: Callable) -> Any:
        """
        Deserialize the result using registry-based conversion to convert
        server response data back to client objects (RemoteTable, RemoteDir, etc.)
        """
        # Special handling for ls function - convert list of dicts back to DataFrame
        if func.__name__ == 'ls' and isinstance(result, list) and result and isinstance(result[0], dict):
            import pandas as pd

            return pd.DataFrame(result)

        return self._deserialize_value_recursive(result)

    def _deserialize_value_recursive(self, value: Any) -> Any:
        """Optimized recursive deserialization using pattern matching."""
        if value is None:
            return None

        if isinstance(value, dict):
            # Fast pattern matching - no loops
            if 'remote_dir_path' in value:
                return RemoteDir(path=value['remote_dir_path'])
            elif 'remote_table_path' in value:
                return RemoteTable(path=value['remote_table_path'])
            elif len(value) == 1:  # Potential serialized ColumnType
                return self._try_column_type_deserialize(value)
            else:
                # Recursively process nested dicts
                return {k: self._deserialize_value_recursive(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._deserialize_value_recursive(item) for item in value]

        return value

    def _try_column_type_deserialize(self, value: dict) -> Any:
        """Try ColumnType deserialization only for single-key dicts."""
        try:
            from .utils import _deserialize_column_type

            return _deserialize_column_type(value)
        except Exception:
            return value
