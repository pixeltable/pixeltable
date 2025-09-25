"""
Simple utilities for remote function handling.
Shared between RemoteClient and RemoteServer.
"""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache, singledispatch
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from urllib.parse import urlparse
from weakref import WeakValueDictionary

import pandas as pd
from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer, create_model

from pixeltable import catalog, type_system as ts
from pixeltable.type_system import ALL_PIXELTABLE_TYPES, ColumnType, _PxtType

from .remote_schema_objects import RemoteDir, RemoteTable

_logger = logging.getLogger('pixeltable.remote_utils')


# Optimized path conversion with caching
@lru_cache(maxsize=512)
def convert_local_path_to_remote(local_path: str, server_name: str = 'local') -> str:
    """Convert local path to remote path format with caching."""
    if local_path.startswith('pxt://'):
        return local_path
    path_part = local_path.replace('.', '/')
    return f'pxt://{server_name}/{path_part}'


@lru_cache(maxsize=512)
def convert_remote_path_to_local(remote_path: str) -> str:
    """Convert remote path to local path format with caching."""
    if not remote_path.startswith('pxt://'):
        return remote_path
    parsed = urlparse(remote_path)
    path_part = parsed.path.lstrip('/')
    return path_part.replace('/', '.').lower() if path_part else ''


# Comprehensive list of Pydantic reserved field names
PYDANTIC_RESERVED = {
    'model_config',
    'model_fields',
    'model_extra',
    'model_computed_fields',
    'model_fields_set',
    'model_dump',
    'model_dump_json',
    'model_copy',
    'model_validate',
    'model_validate_json',
    'model_rebuild',
    'model_json_schema',
    'dict',
    'json',
    'parse_obj',
    'parse_raw',
    'parse_file',
    'from_orm',
    'schema',
    'schema_json',
    'construct',
    'copy',
    'update_forward_refs',
    '__config__',
    '__fields__',
    '__validators__',
    '__pre_root_validators__',
    '__post_root_validators__',
    '__schema_cache__',
    '__fields_set__',
    '__exclude_fields__',
    '__include_fields__',
    '__private_attributes__',
}


# Consolidated serialization using singledispatch
@singledispatch
def serialize_to_remote(obj: Any) -> Any:
    """Single dispatcher for all object-to-remote conversions."""
    return obj


@serialize_to_remote.register
def _(obj: catalog.Table) -> dict[str, str]:
    """Serialize catalog.Table to remote format."""
    path = obj.path if hasattr(obj, 'path') else obj._path()
    return {'remote_table_path': convert_local_path_to_remote(path)}


@serialize_to_remote.register
def _(obj: catalog.Dir) -> dict[str, str]:
    """Serialize catalog.Dir to remote format."""
    path = obj.path if hasattr(obj, 'path') else obj._path()
    return {'remote_dir_path': convert_local_path_to_remote(path)}


@serialize_to_remote.register
def _(obj: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize pandas DataFrame to remote format."""
    return obj.to_dict('records')  # type: ignore


@serialize_to_remote.register
def _(obj: RemoteTable) -> dict[str, str]:
    """Serialize RemoteTable to remote format."""
    return {'remote_table_path': obj.path}


@serialize_to_remote.register
def _(obj: RemoteDir) -> dict[str, str]:
    """Serialize RemoteDir to remote format."""
    return {'remote_dir_path': obj.path}


# Custom serializers and validators for Pixeltable types
def _serialize_column_type(obj: ColumnType | _PxtType | Annotated[Any, ColumnType]) -> dict[str, Any]:
    """Serialize ColumnType or _PxtType objects using as_dict() or to_json_schema()."""
    # Handle ColumnType directly
    if isinstance(obj, ColumnType):
        try:
            return obj.to_json_schema()
        except Exception:
            # Fallback to as_dict for types that don't support JSON schema (like Image)
            return obj.as_dict()

    # Handle _PxtType objects (like pxt.Int, pxt.String)
    if obj in ALL_PIXELTABLE_TYPES:
        col_type = obj.as_col_type(nullable=False)  # Fix: pass required nullable parameter
        try:
            return col_type.to_json_schema()
        except Exception:
            # Fallback to as_dict for types that don't support JSON schema (like Image)
            return col_type.as_dict()

    raise ValueError(f'Expected ColumnType or _PxtType, got {type(obj)}')


def _deserialize_column_type(value: dict[str, Any]) -> Any:
    """Deserialize ColumnType objects using from_json_schema() or from_dict()."""
    try:
        # First try JSON schema format
        result = ColumnType.from_json_schema(value)
        if result is not None:
            return result
    except Exception:
        pass

    # Fallback to dict format for complex types like Image
    return ColumnType.from_dict(value)


def _deserialize_table(value: dict[str, str]) -> Any:
    """Deserialize Table objects - convert remote path strings to RemoteTable objects."""
    if isinstance(value, dict) and 'remote_table_path' in value:
        return RemoteTable(path=value['remote_table_path'])
    return value


def _deserialize_remote_dir(value: dict[str, str]) -> Any:
    """Deserialize Dir objects - convert remote path strings to RemoteDir objects."""
    if isinstance(value, dict) and 'remote_dir_path' in value:
        return RemoteDir(path=value['remote_dir_path'])
    return value


def _deserialize_table_or_dataframe(value: Any) -> Any:
    """Deserialize Table or DataFrame - convert remote path strings to RemoteTable objects."""
    if isinstance(value, str) and value.startswith('pxt://'):
        # Convert remote path string to RemoteTable
        return RemoteTable(path=value)
    elif isinstance(value, dict) and 'remote_table_path' in value:
        return RemoteTable(path=value['remote_table_path'])
    elif isinstance(value, dict):
        # Could be a serialized Pixeltable DataFrame - try to deserialize it
        try:
            from pixeltable.dataframe import DataFrame as PixeltableDataFrame

            return PixeltableDataFrame.from_dict(value)
        except Exception:
            # Not a DataFrame, return as-is
            return value
    return value


# Replace duplicate functions with consolidated singledispatch version
_serialize_table = serialize_to_remote
_serialize_remote_dir = serialize_to_remote
_convert_local_table_to_remote = serialize_to_remote
_convert_local_dir_to_remote = serialize_to_remote


def _serialize_pandas_dataframe(obj: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize pandas DataFrame - raises NotImplementedError for now."""
    return obj.to_dict('records')  # type: ignore


def _deserialize_pandas_dataframe(value: list[dict[str, Any]]) -> pd.DataFrame:
    """Deserialize pandas DataFrame - raises NotImplementedError for now."""
    return pd.DataFrame(value)


def _serialize_dict_with_pixeltable_types(obj: dict) -> dict:
    """Recursively serialize dict values that contain Pixeltable types."""
    if not isinstance(obj, dict):
        return obj

    result = {}
    for key, value in obj.items():
        # Special handling for typing.Annotated Pixeltable types
        if hasattr(value, '__origin__') and hasattr(value, '__metadata__'):
            # This is a typing.Annotated type like pxt.Int, pxt.String, etc.
            # Extract the underlying ColumnType from metadata
            try:
                # Extract the ColumnType from the __metadata__ tuple
                metadata = value.__metadata__
                if metadata and len(metadata) > 0:
                    column_type = metadata[0]  # First item in metadata should be the ColumnType
                    if hasattr(column_type, 'to_json_schema'):
                        try:
                            result[key] = column_type.to_json_schema()
                        except Exception:
                            # Fallback to as_dict for types that don't support JSON schema (like Image)
                            if hasattr(column_type, 'as_dict'):
                                result[key] = column_type.as_dict()
                            else:
                                result[key] = str(value)
                    else:
                        # Fallback: use the string representation
                        result[key] = str(value)
                else:
                    # Fallback: use the string representation
                    result[key] = str(value)
            except Exception:
                # Fallback: use the string representation
                result[key] = str(value)
        else:
            # Check if value is a standard Pixeltable type that needs serialization
            serialized = False
            for pxt_type, (serializer, _) in CUSTOMIZED_FIELD_REGISTRY.items():
                if pxt_type is not dict:  # Avoid infinite recursion
                    try:
                        # Check for direct equality (handles class types like ts.Image)
                        if value == pxt_type or isinstance(value, pxt_type):
                            result[key] = serializer(value)
                            serialized = True
                            break
                    except TypeError:
                        # Skip typing.Annotated types that can't be used with isinstance
                        continue

            if not serialized:
                # Recursively handle nested dicts
                if isinstance(value, dict):
                    result[key] = _serialize_dict_with_pixeltable_types(value)
                else:
                    result[key] = value
    return result


def _deserialize_dict_with_pixeltable_types(value: dict) -> dict:
    """Recursively deserialize dict values that contain Pixeltable types."""
    if not isinstance(value, dict):
        return value

    result = {}
    for key, val in value.items():
        # Try to deserialize as Pixeltable types
        if isinstance(val, dict):
            # Could be a serialized Pixeltable type - try each deserializer
            for pxt_type, (_, deserializer) in CUSTOMIZED_FIELD_REGISTRY.items():
                if pxt_type is not dict:  # Avoid infinite recursion
                    try:
                        result[key] = deserializer(val)
                        break
                    except (ValueError, KeyError, TypeError):
                        continue
            else:
                # Not a Pixeltable type, recursively process
                result[key] = _deserialize_dict_with_pixeltable_types(val)
        else:
            result[key] = val
    return result


def _build_pixeltable_type_registry() -> dict[Any, tuple[Callable[[Any], Any], Callable[[Any], Any]]]:
    """Build registry mapping Pixeltable types to their serializers/validators."""
    registry: dict[Any, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {}

    # Add ColumnType
    registry[ts.ColumnType] = (_serialize_column_type, _deserialize_column_type)
    for pxt_type in ts.ALL_PIXELTABLE_TYPES:
        registry[pxt_type] = (_serialize_column_type, _deserialize_column_type)
    # Add RemoteTable and RemoteDir (for client-side handling)
    registry[RemoteTable] = (_serialize_table, _deserialize_table)
    registry[RemoteDir] = (_serialize_remote_dir, _deserialize_remote_dir)
    # Add base catalog types (for server-side conversion)
    registry[catalog.Table] = (_serialize_table, _deserialize_table)
    registry[catalog.Dir] = (_serialize_remote_dir, _deserialize_remote_dir)
    # Add pandas DataFrame
    registry[pd.DataFrame] = (_serialize_pandas_dataframe, _deserialize_pandas_dataframe)

    # Add Union type handling for Table | DataFrame (Pixeltable's DataFrame, not pandas)
    from typing import Union as TypingUnion

    from pixeltable.dataframe import DataFrame as PixeltableDataFrame

    table_dataframe_union = TypingUnion[catalog.Table, PixeltableDataFrame]
    registry[table_dataframe_union] = (_serialize_table, _deserialize_table_or_dataframe)

    return registry


# Global registry for customized fields - defined after functions to avoid circular references
CUSTOMIZED_FIELD_REGISTRY: dict[Any, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = (
    _build_pixeltable_type_registry()
)

# Add dict handler to registry after functions are defined
CUSTOMIZED_FIELD_REGISTRY[dict] = (_serialize_dict_with_pixeltable_types, _deserialize_dict_with_pixeltable_types)


def _find_pixeltable_types_recursive(field_type: Any) -> tuple[Optional[Callable], Optional[Callable]]:
    """
    Recursively find Pixeltable types within complex type structures using registry.
    Only used for types that don't have native Pydantic support.

    Args:
        field_type: The type to analyze

    Returns:
        Tuple of (serializer_func, validator_func) or (None, None) if no Pixeltable types found
    """
    # Direct type match in registry
    if field_type in CUSTOMIZED_FIELD_REGISTRY:
        return CUSTOMIZED_FIELD_REGISTRY[field_type]

    # Handle generic types
    origin = get_origin(field_type)
    args = get_args(field_type)

    # **SPECIAL CASE: Handle Optional[T] when get_origin returns None**
    # This happens with typing.Optional in some Python versions
    field_type_str = str(field_type)
    if (
        origin is None
        and ('Optional[' in field_type_str or 'typing.Union[' in field_type_str)
        and 'dict[str, Any]' in field_type_str
    ):
        # For Optional[dict[str, Any]], we want to find the dict serializer
        return CUSTOMIZED_FIELD_REGISTRY[dict]

    # **NEW: Handle base generic types**
    # For dict[str, Any] -> match with dict in registry
    if origin is not None and origin in CUSTOMIZED_FIELD_REGISTRY:
        return CUSTOMIZED_FIELD_REGISTRY[origin]

    # Handle Union types (including Optional[T])
    if origin is Union:
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return _find_pixeltable_types_recursive(non_none_type)

        # Check if any of the Union types are in the registry
        for arg in args:
            if arg is not type(None):
                if arg in CUSTOMIZED_FIELD_REGISTRY:
                    return CUSTOMIZED_FIELD_REGISTRY[arg]
                serializer, validator = _find_pixeltable_types_recursive(arg)
                if serializer and validator:
                    return serializer, validator

    # Handle List[T], Set[T], Tuple[T, ...] - check inner types
    elif origin in (list, set, tuple):
        if args:
            return _find_pixeltable_types_recursive(args[0])

    # Handle Dict[K, V] - check both key and value types
    elif origin is dict:
        if len(args) >= 2:
            # Check value type first (more likely to contain Pixeltable types)
            serializer, validator = _find_pixeltable_types_recursive(args[1])
            if serializer and validator:
                return serializer, validator
            # Also check key type
            return _find_pixeltable_types_recursive(args[0])

    # Handle other generic types
    elif origin is not None and args:
        for arg in args:
            serializer, validator = _find_pixeltable_types_recursive(arg)
            if serializer and validator:
                return serializer, validator

    return None, None


def _replace_catalog_types_with_remote_types(return_type: Any) -> Any:
    """Replace catalog.Dir and catalog.Table with RemoteDir and RemoteTable in return type annotations."""
    if return_type == catalog.Dir:
        return RemoteDir
    elif return_type == catalog.Table:
        return RemoteTable

    # Handle Optional[T] -> Optional[RemoteT]
    origin = get_origin(return_type)
    args = get_args(return_type)

    if origin is Union:  # Optional[T] is Union[T, None]
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            remote_type = _replace_catalog_types_with_remote_types(non_none_type)
            return Union[remote_type, type(None)]
        else:
            # For other Union types, replace each argument
            new_args = [_replace_catalog_types_with_remote_types(arg) for arg in args]
            return Union[tuple(new_args)]

    # Handle List[T], Set[T], Tuple[T, ...]
    elif origin in (list, set, tuple):
        if args:
            inner_type = _replace_catalog_types_with_remote_types(args[0])
            if origin is list:
                return List[inner_type]  # type: ignore
            elif origin is set:
                return Set[inner_type]  # type: ignore
            elif origin is tuple:
                return Tuple[inner_type, ...]

    # Handle Dict[K, V]
    elif origin is dict and len(args) >= 2:
        key_type = _replace_catalog_types_with_remote_types(args[0])
        value_type = _replace_catalog_types_with_remote_types(args[1])
        return Dict[key_type, value_type]  # type: ignore

    # For other types, return as-is
    return return_type


def _create_pixeltable_customized_field(field_name: str, field_type: Any) -> tuple[Any, Any]:
    """
    Create a Pydantic field with Pixeltable customized fields handling.

    Priority:
    1. If type has native Pydantic support (__get_pydantic_core_schema__) -> use it
    2. Otherwise -> fall back to registry-based approach

    Args:
        field_name: Name of the field
        field_type: Type annotation for the field

    Returns:
        Tuple of (field_type_with_annotations, field_info)
    """

    # Debug logging for schema field
    _logger.debug(f'_create_pixeltable_customized_field called with field_name={field_name}, field_type={field_type}')

    # Check if type has native Pydantic support
    if hasattr(field_type, '__get_pydantic_core_schema__'):
        # Debug logging for schema field
        if 'schema' in str(field_type).lower():
            _logger.debug(f'Type has native Pydantic support: {field_type}')
        # Type handles itself natively - no custom annotations needed
        return field_type, Field(default=...)

    # Fall back to registry-based approach
    serializer_func, validator_func = _find_pixeltable_types_recursive(field_type)

    if serializer_func and validator_func:
        # Use registry-based custom serializer/validator
        annotated_type = Annotated[
            field_type, BeforeValidator(validator_func), PlainSerializer(serializer_func, return_type=dict)
        ]
        return annotated_type, Field(default=...)

    # Regular field - no special handling needed
    return field_type, Field(default=...)


# Path conversion functions are now defined above with @lru_cache


def is_remote_path(path: str) -> bool:
    """Check if a path is a remote path (starts with pxt://)"""
    return isinstance(path, str) and path.startswith('pxt://')


def create_pydantic_model_from_function(
    func: Callable, func_name: str | None = None
) -> tuple[type[BaseModel], dict[str, str]]:
    """
    Create a Pydantic model from a function's signature for argument validation.

    Args:
        func: The function to create a model for
        func_name: Optional name override (defaults to func.__name__)

    Returns:
        Tuple of (dynamically created Pydantic model class, field name mapping)
    """
    func_name = func_name or func.__name__
    signature = inspect.signature(func)

    # Get type hints
    try:
        type_hints = get_type_hints(func)
    except (NameError, AttributeError):
        type_hints = getattr(func, '__annotations__', {})

    fields = {}
    field_mapping = {}  # Maps model field names to original parameter names

    for name, param in signature.parameters.items():
        # Skip 'self' and 'cls' for methods
        if name in ['self', 'cls']:
            continue

        # Rename conflicting field names to avoid BaseModel attribute conflicts
        field_name = name
        if name in PYDANTIC_RESERVED:
            field_name = f'{name}_param'
            field_mapping[field_name] = name

        # Get the type annotation
        field_type = type_hints.get(name, param.annotation)
        if field_type == inspect.Parameter.empty:
            field_type = Any

        # Use the new create_pixeltable_field function for automatic type handling
        annotated_type, field_info = _create_pixeltable_customized_field(field_name, field_type)

        # Handle required vs optional parameters
        if param.default is inspect.Parameter.empty:
            fields[field_name] = (annotated_type, field_info)
        else:
            # Update field_info with the default value
            field_info.default = param.default
            fields[field_name] = (annotated_type, field_info)

    # Create the dynamic model
    model_name = f'{func_name}Args'
    # Create the model with proper configuration
    model_config = {'arbitrary_types_allowed': True, 'extra': 'forbid'}

    model = create_model(model_name, __config__=model_config, **fields)  # type: ignore[call-overload]

    # Set up the model's module for proper forward reference resolution
    if hasattr(model, '__module__'):
        model.__module__ = func.__module__

    return model, field_mapping


def discover_remote_functions() -> list[tuple[str, Callable[..., Any], str]]:
    """
    Discover @remote functions from pixeltable.globals, fallback to local globals.py for testing.

    Returns:
        List of tuples: (function_name, function, source_module)
    """
    functions: list[tuple[str, Callable[..., Any], str]] = []

    try:
        from pixeltable import globals as pxt_globals

        for name in dir(pxt_globals):
            obj = getattr(pxt_globals, name)

            if inspect.isfunction(obj) and getattr(obj, '_is_remote', False):
                functions.append((name, obj, 'pixeltable.globals'))

        _logger.info(f'Found {len(functions)} @remote functions in pixeltable.globals')
        return functions

    except ImportError:
        _logger.info('Could not import pixeltable.globals')
        # Return empty list if pixeltable.globals is not available
        return []
    return functions


class ModelCache:
    """
    Cache for Pydantic models created from function signatures.
    Preloads all models from @remote functions during initialization.
    """

    def __init__(self, cache_name: str = 'model_cache'):
        self.cache_name = cache_name
        self._cache: dict[str, type[BaseModel]] = {}
        self._response_cache: dict[str, type[BaseModel]] = {}  # Cache for response models
        # Auto-cleanup unused models
        self._weak_response_cache: WeakValueDictionary[str, type[BaseModel]] = WeakValueDictionary()
        self._functions: dict[str, Callable] = {}
        self._field_mappings: dict[str, dict[str, str]] = {}  # Maps function names to field mappings
        self._return_types: list[Any] = []
        self._union_return_type: Any = None
        self._preload_models()

    def _get_function_signature_hash(self, func_name: str) -> str:
        """Cache function signature hashes."""
        if func_name in self._functions:
            return str(hash(str(inspect.signature(self._functions[func_name]))))
        return func_name

    def _preload_models(self) -> None:
        """Preload all models from @remote functions."""
        functions = discover_remote_functions()
        return_types = []

        for func_name, func, source_module in functions:
            original_func = getattr(func, '_original_func', func)
            model, field_mapping = create_pydantic_model_from_function(original_func, func_name)

            # Rebuild model to resolve forward references like Optional and TableDataSource
            try:
                model.model_rebuild(
                    _types_namespace={
                        'Optional': Optional,
                        'TableDataSource': Any,  # Use Any since we don't need the actual type
                    }
                )
                _logger.debug(f'Successfully rebuilt model for {func_name}')
            except Exception as e:
                _logger.debug(f'Model rebuild failed for {func_name} (this is often OK since we use Any): {e}')
                # Continue anyway - models with Any types often don't need rebuilding

            self._cache[func_name] = model
            self._functions[func_name] = original_func
            self._field_mappings[func_name] = field_mapping

            # Collect return types
            try:
                type_hints = get_type_hints(original_func)
                return_type = type_hints.get('return', Any)
                if return_type != inspect.Parameter.empty:
                    return_types.append(return_type)
            except (NameError, AttributeError):
                pass

            _logger.debug(f'Preloaded model for {func_name} from {source_module}')

        # Create union type from all return types
        if return_types:
            self._return_types = return_types
            # For now, just use Any to avoid complex Union type issues
            self._union_return_type = Any

        _logger.info(f'Preloaded {len(functions)} models from @remote functions')

    def get_model(self, func: Callable, func_name: str | None = None) -> type[BaseModel]:
        """Get a preloaded Pydantic model for the given function."""
        func_name = func_name or func.__name__

        # Check cache first
        if func_name in self._cache:
            model = self._cache[func_name]
            # Model should already be rebuilt during preloading
            return model

        # If not found, it's not a @remote function
        raise ValueError(f"Function '{func_name}' not found in preloaded cache. Only @remote functions are supported.")

    def get_function(self, func_name: str) -> Callable:
        """Get a preloaded function by name."""
        if func_name not in self._functions:
            raise ValueError(
                f"Function '{func_name}' not found in preloaded cache. Only @remote functions are supported."
            )
        return self._functions[func_name]

    def get_response_model(self, func_name: str) -> type[BaseModel]:
        """Get cached response model for function name."""
        if func_name not in self._response_cache:
            # Create response model on demand
            if func_name not in self._functions:
                raise KeyError(f'Function {func_name} not found in cache. It may not be decorated with @remote.')
            try:
                response_fields = {
                    'success': (bool, Field(...)),
                    'result': (Any, Field(default=None)),
                    'error': (Optional[str], Field(default=None)),
                }

                model_config = {'arbitrary_types_allowed': True, 'extra': 'forbid'}
                response_model = create_model(f'{func_name}Response', **response_fields, __config__=model_config)  # type: ignore

                self._response_cache[func_name] = response_model

            except Exception as e:
                _logger.warning(f'Could not create response model for {func_name}: {e}')
                # Fallback to basic response model
                response_model = create_model(
                    f'{func_name}Response',
                    __config__={'arbitrary_types_allowed': True, 'extra': 'forbid'},
                    success=(bool, Field(...)),
                    result=(Any, Field(default=None)),
                    error=(Optional[str], Field(default=None)),
                )
                self._response_cache[func_name] = response_model

        return self._response_cache[func_name]

    def get_field_mapping(self, func_name: str) -> dict[str, str]:
        """Get the field mapping for a function (maps model field names to original parameter names)."""
        if func_name not in self._field_mappings:
            raise ValueError(
                f"Function '{func_name}' not found in preloaded cache. Only @remote functions are supported."
            )
        return self._field_mappings[func_name]

    def get_union_return_type(self) -> Any:
        """Get the union type of all return types from @remote functions."""
        return self._union_return_type

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache.clear()

    def cache_info(self) -> dict:
        """Get cache statistics."""
        return {
            'cache_name': self.cache_name,
            'cached_models': len(self._cache),
            'model_names': list(self._cache.keys()),
        }
