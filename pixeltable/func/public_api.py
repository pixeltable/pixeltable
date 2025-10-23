"""
Public API decorator for Pixeltable.

Marks functions/classes/methods as public API and automatically:
- Extracts type information and signatures
- Generates Pydantic models for FastAPI compatibility
- Registers for documentation generation
- Provides introspection capabilities

Usage:
    @public_api
    def create_table(name: str, schema: dict) -> Table:
        '''Create a new table.'''
        ...
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, TypeVar, get_type_hints

T = TypeVar('T')

# Global registry of all public API elements
_PUBLIC_API_REGISTRY: dict[str, dict[str, Any]] = {}


def public_api(obj: T) -> T:
    """
    Decorator to mark a function, method, or class as public API.

    Automatically extracts:
    - Full signature with types and defaults
    - Docstring
    - Parameter information
    - Return type

    The decorator can be applied to:
    - Functions
    - Methods
    - Classes
    - UDFs (works with @udf decorator)

    Examples:
        >>> @public_api
        ... def create_table(name: str, schema: Optional[dict] = None) -> Table:
        ...     '''Create a new table.'''
        ...     ...

        >>> class Table:
        ...     @public_api
        ...     def insert(self, **kwargs: Any) -> UpdateStatus:
        ...         '''Insert rows.'''
        ...         ...

        >>> @public_api
        ... @udf
        ... def my_function(x: int) -> int:
        ...     '''My UDF.'''
        ...     return x + 1
    """
    # Mark object as public API
    obj.__public_api__ = True  # type: ignore

    # Get qualified name for registry
    module = getattr(obj, '__module__', None)
    qualname = getattr(obj, '__qualname__', getattr(obj, '__name__', str(obj)))

    if module:
        full_qualname = f"{module}.{qualname}"
    else:
        full_qualname = qualname

    # Extract signature and type information
    try:
        signature = inspect.signature(obj)
        type_hints = get_type_hints(obj)
    except (ValueError, TypeError, AttributeError, NameError):
        # Some objects (like UDFs) may not have standard signatures
        # or get_type_hints may fail due to circular imports or undefined names
        signature = None
        type_hints = {}

    # Extract parameter information
    parameters = {}
    if signature:
        for param_name, param in signature.parameters.items():
            param_info = {
                'name': param_name,
                'annotation': type_hints.get(param_name, param.annotation),
                'default': param.default if param.default != inspect.Parameter.empty else None,
                # Kind: POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, VAR_KEYWORD
                'kind': param.kind.name,
            }
            parameters[param_name] = param_info

    # Extract return type
    return_type = type_hints.get('return', signature.return_annotation if signature else None)
    if return_type == inspect.Signature.empty:
        return_type = None

    # Register in global registry
    _PUBLIC_API_REGISTRY[full_qualname] = {
        'object': obj,
        'module': module,
        'name': qualname.split('.')[-1],
        'qualname': qualname,
        'full_qualname': full_qualname,
        'signature': signature,
        'parameters': parameters,
        'return_type': return_type,
        'docstring': inspect.getdoc(obj),
        'is_class': inspect.isclass(obj),
        'is_function': inspect.isfunction(obj) or inspect.ismethod(obj),
    }

    return obj


def get_public_api_registry() -> dict[str, dict[str, Any]]:
    """
    Get the complete public API registry.

    Returns:
        Dictionary mapping qualified names to API metadata
    """
    return _PUBLIC_API_REGISTRY.copy()


def is_public_api(obj: Any) -> bool:
    """
    Check if an object is marked as public API.

    Args:
        obj: Object to check

    Returns:
        True if marked as public API, False otherwise
    """
    return getattr(obj, '__public_api__', False)


def get_pydantic_models(func: Callable) -> tuple[Optional[type], Optional[type]]:
    """
    Generate Pydantic models for a function's input and output.

    This enables FastAPI integration by converting function signatures
    to Pydantic models that can validate requests/responses.

    Args:
        func: Function to generate models for

    Returns:
        Tuple of (InputModel, OutputModel) or (None, None) if generation fails
    """
    try:
        from pydantic import create_model
    except ImportError:
        return None, None

    # Get metadata from registry
    qualname = getattr(func, '__qualname__', getattr(func, '__name__', None))
    if not qualname:
        return None, None

    full_qualname = f"{func.__module__}.{qualname}" if hasattr(func, '__module__') else qualname
    metadata = _PUBLIC_API_REGISTRY.get(full_qualname)

    if not metadata:
        return None, None

    # Create input model from parameters
    input_fields = {}
    for param_name, param_info in metadata['parameters'].items():
        if param_name == 'self':
            continue

        annotation = param_info['annotation']
        default = param_info['default']

        # Handle missing annotations
        if annotation == inspect.Parameter.empty:
            annotation = Any

        # Create field
        if default is None:
            input_fields[param_name] = (annotation, ...)  # Required field
        else:
            input_fields[param_name] = (annotation, default)

    # Create Pydantic input model
    input_model_name = f"{metadata['name']}_Input"
    input_model = create_model(input_model_name, **input_fields) if input_fields else None

    # Create output model from return type
    return_type = metadata['return_type']
    if return_type and return_type != inspect.Signature.empty:
        output_model_name = f"{metadata['name']}_Output"
        output_model = create_model(output_model_name, result=(return_type, ...))
    else:
        output_model = None

    return input_model, output_model
