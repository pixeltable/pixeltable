"""
Public API decorator for Pixeltable.

Marks functions/classes/methods as public API and automatically:
- Extracts type information and signatures
- Registers for documentation generation
- Provides introspection capabilities
- Enables external tools and integrations

Usage:
    @public_api
    def create_table(name: str, schema: dict) -> Table:
        '''Create a new table.'''
        ...
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, TypeVar, get_type_hints

T = TypeVar('T')

# Global registry of all public API elements
_PUBLIC_API_REGISTRY: dict[str, dict[str, Any]] = {}


def _get_return_type_name(return_type: Any) -> Optional[str]:
    """
    Get a simple string representation of the return type.

    This is just for convenience - the actual type object is also stored
    in the registry, so external tools can introspect it themselves.
    """
    if return_type is None:
        return None

    return getattr(return_type, '__name__', str(return_type))


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

    # For CallableFunction (UDF) objects, use the underlying Python function's metadata
    from .callable_function import CallableFunction
    underlying_fn = None
    if isinstance(obj, CallableFunction):
        # Use the original Python function for metadata extraction
        underlying_fn = obj.py_fns[0]  # Use first signature's function
        module = getattr(underlying_fn, '__module__', None)
        qualname = getattr(underlying_fn, '__qualname__', getattr(underlying_fn, '__name__', str(underlying_fn)))
    else:
        # Get qualified name for registry from the object itself
        module = getattr(obj, '__module__', None)
        qualname = getattr(obj, '__qualname__', getattr(obj, '__name__', str(obj)))

    if module:
        full_qualname = f"{module}.{qualname}"
    else:
        full_qualname = qualname

    # Extract signature and type information
    # For UDFs, use the underlying Python function
    target_for_inspection = underlying_fn if underlying_fn else obj

    try:
        signature = inspect.signature(target_for_inspection)
        type_hints = get_type_hints(target_for_inspection)
    except (ValueError, TypeError, AttributeError, NameError):
        # Some objects may not have standard signatures
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

    # Extract source file information (for documentation links to GitHub)
    source_file = None
    source_line = None
    try:
        source_file = inspect.getfile(target_for_inspection)
        _, source_line = inspect.getsourcelines(target_for_inspection)
    except (TypeError, OSError):
        # Some objects (like built-ins or dynamically created) don't have source files
        pass

    # Get simple return type name for convenience
    return_type_name = _get_return_type_name(return_type)

    # Determine type
    is_class = inspect.isclass(obj)
    is_method = inspect.ismethod(obj) or (not is_class and '.' in qualname and qualname.split('.')[-2][0].isupper())
    is_function = inspect.isfunction(obj)

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
        'return_type_name': return_type_name,
        'docstring': inspect.getdoc(obj),
        'is_class': is_class,
        'is_method': is_method,
        'is_function': is_function,
        'source_file': source_file,
        'source_line': source_line,
    }

    return obj


@public_api
def get_public_api_registry() -> dict[str, dict[str, Any]]:
    """
    Get the complete public API registry.

    This is the directory of all Pixeltable public APIs with their metadata.
    Useful for building tools, documentation, LLM integrations, and more.

    Returns:
        Dictionary mapping qualified names to API metadata, including:
        - signature: Function signature
        - parameters: Parameter names, types, defaults
        - return_type: Return type annotation
        - docstring: Full documentation
        - module: Source module

    Example:
        >>> registry = get_public_api_registry()
        >>> print(f"Total APIs: {len(registry)}")
        >>> print(registry['pixeltable.globals.create_table']['signature'])
    """
    return _PUBLIC_API_REGISTRY.copy()


@public_api
def is_public_api(obj: Any) -> bool:
    """
    Check if an object is marked as public API.

    Args:
        obj: Object to check (function, class, method)

    Returns:
        True if marked as public API, False otherwise

    Example:
        >>> import pixeltable as pxt
        >>> is_public_api(pxt.create_table)
        True
        >>> is_public_api(some_internal_function)
        False
    """
    return getattr(obj, '__public_api__', False)
