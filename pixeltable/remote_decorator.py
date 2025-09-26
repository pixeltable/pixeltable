"""
Remote decorator for Pixeltable functions.

This module provides the @remote decorator that enables functions to be called
remotely via HTTP requests.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, ParamSpec, TypeVar, Union, cast

from pixeltable.env import Env

P = ParamSpec('P')
T = TypeVar('T')

_logger = logging.getLogger('pixeltable')


def is_remote_path(path: str) -> bool:
    """Check if a path is a remote path (starts with pxt://)"""
    return isinstance(path, str) and path.startswith('pxt://')


def remote(path_params: list[str]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to mark functions as remotely callable.

    Args:
        path_params: List of parameter names that are paths (for validation)
    Returns:
        Decorator function
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        func_name = func.__name__
        _logger.debug(f'Registered {func_name} as remote function with path_params: {path_params}')

        @functools.wraps(func)  # This preserves the signature
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get the function signature to map positional args to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check only the specified path parameters
            remote_path_params = []
            local_path_params = []

            for param_name in path_params:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if isinstance(value, str):
                        if value.startswith('pxt://'):
                            remote_path_params.append(param_name)
                        else:
                            local_path_params.append(param_name)

            remote_count = len(remote_path_params)
            local_count = len(local_path_params)

            # Validation: ALL path params must be remote OR ALL must be local
            if remote_count > 0 and local_count > 0:
                raise ValueError(
                    f'Mixed remote and local paths not allowed. '
                    f'Remote path params: {remote_path_params}, Local path params: {local_path_params}'
                )

            # If no remote paths found, call the original function locally
            if remote_count == 0:
                _logger.debug(f'Local call to {func_name}')
                return func(*args, **kwargs)

            # Convert string paths to RemoteTable/RemoteDir objects for remote calls
            converted_args: dict[str, Any] = {}
            for param_name, value in bound_args.arguments.items():
                if param_name in remote_path_params and isinstance(value, str) and value.startswith('pxt://'):
                    # Convert string path to appropriate remote object based on function signature
                    sig = inspect.signature(func)
                    param_type = sig.parameters[param_name].annotation
                    # Check if this parameter expects a Table or Dir type
                    if hasattr(param_type, '__origin__') and param_type.__origin__ is Union:
                        # Handle Union types like catalog.Table | DataFrame
                        type_args = getattr(param_type, '__args__', ())
                        if any('Table' in str(arg) for arg in type_args):
                            from pixeltable.share.remote import RemoteTable

                            converted_args[param_name] = RemoteTable(path=value)
                        elif any('Dir' in str(arg) for arg in type_args):
                            from pixeltable.share.remote import RemoteDir

                            converted_args[param_name] = RemoteDir(path=value)
                        else:
                            converted_args[param_name] = value
                    elif 'Table' in str(param_type):
                        from pixeltable.share.remote import RemoteTable

                        converted_args[param_name] = RemoteTable(path=value)
                    elif 'Dir' in str(param_type):
                        from pixeltable.share.remote import RemoteDir

                        converted_args[param_name] = RemoteDir(path=value)
                    else:
                        converted_args[param_name] = value
                else:
                    converted_args[param_name] = value

            # Make the remote call using the client from environment
            _logger.debug(f'Remote call to {func_name} with path params: {remote_path_params}')
            client = Env.get().remote_client
            return client.make_remote_call(func, **converted_args)

        # Add marker attributes
        wrapper._is_remote = True  # type: ignore[attr-defined]
        wrapper._original_func = func  # type: ignore[attr-defined]

        # Force mypy to see this as the original function type
        return cast(Callable[P, T], wrapper)

    return decorator
