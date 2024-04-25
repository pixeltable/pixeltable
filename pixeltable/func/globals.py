import inspect


def get_caller_module_path() -> str:
    """Return the module path of our caller's caller"""
    stack = inspect.stack()
    try:
        caller_frame = stack[2].frame
        module_path = caller_frame.f_globals['__name__']
    finally:
        # remove references to stack frames to avoid reference cycles
        del stack
    return module_path
