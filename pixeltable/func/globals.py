import importlib


def resolve_symbol(module_name: str, symbol: str) -> object:
    module = importlib.import_module(module_name)
    obj = module
    for el in symbol.split('.'):
        obj = getattr(obj, el)
    return obj
