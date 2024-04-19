from typing import Tuple, List, Optional
import types
import sys

import pixeltable.func as func
import pixeltable.type_system as ts
import pixeltable.env as env


def create_nos_modules() -> List[types.ModuleType]:
    """Create module pixeltable.functions.nos with one submodule per task and return the submodules"""
    models = env.Env.get().nos_client.ListModels()
    model_info = [env.Env.get().nos_client.GetModelInfo(model) for model in models]
    model_info.sort(key=lambda info: info.task.value)

    module_name = 'pixeltable.functions.nos'
    nos_module = types.ModuleType(module_name)
    nos_module.__package__ = 'pixeltable.functions'
    sys.modules[module_name] = nos_module

    prev_task = ''
    new_modules: List[types.ModuleType] = []
    sub_module: Optional[types.ModuleType] = None
    for info in model_info:
        if info.task.value != prev_task:
            # we construct one submodule per task
            namespace = info.task.name.lower()
            submodule_name = f'{module_name}.{namespace}'
            sub_module = types.ModuleType(submodule_name)
            sub_module.__package__ = module_name
            setattr(nos_module, namespace, sub_module)
            new_modules.append(sub_module)
            sys.modules[submodule_name] = sub_module
            prev_task = info.task.value

        # add a Function for this model to the module
        model_id = info.name.replace("/", "_").replace("-", "_")
        pt_func = func.NOSFunction(info, f'{submodule_name}.{model_id}')
        setattr(sub_module, model_id, pt_func)

    return new_modules


def resolve_torch_device(device: str) -> str:
    import torch
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device
