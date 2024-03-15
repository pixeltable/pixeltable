from typing import Tuple, List, Optional
import types
import sys

import pixeltable.func as func
import pixeltable.type_system as ts
import pixeltable.env as env


def create_together_module() -> types.ModuleType:
    """Create module pixeltable.functions.together and populate it with functions for the Together functions"""
    module_name = 'pixeltable.functions.together'
    pt_module = types.ModuleType(module_name)
    pt_module.__package__ = 'pixeltable.functions'
    sys.modules[module_name] = pt_module
    specs = [
        func.TogetherFunctionSpec(
            name='completion',
            call=['Complete', 'create'],
            params={
                'prompt': ts.StringType(),
                'model': ts.StringType(),
                'max_tokens': ts.IntType(nullable=True),
                'repetition_penalty': ts.FloatType(nullable=True),
                'response_format': ts.JsonType(nullable=True),
                'seed': ts.IntType(nullable=True),
                'stop': ts.JsonType(nullable=True),
                'top_k': ts.IntType(nullable=True),
                'top_p': ts.FloatType(nullable=True),
                'temperature': ts.FloatType(nullable=True),
            },
            batch_params=[],
            output_path=None,
            output_type=ts.JsonType(nullable=False),
        ),
    ]

    for spec in specs:
        fn = func.TogetherFunction(spec, module_name=module_name)
        setattr(pt_module, spec.name, fn)

    return pt_module

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
        pt_func = func.NOSFunction(info, module_name)
        setattr(sub_module, model_id, pt_func)

    return new_modules
