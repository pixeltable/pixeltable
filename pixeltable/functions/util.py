from typing import Tuple, List, Optional
import types
import sys

import pixeltable.func as func
import pixeltable.type_system as ts
import pixeltable.env as env


def create_openai_module() -> types.ModuleType:
    """Create module pixeltable.functions.openai and populate it with functions for the OpenAPI functions"""
    module_name = 'pixeltable.functions.openai'
    pt_module = types.ModuleType(module_name)
    pt_module.__package__ = 'pixeltable.functions'
    sys.modules[module_name] = pt_module
    specs = [
        func.OpenAIFunctionSpec(
            name='chat_completion',
            call=['chat', 'completions', 'create'],
            params={
                'messages': ts.JsonType(nullable=False),
                'model': ts.StringType(nullable=False),
                'frequency_penalty': ts.FloatType(nullable=True),
                'logit_bias': ts.JsonType(nullable=True),
                'max_tokens': ts.IntType(nullable=True),
                'n': ts.IntType(nullable=True),
                'presence_penalty': ts.FloatType(nullable=True),
                'response_format': ts.JsonType(nullable=True),
                'seed': ts.IntType(nullable=True),
                # tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
                # tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
                'top_p': ts.FloatType(nullable=True),
                'temperature': ts.FloatType(nullable=True),
            },
            batch_params=[],
            output_path=None,
            output_type=ts.JsonType(nullable=False),
        ),
        func.OpenAIFunctionSpec(
            name='embedding',
            call=['embeddings', 'create'],
            params={
                'input': ts.StringType(nullable=False),
                'model': ts.StringType(nullable=False),  # TODO: defaults? text-embedding-ada-002
                'encoding_format': ts.StringType(nullable=True),
            },
            batch_params=['input'],
            output_path='data[].embedding',
            output_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False),
        ),
        func.OpenAIFunctionSpec(
            name='moderation',
            call=['moderations', 'create'],
            params={
                'input': ts.StringType(nullable=False),
                'model': ts.StringType(nullable=True),
            },
            batch_params=[],
            output_path='results[0]',
            output_type=ts.JsonType(nullable=False),
        ),
    ]

    for spec in specs:
        fn = func.OpenAIFunction(spec, module_name=module_name)
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
