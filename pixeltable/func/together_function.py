from typing import Optional, Any, Dict, List, Tuple
import logging
import inspect
import dataclasses

import numpy as np
import jmespath

from .signature import Signature, Parameter
from .function_md import FunctionMd
from .external_function import ExternalFunction
import pixeltable.type_system as ts


_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass
class TogetherFunctionSpec:
    name: str

    # sequence of symbols that lead to the function call, starting from the OpenAI client
    # ex.: ['chat', 'completions', 'create'] for client.chat.completions.create()
    call: List[str]

    params: Dict[str, ts.ColumnType]
    batch_params: List[str]  # list of parameter names that accept argument batches
    output_type: ts.ColumnType

    # None: return the whole output as a dict
    # otherwise: json path within the output, evaluated with jmespath
    output_path: Optional[str]


# TODO: Abstract out into a generalized pattern that unifies with OpenAIFunction[Spec]
class TogetherFunction(ExternalFunction):
    def __init__(self, spec: TogetherFunctionSpec, module_name: str):
        self.spec = spec
        parameters = [
            Parameter(name, col_type, inspect.Parameter.KEYWORD_ONLY)
            for name, col_type in spec.params.items()
        ]
        sig = Signature(return_type=spec.output_type, parameters=parameters)
        md = FunctionMd(signature=sig, is_agg=False, is_library_fn=True)
        # construct inspect.Signature
        required_params = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name, col_type in spec.params.items() if not col_type.nullable
        ]
        opt_params = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
            for name, col_type in spec.params.items() if col_type.nullable
        ]
        py_signature = inspect.Signature(required_params + opt_params)
        super().__init__(md, module_name=module_name, py_signature=py_signature)

        self.compiled_output_path = \
            jmespath.compile(self.spec.output_path) if self.spec.output_path is not None else None

    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        return 1

    def invoke(self, arg_batches: List[List[Any]], kwarg_batches: Dict[str, List[Any]]) -> List[Any]:
        import together
        endpoint = together
        for e in self.spec.call:
            endpoint = getattr(endpoint, e)
        assert len(kwarg_batches) > 0
        assert len(kwarg_batches[next(iter(kwarg_batches))]) == 1
        kwargs = {k: v[0] for k, v in kwarg_batches.items()}
        output_dict = endpoint(**kwargs)
        if self.compiled_output_path is not None:
            val = self.compiled_output_path.search(output_dict)
            if self.spec.output_type.is_array_type():
                return [np.array(val)]
            return [val]
        return [output_dict]

