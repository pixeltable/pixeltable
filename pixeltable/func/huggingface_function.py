from typing import Optional, List, Any, Dict, Callable, Type, Union
import inspect
import importlib

from .signature import Signature, Parameter
from .function_md import FunctionMd
from .external_function import ExternalFunction
import pixeltable.type_system as ts
import pixeltable.env as env


class HuggingFaceFunction(ExternalFunction):
    """Base class of Functions for HF models

    A specific HF model subclass captures
    - the model's processor and model classes (eg, AutoTokenizer/AutoModel, or CLIPProcessor/CLIPModel)
    - the model's custom invocation logic (eg, which model method to call, what parameters to expose, etc.)

    For each subclass:
    - implement the invoke_model() method and possibly verify_call(), if needed
    - use the huggingface_fn() decorator to create instances of the subclass in functions/huggingface.py

    TODO:
    - ExprEvalNode doesn't schedule these calls correctly (doesn't assign different model ids to separate cohorts)
    """
    MODEL_ID_PARAM_NAME = 'model_id'

    # caches to avoid repeated instantiations
    # key: model id
    model_cache: Dict[str, Any] = {}
    processor_cache: Dict[str, Any] = {}

    def __init__(
            self, package_name: str, processor_class_name: Optional[str], model_class_name: str,
            md: FunctionMd, py_signature: inspect.Signature, batch_size: int, constant_params: List[str]):
        """
        Requires that 'model_id' is a string parameter.
        Args:
            package_name: name of the package containing the model and processor classes
            processor_class_name: name of the processor class, or None if the model doesn't require a processor
            model_class_name: name of the model class
            md: metadata for the function
            py_signature: signature of the function
            batch_size: batch size for invoking the model
            constant_params: list of parameter names that need to receive constant values (ie, not ColumnRefs)
        """
        assert self.MODEL_ID_PARAM_NAME in md.signature.parameters
        assert len(md.signature.parameters) == len(py_signature.parameters)
        # don't resolve the processor and model classes here, we don't know whether the package is available
        self.package_name = package_name
        self.processor_class_name = processor_class_name
        self.model_class_name = model_class_name
        self.batch_size = batch_size
        self.constant_params = constant_params + [self.MODEL_ID_PARAM_NAME]
        module_name = '.'.join(md.fqn.split('.')[:-1])
        super().__init__(md, module_name=module_name, py_signature=py_signature)

    def verify_call(self, bound_args: Dict[str, Any]) -> None:
        """Verify constant parameters"""
        import pixeltable.exprs as exprs
        for param_name in self.constant_params:
            if param_name in bound_args and isinstance(bound_args[param_name], exprs.Expr):
                raise ValueError((
                    f'{self.display_name}(): ',
                    f'parameter {param_name} must be a constant value, not a Pixeltable expression'
                ))

    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        return self.batch_size

    def invoke(self, arg_batches: List[List[Any]], kwarg_batches: Dict[str, List[Any]]) -> List[Any]:
        assert self.MODEL_ID_PARAM_NAME in kwarg_batches
        env.Env.get().require_package(self.package_name)
        package = importlib.import_module(self.package_name)
        model_id = kwarg_batches[self.MODEL_ID_PARAM_NAME][0]

        # create processor instance
        if self.processor_class_name is not None:
            processor_class = getattr(package, self.processor_class_name)
            if model_id not in self.processor_cache:
                self.processor_cache[model_id] = processor_class.from_pretrained(model_id)
            processor = self.processor_cache[model_id]
        else:
            processor = None

        # create model instance
        model_class = getattr(package, self.model_class_name)
        if model_id not in self.model_cache:
            if hasattr(model_class, 'from_pretrained'):
                self.model_cache[model_id] = model_class.from_pretrained(model_id)
            else:
                self.model_cache[model_id] = model_class(model_id)
        model = self.model_cache[model_id]

        # remove constant parameters and model_id
        kwargs = {
            k: v[0] for k, v in kwarg_batches.items() if k in self.constant_params and k != self.MODEL_ID_PARAM_NAME
        }
        kwarg_batches = {k: v for k, v in kwarg_batches.items() if k not in self.constant_params}
        return self.invoke_model(processor, model, *arg_batches, **kwargs, **kwarg_batches)

    def invoke_model(self, processor: Optional[Any], model: Any, *args: Any, **kwargs: Any) -> List[Any]:
        raise NotImplementedError


def huggingface_fn(
        *, return_type: ts.ColumnType, param_types: List[ts.ColumnType], subclass: Type[HuggingFaceFunction],
        batch_size: int = 1, constant_params: Optional[List[str]] = None
) -> Callable:
    """Returns decorator to create an instance of a subclass of HuggingfaceFunction.
    The purpose of this decorator is purely to simplify constructing the FunctionMd and inspect.Signature objects.
    The body of the Callable is never executed.
    """
    if constant_params is None:
        constant_params = []
    def decorator(fn: Callable) -> HuggingFaceFunction:
        py_signature = inspect.signature(fn)
        assert len(param_types) == len(py_signature.parameters)
        assert HuggingFaceFunction.MODEL_ID_PARAM_NAME in py_signature.parameters
        parameters = [Parameter(p.name, param_types[i], p.kind) for i, p in enumerate(py_signature.parameters.values())]
        md = FunctionMd(Signature(return_type=return_type, parameters=parameters), is_agg=False, is_library_fn=False)
        md.fqn = f'{fn.__module__}.{fn.__name__}'
        return subclass(md, py_signature=py_signature, batch_size=batch_size, constant_params=constant_params)
    return decorator


class SentenceTransformerFunction(HuggingFaceFunction):
    """Function for sentence_transformers.SentenceTransformer.encode()

    Pixeltable doesn't support Union types, which is why this requires two functions, one for a single sentence
    and another for a list of sentences.
    """
    def __init__(self, md: FunctionMd, py_signature: inspect.Signature, batch_size: int, constant_params: List[str]):
        super().__init__(
            'sentence_transformers', None, 'SentenceTransformer', md, py_signature, batch_size, constant_params)

    def invoke_model(
            self, _: Optional[Any], model: Any, sentences: Union[List[str], List[List[str]]],
            normalize_embeddings: bool = False
    ) -> List[Any]:
        encode_list = isinstance(sentences[0], list)
        if encode_list:
            # this is the '_list' version of the function
            assert len(sentences) == 1
        array = model.encode(sentences[0] if encode_list else sentences, normalize_embeddings=normalize_embeddings)
        if encode_list:
            # convert to json-serializable structure and return batch of size 1
            # TODO: support ndarrays in JsonType() values
            return [[array[i].tolist() for i in range(array.shape[0])]]
        else:
            return [array[i] for i in range(array.shape[0])]


class CrossEncoderFunction(HuggingFaceFunction):
    """Function for sentence_transformers.CrossEncoder.predict()
    """
    def __init__(self, md: FunctionMd, py_signature: inspect.Signature, batch_size: int, constant_params: List[str]):
        super().__init__(
            'sentence_transformers', None, 'CrossEncoder', md, py_signature, batch_size, constant_params)

    def invoke_model(
            self, _: Optional[Any], model: Any, sentences1: List[str], sentences2: Union[List[str], List[List[str]]]
    ) -> List[Any]:
        assert len(sentences1) == len(sentences2)
        predict_list = isinstance(sentences2[0], list)
        if predict_list:
            # this is the '_list' version of the function
            assert len(sentences2) == 1
            s1 = sentences1[0]
            array = model.predict([[s1, s2] for s2 in sentences2[0]], convert_to_numpy=True)
            return [array.tolist()]
        else:
            array = model.predict([[s1, s2] for s1, s2 in zip(sentences1, sentences2)], convert_to_numpy=True)
            return array.tolist()


class ClipFunction(HuggingFaceFunction):
    """Function for transformers.CLIPModel
    """
    def __init__(self, md: FunctionMd, py_signature: inspect.Signature, batch_size: int, constant_params: List[str]):
        super().__init__(
            'transformers', 'CLIPProcessor', 'CLIPModel', md, py_signature, batch_size, constant_params)

    def verify_call(self, bound_args: Dict[str, Any]) -> None:
        """Verify that only one of 'text' and 'img' is specified"""
        if 'text' in bound_args and 'img' in bound_args:
            raise ValueError(f'{self.display_name}(): only one of "text" and "img" can be specified')
        super().verify_call(bound_args)

    def invoke_model(
            self, processor: Optional[Any], model: Any,
            text: Optional[List[str]] = None, img: Optional[List['PIL.Image.Image']] = None
    ) -> List[Any]:
        assert (text is not None) ^ (img is not None)
        if text is not None:
            inputs = processor(text=text, return_tensors='pt', padding=True, truncation=True)
            embeddings = model.get_text_features(**inputs).detach().numpy()
        else:
            inputs = processor(images=img, return_tensors='pt', padding=True)
            embeddings = model.get_image_features(**inputs).detach().numpy()
        return [embeddings[i] for i in range(embeddings.shape[0])]
