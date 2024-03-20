from typing import Optional, List, Any, Callable, Iterable

import PIL.Image
import numpy as np

import pixeltable as pxt
import pixeltable.env as env
import pixeltable.func.huggingface_function as hf
import pixeltable.type_system as ts


@pxt.udf(batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType()))
def sentence_transformer_2(sentences: Iterable[str], *, model_id: str, normalize_embeddings: bool = False) -> Iterable[np.ndarray]:

    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import SentenceTransformer

    model = _lookup_model(model_id, SentenceTransformer, lambda x: SentenceTransformer(x))

    array = model.encode(sentences, normalize_embeddings=normalize_embeddings)
    return [array[i] for i in range(array.shape[0])]


@pxt.udf
def sentence_transformer_list_2(sentences: list, *, model_id: str, normalize_embeddings: bool = False) -> list:

    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import SentenceTransformer

    model = _lookup_model(model_id, SentenceTransformer, lambda x: SentenceTransformer(x))

    array = model.encode(sentences, normalize_embeddings=normalize_embeddings)
    return [array[i].tolist() for i in range(array.shape[0])]


@pxt.udf(batch_size=32)
def cross_encoder_2(sentences1: Iterable[str], sentences2: Iterable[str], *, model_id: str) -> Iterable[float]:

    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import CrossEncoder

    model = _lookup_model(model_id, CrossEncoder, lambda x: CrossEncoder(x))

    array = model.predict([[s1, s2] for s1, s2 in zip(sentences1, sentences2)], convert_to_numpy=True)
    return array.tolist()


@pxt.udf
def cross_encoder_list_2(sentence1: str, sentences2: list, *, model_id: str) -> list:

    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import CrossEncoder

    model = _lookup_model(model_id, CrossEncoder, lambda x: CrossEncoder(x))

    array = model.predict([[sentence1, s2] for s2 in sentences2], convert_to_numpy=True)
    return array.tolist()


def _lookup_model(model_id: str, model_class: type, create: Callable) -> Any:
    key = (model_id, model_class)
    if key not in _model_cache:
        _model_cache[key] = create(model_id)
    return _model_cache[key]


_model_cache = {}


@hf.huggingface_fn(
    return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False),
    param_types=[ts.StringType(), ts.StringType(), ts.BoolType()],
    batch_size=32,
    constant_params=['normalize_embeddings'],
    subclass=hf.SentenceTransformerFunction)
def sentence_transformer(sentence: str, *, model_id: str, normalize_embeddings: bool = True):
    pass

@hf.huggingface_fn(
    return_type=ts.JsonType(),
    param_types=[ts.JsonType(), ts.StringType(), ts.BoolType()],
    batch_size=1,
    constant_params=['normalize_embeddings'],
    subclass=hf.SentenceTransformerFunction)
def sentence_transformer_list(sentence: List[str], *, model_id: str, normalize_embeddings: bool = True):
    pass

@hf.huggingface_fn(
    return_type=ts.FloatType(),
    param_types=[ts.StringType(), ts.StringType(), ts.StringType()],
    batch_size=32,
    subclass=hf.CrossEncoderFunction)
def cross_encoder(sent1: str, sent2: str, *, model_id: str):
    pass

@hf.huggingface_fn(
    return_type=ts.JsonType(),  # list of floats
    param_types=[ts.StringType(), ts.JsonType(), ts.StringType()],
    batch_size=1,
    subclass=hf.CrossEncoderFunction)
def cross_encoder_list(sent1: str, sent2: List[str], *, model_id: str):
    pass

@hf.huggingface_fn(
    return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False),
    param_types=[ts.StringType(nullable=False), ts.StringType(nullable=True), ts.ImageType(nullable=True)],
    batch_size=32,
    subclass=hf.ClipFunction)
def clip(*, model_id: str, text: Optional[str] = None, img: Optional[PIL.Image.Image] = None):
    pass
