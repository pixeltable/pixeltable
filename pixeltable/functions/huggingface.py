from typing import Optional, List

import PIL.Image

import pixeltable.func as func
import pixeltable.type_system as ts

# To add new models, see instructions in func/huggingface_function.py

@func.huggingface_fn(
    return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False),
    param_types=[ts.StringType(), ts.StringType(), ts.BoolType()],
    batch_size=32,
    constant_params=['normalize_embeddings'],
    subclass=func.SentenceTransformerFunction)
def sentence_transformer(sentence: str, *, model_id: str, normalize_embeddings: bool = True):
    pass

@func.huggingface_fn(
    return_type=ts.JsonType(),
    param_types=[ts.JsonType(), ts.StringType(), ts.BoolType()],
    batch_size=1,
    constant_params=['normalize_embeddings'],
    subclass=func.SentenceTransformerFunction)
def sentence_transformer_list(sentence: List[str], *, model_id: str, normalize_embeddings: bool = True):
    pass

@func.huggingface_fn(
    return_type=ts.FloatType(),
    param_types=[ts.StringType(), ts.StringType(), ts.StringType()],
    batch_size=32,
    subclass=func.CrossEncoderFunction)
def cross_encoder(sent1: str, sent2: str, *, model_id: str):
    pass

@func.huggingface_fn(
    return_type=ts.JsonType(),  # list of floats
    param_types=[ts.StringType(), ts.JsonType(), ts.StringType()],
    batch_size=1,
    subclass=func.CrossEncoderFunction)
def cross_encoder_list(sent1: str, sent2: List[str], *, model_id: str):
    pass

@func.huggingface_fn(
    return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False),
    param_types=[ts.StringType(nullable=False), ts.StringType(nullable=True), ts.ImageType(nullable=True)],
    batch_size=32,
    subclass=func.ClipFunction)
def clip(*, model_id: str, text: Optional[str] = None, img: Optional[PIL.Image.Image] = None):
    pass

#func.FunctionRegistry.get().register_module()