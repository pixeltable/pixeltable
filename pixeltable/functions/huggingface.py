from typing import Callable, TypeVar, Optional, Any

import PIL.Image
import numpy as np

import pixeltable as pxt
import pixeltable.env as env
import pixeltable.type_system as ts
from pixeltable.func import Batch
from pixeltable.functions.util import resolve_torch_device


def _sentence_transformer_call_return_type(model_id: str) -> ts.ColumnType:
    try:
        from sentence_transformers import SentenceTransformer
        model = _lookup_model(model_id, SentenceTransformer)
        return ts.ArrayType((model.get_sentence_embedding_dimension(),), dtype=ts.FloatType(), nullable=False)
    except ImportError:
        return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)

@pxt.udf(
    batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType()),
    call_return_type=_sentence_transformer_call_return_type)
def sentence_transformer(
        sentences: Batch[str], *, model_id: str, normalize_embeddings: bool = False
) -> Batch[np.ndarray]:
    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import SentenceTransformer

    model = _lookup_model(model_id, SentenceTransformer)

    array = model.encode(sentences, normalize_embeddings=normalize_embeddings)
    return [array[i] for i in range(array.shape[0])]


@pxt.udf
def sentence_transformer_list(sentences: list, *, model_id: str, normalize_embeddings: bool = False) -> list:
    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import SentenceTransformer

    model = _lookup_model(model_id, SentenceTransformer)

    array = model.encode(sentences, normalize_embeddings=normalize_embeddings)
    return [array[i].tolist() for i in range(array.shape[0])]


@pxt.udf(batch_size=32)
def cross_encoder(sentences1: Batch[str], sentences2: Batch[str], *, model_id: str) -> Batch[float]:
    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import CrossEncoder

    model = _lookup_model(model_id, CrossEncoder)

    array = model.predict([[s1, s2] for s1, s2 in zip(sentences1, sentences2)], convert_to_numpy=True)
    return array.tolist()


@pxt.udf
def cross_encoder_list(sentence1: str, sentences2: list, *, model_id: str) -> list:
    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import CrossEncoder

    model = _lookup_model(model_id, CrossEncoder)

    array = model.predict([[sentence1, s2] for s2 in sentences2], convert_to_numpy=True)
    return array.tolist()

def _clip_call_return_type(model_id: str) -> ts.ColumnType:
    try:
        from transformers import CLIPModel
        model = _lookup_model(model_id, CLIPModel.from_pretrained)
        return ts.ArrayType((model.config.projection_dim,), dtype=ts.FloatType(), nullable=False)
    except ImportError:
        return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)

@pxt.udf(
    batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False),
    call_return_type=_clip_call_return_type)
def clip_text(text: Batch[str], *, model_id: str) -> Batch[np.ndarray]:
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = _lookup_model(model_id, CLIPModel.from_pretrained, device=device)
    processor = _lookup_processor(model_id, CLIPProcessor.from_pretrained)

    with torch.no_grad():
        inputs = processor(text=text, return_tensors='pt', padding=True, truncation=True)
        embeddings = model.get_text_features(**inputs.to(device)).detach().to('cpu').numpy()

    return [embeddings[i] for i in range(embeddings.shape[0])]


@pxt.udf(
    batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False),
    call_return_type=_clip_call_return_type)
def clip_image(image: Batch[PIL.Image.Image], *, model_id: str) -> Batch[np.ndarray]:
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = _lookup_model(model_id, CLIPModel.from_pretrained, device=device)
    processor = _lookup_processor(model_id, CLIPProcessor.from_pretrained)

    with torch.no_grad():
        inputs = processor(images=image, return_tensors='pt', padding=True)
        embeddings = model.get_image_features(**inputs.to(device)).detach().to('cpu').numpy()

    return [embeddings[i] for i in range(embeddings.shape[0])]


@pxt.udf(batch_size=4)
def detr_for_object_detection(image: Batch[PIL.Image.Image], *, model_id: str, threshold: float = 0.5) -> Batch[dict]:
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import DetrImageProcessor, DetrForObjectDetection

    model = _lookup_model(
        model_id, lambda x: DetrForObjectDetection.from_pretrained(x, revision='no_timm'), device=device)
    processor = _lookup_processor(model_id, lambda x: DetrImageProcessor.from_pretrained(x, revision='no_timm'))

    with torch.no_grad():
        inputs = processor(images=image, return_tensors='pt')
        outputs = model(**inputs.to(device))
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(img.height, img.width) for img in image]
        )

    return [
        {
            'scores': [score.item() for score in result['scores']],
            'labels': [label.item() for label in result['labels']],
            'boxes': [box.tolist() for box in result['boxes']]
        }
        for result in results
    ]


@pxt.udf
def detr_to_coco(image: PIL.Image.Image, detr_info: dict[str, Any]) -> dict[str, Any]:
    bboxes, labels = detr_info['boxes'], detr_info['labels']
    annotations = [
        {
            'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
            'category': label
        }
        for bbox, label in zip(bboxes, labels)
    ]
    return {
        'image': {
            'width': image.width,
            'height': image.height
        },
        'annotations': annotations
    }


T = TypeVar('T')


def _lookup_model(model_id: str, create: Callable[[str], T], device: Optional[str] = None) -> T:
    from torch import nn
    key = (model_id, create, device)  # For safety, include the `create` callable in the cache key
    if key not in _model_cache:
        model = create(model_id)
        if device is not None:
            model.to(device)
        if isinstance(model, nn.Module):
            model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def _lookup_processor(model_id: str, create: Callable[[str], T]) -> T:
    key = (model_id, create)  # For safety, include the `create` callable in the cache key
    if key not in _processor_cache:
        _processor_cache[key] = create(model_id)
    return _processor_cache[key]


_model_cache = {}
_processor_cache = {}
