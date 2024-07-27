"""
A collection of Pixeltable UDFs that wrap various models from the Huggingface transformers libraries.

These UDFs will cause Pixeltable to invoke the relevant models locally. In order to use them, you must
first `pip install transformers` (or in some cases, `sentence-transformers`, as noted in the specific
UDFs).
"""

from typing import Callable, TypeVar, Optional, Any

import PIL.Image
import numpy as np

import pixeltable as pxt
import pixeltable.env as env
import pixeltable.type_system as ts
from pixeltable.func import Batch
from pixeltable.functions.util import resolve_torch_device, normalize_image_mode
from pixeltable.utils.code import local_public_names


@pxt.udf(batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType()))
def sentence_transformer(
    sentence: Batch[str], *, model_id: str, normalize_embeddings: bool = False
) -> Batch[np.ndarray]:
    """
    Runs the specified pretrained sentence-transformers model. `model_id` should be a pretrained model, as described
    in the [Sentence Transformers Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
    documentation.

    __Requirements:__

    - `pip install sentence-transformers`

    Args:
        sentence: The sentence to embed.
        model_id: The pretrained model to use for the encoding.
        normalize_embeddings: If `True`, normalizes embeddings to length 1; see the
            [Sentence Transformers API Docs](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)
            for more details

    Returns:
        An array containing the output of the embedding model.

    Examples:
        Add a computed column that applies the model `all-mpnet-base-2` to an existing Pixeltable column `tbl.sentence`
        of the table `tbl`:

        >>> tbl['result'] = sentence_transformer(tbl.sentence, model_id='all-mpnet-base-v2')
    """
    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import SentenceTransformer

    model = _lookup_model(model_id, SentenceTransformer)

    array = model.encode(sentence, normalize_embeddings=normalize_embeddings)
    return [array[i] for i in range(array.shape[0])]


@sentence_transformer.conditional_return_type
def _(model_id: str) -> ts.ArrayType:
    try:
        from sentence_transformers import SentenceTransformer

        model = _lookup_model(model_id, SentenceTransformer)
        return ts.ArrayType((model.get_sentence_embedding_dimension(),), dtype=ts.FloatType(), nullable=False)
    except ImportError:
        return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)


@pxt.udf
def sentence_transformer_list(sentences: list, *, model_id: str, normalize_embeddings: bool = False) -> list:
    env.Env.get().require_package('sentence_transformers')
    from sentence_transformers import SentenceTransformer

    model = _lookup_model(model_id, SentenceTransformer)

    array = model.encode(sentences, normalize_embeddings=normalize_embeddings)
    return [array[i].tolist() for i in range(array.shape[0])]


@pxt.udf(batch_size=32)
def cross_encoder(sentences1: Batch[str], sentences2: Batch[str], *, model_id: str) -> Batch[float]:
    """
    Runs the specified cross-encoder model to compute similarity scores for pairs of sentences.
    `model_id` should be a pretrained model, as described in the
    [Cross-Encoder Pretrained Models](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
    documentation.

    __Requirements:__

    - `pip install sentence-transformers`

    Parameters:
        sentences1: The first sentence to be paired.
        sentences2: The second sentence to be paired.
        model_id: The identifier of the cross-encoder model to use.

    Returns:
        The similarity score between the inputs.

    Examples:
        Add a computed column that applies the model `ms-marco-MiniLM-L-4-v2` to the sentences in
        columns `tbl.sentence1` and `tbl.sentence2`:

        >>> tbl['result'] = sentence_transformer(
                tbl.sentence1, tbl.sentence2, model_id='ms-marco-MiniLM-L-4-v2'
            )
    """
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


@pxt.udf(batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False))
def clip_text(text: Batch[str], *, model_id: str) -> Batch[np.ndarray]:
    """
    Runs the specified CLIP embedding model on text. `model_id` should be a reference to a pretrained
    [CLIP Model](https://huggingface.co/docs/transformers/model_doc/clip).

    __Requirements:__

    - `pip install transformers`

    Args:
        text: The string to embed.
        model_id: The pretrained model to use for the embedding.

    Returns:
        An array containing the output of the embedding model.

    Examples:
        Add a computed column that applies the model `openai/clip-vit-base-patch32` to an existing
        Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl['result'] = clip_text(tbl.text, model_id='openai/clip-vit-base-patch32')
    """
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


@pxt.udf(batch_size=32, return_type=ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False))
def clip_image(image: Batch[PIL.Image.Image], *, model_id: str) -> Batch[np.ndarray]:
    """
    Runs the specified CLIP embedding model on images. `model_id` should be a reference to a pretrained
    [CLIP Model](https://huggingface.co/docs/transformers/model_doc/clip).

    __Requirements:__

    - `pip install transformers`

    Args:
        image: The image to embed.
        model_id: The pretrained model to use for the embedding.

    Returns:
        An array containing the output of the embedding model.

    Examples:
        Add a computed column that applies the model `openai/clip-vit-base-patch32` to an existing
        Pixeltable column `tbl.image` of the table `tbl`:

        >>> tbl['result'] = clip_image(tbl.image, model_id='openai/clip-vit-base-patch32')
    """
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


@clip_text.conditional_return_type
@clip_image.conditional_return_type
def _(model_id: str) -> ts.ArrayType:
    try:
        from transformers import CLIPModel

        model = _lookup_model(model_id, CLIPModel.from_pretrained)
        return ts.ArrayType((model.config.projection_dim,), dtype=ts.FloatType(), nullable=False)
    except ImportError:
        return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)


@pxt.udf(batch_size=4)
def detr_for_object_detection(image: Batch[PIL.Image.Image], *, model_id: str, threshold: float = 0.5) -> Batch[dict]:
    """
    Runs the specified DETR object detection model. `model_id` should be a reference to a pretrained
    [DETR Model](https://huggingface.co/docs/transformers/model_doc/detr).

    __Requirements:__

    - `pip install transformers`

    Args:
        image: The image to embed.
        model_id: The pretrained model to use for the embedding.

    Returns:
        A dictionary containing the output of the object detection model, in the following format:

    ```python
    {
        'scores': [0.99, 0.999],  # list of confidence scores for each detected object
        'labels': [25, 25],  # list of COCO class labels for each detected object
        'label_text': ['giraffe', 'giraffe'],  # corresponding text names of class labels
        'boxes': [[51.942, 356.174, 181.481, 413.975], [383.225, 58.66, 605.64, 361.346]]
            # list of bounding boxes for each detected object, as [x1, y1, x2, y2]
    }
    ```

    Examples:
        Add a computed column that applies the model `facebook/detr-resnet-50` to an existing
        Pixeltable column `tbl.image` of the table `tbl`:

        >>> tbl['detections'] = detr_for_object_detection(
                tbl.image,
                model_id='facebook/detr-resnet-50',
                threshold=0.8
            )
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import DetrImageProcessor, DetrForObjectDetection

    model = _lookup_model(
        model_id, lambda x: DetrForObjectDetection.from_pretrained(x, revision='no_timm'), device=device
    )
    processor = _lookup_processor(model_id, lambda x: DetrImageProcessor.from_pretrained(x, revision='no_timm'))
    normalized_images = [normalize_image_mode(img) for img in image]

    with torch.no_grad():
        inputs = processor(images=normalized_images, return_tensors='pt')
        outputs = model(**inputs.to(device))
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(img.height, img.width) for img in image]
        )

    return [
        {
            'scores': [score.item() for score in result['scores']],
            'labels': [label.item() for label in result['labels']],
            'label_text': [model.config.id2label[label.item()] for label in result['labels']],
            'boxes': [box.tolist() for box in result['boxes']],
        }
        for result in results
    ]


@pxt.udf
def detr_to_coco(image: PIL.Image.Image, detr_info: dict[str, Any]) -> dict[str, Any]:
    bboxes, labels = detr_info['boxes'], detr_info['labels']
    annotations = [
        {'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], 'category': label}
        for bbox, label in zip(bboxes, labels)
    ]
    return {'image': {'width': image.width, 'height': image.height}, 'annotations': annotations}


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


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
