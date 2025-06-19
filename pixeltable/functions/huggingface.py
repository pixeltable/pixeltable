"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various models from the Hugging Face `transformers` package.

These UDFs will cause Pixeltable to invoke the relevant models locally. In order to use them, you must
first `pip install transformers` (or in some cases, `sentence-transformers`, as noted in the specific
UDFs).
"""

from typing import Any, Callable, Optional, TypeVar

import PIL.Image

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.functions.util import normalize_image_mode, resolve_torch_device
from pixeltable.utils.code import local_public_names


@pxt.udf(batch_size=32)
def sentence_transformer(
    sentence: Batch[str], *, model_id: str, normalize_embeddings: bool = False
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Computes sentence embeddings. `model_id` should be a pretrained Sentence Transformers model, as described
    in the [Sentence Transformers Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
    documentation.

    __Requirements:__

    - `pip install torch sentence-transformers`

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

        >>> tbl.add_computed_column(result=sentence_transformer(tbl.sentence, model_id='all-mpnet-base-v2'))
    """
    env.Env.get().require_package('sentence_transformers')
    device = resolve_torch_device('auto')
    from sentence_transformers import SentenceTransformer

    # specifying the device, moves the model to device (gpu:cuda/mps, cpu)
    model = _lookup_model(model_id, SentenceTransformer, device=device, pass_device_to_create=True)

    # specifying the device, uses it for computation
    array = model.encode(sentence, device=device, normalize_embeddings=normalize_embeddings)
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
    device = resolve_torch_device('auto')
    from sentence_transformers import SentenceTransformer

    # specifying the device, moves the model to device (gpu:cuda/mps, cpu)
    model = _lookup_model(model_id, SentenceTransformer, device=device, pass_device_to_create=True)

    # specifying the device, uses it for computation
    array = model.encode(sentences, device=device, normalize_embeddings=normalize_embeddings)
    return [array[i].tolist() for i in range(array.shape[0])]


@pxt.udf(batch_size=32)
def cross_encoder(sentences1: Batch[str], sentences2: Batch[str], *, model_id: str) -> Batch[float]:
    """
    Performs predicts on the given sentence pair.
    `model_id` should be a pretrained Cross-Encoder model, as described in the
    [Cross-Encoder Pretrained Models](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
    documentation.

    __Requirements:__

    - `pip install torch sentence-transformers`

    Parameters:
        sentences1: The first sentence to be paired.
        sentences2: The second sentence to be paired.
        model_id: The identifier of the cross-encoder model to use.

    Returns:
        The similarity score between the inputs.

    Examples:
        Add a computed column that applies the model `ms-marco-MiniLM-L-4-v2` to the sentences in
        columns `tbl.sentence1` and `tbl.sentence2`:

        >>> tbl.add_computed_column(result=sentence_transformer(
        ...     tbl.sentence1, tbl.sentence2, model_id='ms-marco-MiniLM-L-4-v2'
        ... ))
    """
    env.Env.get().require_package('sentence_transformers')
    device = resolve_torch_device('auto')
    from sentence_transformers import CrossEncoder

    # specifying the device, moves the model to device (gpu:cuda/mps, cpu)
    # and uses the device for predict computation
    model = _lookup_model(model_id, CrossEncoder, device=device, pass_device_to_create=True)

    array = model.predict([[s1, s2] for s1, s2 in zip(sentences1, sentences2)], convert_to_numpy=True)
    return array.tolist()


@pxt.udf
def cross_encoder_list(sentence1: str, sentences2: list, *, model_id: str) -> list:
    env.Env.get().require_package('sentence_transformers')
    device = resolve_torch_device('auto')
    from sentence_transformers import CrossEncoder

    # specifying the device, moves the model to device (gpu:cuda/mps, cpu)
    # and uses the device for predict computation
    model = _lookup_model(model_id, CrossEncoder, device=device, pass_device_to_create=True)

    array = model.predict([[sentence1, s2] for s2 in sentences2], convert_to_numpy=True)
    return array.tolist()


@pxt.udf(batch_size=32)
def clip(text: Batch[str], *, model_id: str) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Computes a CLIP embedding for the specified text or image. `model_id` should be a reference to a pretrained
    [CLIP Model](https://huggingface.co/docs/transformers/model_doc/clip).

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The string to embed.
        model_id: The pretrained model to use for the embedding.

    Returns:
        An array containing the output of the embedding model.

    Examples:
        Add a computed column that applies the model `openai/clip-vit-base-patch32` to an existing
        Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     result=clip(tbl.text, model_id='openai/clip-vit-base-patch32')
        ... )

        The same would work with an image column `tbl.image` in place of `tbl.text`.
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


@clip.overload
def _(image: Batch[PIL.Image.Image], *, model_id: str) -> Batch[pxt.Array[(None,), pxt.Float]]:
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


@clip.conditional_return_type
def _(model_id: str) -> ts.ArrayType:
    try:
        from transformers import CLIPModel

        model = _lookup_model(model_id, CLIPModel.from_pretrained)
        return ts.ArrayType((model.config.projection_dim,), dtype=ts.FloatType(), nullable=False)
    except ImportError:
        return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)


@pxt.udf(batch_size=4)
def detr_for_object_detection(
    image: Batch[PIL.Image.Image], *, model_id: str, threshold: float = 0.5, revision: str = 'no_timm'
) -> Batch[dict]:
    """
    Computes DETR object detections for the specified image. `model_id` should be a reference to a pretrained
    [DETR Model](https://huggingface.co/docs/transformers/model_doc/detr).

    __Requirements:__

    - `pip install torch transformers`

    Args:
        image: The image to embed.
        model_id: The pretrained model to use for object detection.

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
        Pixeltable column `image` of the table `tbl`:

        >>> tbl.add_computed_column(detections=detr_for_object_detection(
        ...     tbl.image,
        ...     model_id='facebook/detr-resnet-50',
        ...     threshold=0.8
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor

    model = _lookup_model(
        model_id, lambda x: DetrForObjectDetection.from_pretrained(x, revision=revision), device=device
    )
    processor = _lookup_processor(model_id, lambda x: DetrImageProcessor.from_pretrained(x, revision=revision))
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


@pxt.udf(batch_size=4)
def vit_for_image_classification(
    image: Batch[PIL.Image.Image], *, model_id: str, top_k: int = 5
) -> Batch[dict[str, Any]]:
    """
    Computes image classifications for the specified image using a Vision Transformer (ViT) model.
    `model_id` should be a reference to a pretrained [ViT Model](https://huggingface.co/docs/transformers/en/model_doc/vit).

    __Note:__ Be sure the model is a ViT model that is trained for image classification (that is, a model designed for
    use with the
    [ViTForImageClassification](https://huggingface.co/docs/transformers/en/model_doc/vit#transformers.ViTForImageClassification)
    class), such as `google/vit-base-patch16-224`. General feature-extraction models such as
    `google/vit-base-patch16-224-in21k` will not produce the desired results.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        image: The image to classify.
        model_id: The pretrained model to use for the classification.
        top_k: The number of classes to return.

    Returns:
        A dictionary containing the output of the image classification model, in the following format:

        ```python
        {
            'scores': [0.325, 0.198, 0.105],  # list of probabilities of the top-k most likely classes
            'labels': [340, 353, 386],  # list of class IDs for the top-k most likely classes
            'label_text': ['zebra', 'gazelle', 'African elephant, Loxodonta africana'],
                # corresponding text names of the top-k most likely classes
        ```

    Examples:
        Add a computed column that applies the model `google/vit-base-patch16-224` to an existing
        Pixeltable column `image` of the table `tbl`, returning the 10 most likely classes for each image:

        >>> tbl.add_computed_column(image_class=vit_for_image_classification(
        ...     tbl.image,
        ...     model_id='google/vit-base-patch16-224',
        ...     top_k=10
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import ViTForImageClassification, ViTImageProcessor

    model: ViTForImageClassification = _lookup_model(model_id, ViTForImageClassification.from_pretrained, device=device)
    processor = _lookup_processor(model_id, ViTImageProcessor.from_pretrained)
    normalized_images = [normalize_image_mode(img) for img in image]

    with torch.no_grad():
        inputs = processor(images=normalized_images, return_tensors='pt')
        outputs = model(**inputs.to(device))
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

    # There is no official post_process method for ViT models; for consistency, we structure the output
    # the same way as the output of the DETR model given by `post_process_object_detection`.
    return [
        {
            'scores': [top_k_probs[n, k].item() for k in range(top_k_probs.shape[1])],
            'labels': [top_k_indices[n, k].item() for k in range(top_k_probs.shape[1])],
            'label_text': [model.config.id2label[top_k_indices[n, k].item()] for k in range(top_k_probs.shape[1])],
        }
        for n in range(top_k_probs.shape[0])
    ]


@pxt.udf
def speech2text_for_conditional_generation(audio: pxt.Audio, *, model_id: str, language: Optional[str] = None) -> str:
    """
    Transcribes or translates speech to text using a Speech2Text model. `model_id` should be a reference to a
    pretrained [Speech2Text](https://huggingface.co/docs/transformers/en/model_doc/speech_to_text) model.

    __Requirements:__

    - `pip install torch torchaudio sentencepiece transformers`

    Args:
        audio: The audio clip to transcribe or translate.
        model_id: The pretrained model to use for the transcription or translation.
        language: If using a multilingual translation model, the language code to translate to. If not provided,
            the model's default language will be used. If the model is not translation model, is not a
            multilingual model, or does not support the specified language, an error will be raised.

    Returns:
        The transcribed or translated text.

    Examples:
        Add a computed column that applies the model `facebook/s2t-small-librispeech-asr` to an existing
        Pixeltable column `audio` of the table `tbl`:

        >>> tbl.add_computed_column(transcription=speech2text_for_conditional_generation(
        ...     tbl.audio,
        ...     model_id='facebook/s2t-small-librispeech-asr'
        ... ))

        Add a computed column that applies the model `facebook/s2t-medium-mustc-multilingual-st` to an existing
        Pixeltable column `audio` of the table `tbl`, translating the audio to French:

        >>> tbl.add_computed_column(translation=speech2text_for_conditional_generation(
        ...     tbl.audio,
        ...     model_id='facebook/s2t-medium-mustc-multilingual-st',
        ...     language='fr'
        ... ))
    """
    env.Env.get().require_package('transformers')
    env.Env.get().require_package('torchaudio')
    env.Env.get().require_package('sentencepiece')
    device = resolve_torch_device('auto', allow_mps=False)  # Doesn't seem to work on 'mps'; use 'cpu' instead
    import torch
    import torchaudio  # type: ignore[import-untyped]
    from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor, Speech2TextTokenizer

    model = _lookup_model(model_id, Speech2TextForConditionalGeneration.from_pretrained, device=device)
    processor = _lookup_processor(model_id, Speech2TextProcessor.from_pretrained)
    tokenizer = processor.tokenizer
    assert isinstance(processor, Speech2TextProcessor)
    assert isinstance(tokenizer, Speech2TextTokenizer)

    if language is not None and language not in tokenizer.lang_code_to_id:
        raise excs.Error(
            f"Language code '{language}' is not supported by the model '{model_id}'. "
            f'Supported languages are: {list(tokenizer.lang_code_to_id.keys())}'
        )

    forced_bos_token_id: Optional[int] = None if language is None else tokenizer.lang_code_to_id[language]

    # Get the model's sampling rate. Default to 16 kHz (the standard) if not in config
    model_sampling_rate = getattr(model.config, 'sampling_rate', 16_000)

    waveform, sampling_rate = torchaudio.load(audio)

    # Resample to the model's sampling rate, if necessary
    if sampling_rate != model_sampling_rate:
        waveform = torchaudio.transforms.Resample(sampling_rate, model_sampling_rate)(waveform)

    # Average the channels to get a single-channel waveform as a 1D tensor (if the original waveform is already
    # mono, this will simply squeeze the tensor)
    assert waveform.dim() == 2
    waveform = torch.mean(waveform, dim=0)
    assert waveform.dim() == 1

    with torch.no_grad():
        inputs = processor(waveform, sampling_rate=model_sampling_rate, return_tensors='pt')
        generated_ids = model.generate(**inputs.to(device), forced_bos_token_id=forced_bos_token_id).to('cpu')

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription


@pxt.udf
def detr_to_coco(image: PIL.Image.Image, detr_info: dict[str, Any]) -> dict[str, Any]:
    """
    Converts the output of a DETR object detection model to COCO format.

    Args:
        image: The image for which detections were computed.
        detr_info: The output of a DETR object detection model, as returned by `detr_for_object_detection`.

    Returns:
        A dictionary containing the data from `detr_info`, converted to COCO format.

    Examples:
        Add a computed column that converts the output `tbl.detections` to COCO format, where `tbl.image`
        is the image for which detections were computed:

        >>> tbl.add_computed_column(detections_coco=detr_to_coco(tbl.image, tbl.detections))
    """
    bboxes, labels = detr_info['boxes'], detr_info['labels']
    annotations = [
        {'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], 'category': label}
        for bbox, label in zip(bboxes, labels)
    ]
    return {'image': {'width': image.width, 'height': image.height}, 'annotations': annotations}


T = TypeVar('T')


def _lookup_model(
    model_id: str, create: Callable[..., T], device: Optional[str] = None, pass_device_to_create: bool = False
) -> T:
    from torch import nn

    key = (model_id, create, device)  # For safety, include the `create` callable in the cache key
    if key not in _model_cache:
        if pass_device_to_create:
            model = create(model_id, device=device)
        else:
            model = create(model_id)
        if isinstance(model, nn.Module):
            if not pass_device_to_create and device is not None:
                model.to(device)
            model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def _lookup_processor(model_id: str, create: Callable[[str], T]) -> T:
    key = (model_id, create)  # For safety, include the `create` callable in the cache key
    if key not in _processor_cache:
        _processor_cache[key] = create(model_id)
    return _processor_cache[key]


_model_cache: dict[tuple[str, Callable, Optional[str]], Any] = {}
_processor_cache: dict[tuple[str, Callable], Any] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
