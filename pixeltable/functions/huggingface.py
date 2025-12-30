"""
Pixeltable UDFs
that wrap various models from the Hugging Face `transformers` package.

These UDFs will cause Pixeltable to invoke the relevant models locally. In order to use them, you must
first `pip install transformers` (or in some cases, `sentence-transformers`, as noted in the specific
UDFs).
"""

from typing import Any, Callable, Literal, TypeVar

import av
import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.functions.util import normalize_image_mode, resolve_torch_device
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

T = TypeVar('T')


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
    from sentence_transformers import SentenceTransformer

    model = _lookup_model(model_id, SentenceTransformer)
    return ts.ArrayType((model.get_sentence_embedding_dimension(),), dtype=ts.FloatType(), nullable=False)


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
    from transformers import CLIPModel

    model = _lookup_model(model_id, CLIPModel.from_pretrained)
    return ts.ArrayType((model.config.projection_dim,), dtype=ts.FloatType(), nullable=False)


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
def detr_for_segmentation(image: Batch[PIL.Image.Image], *, model_id: str, threshold: float = 0.5) -> Batch[dict]:
    """
    Computes DETR panoptic segmentation for the specified image. `model_id` should be a reference to a pretrained
    [DETR Model](https://huggingface.co/docs/transformers/model_doc/detr) with a segmentation head.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        image: The image to segment.
        model_id: The pretrained model to use for segmentation (e.g., 'facebook/detr-resnet-50-panoptic').
        threshold: Confidence threshold for filtering segments.

    Returns:
        A dictionary containing the output of the segmentation model, in the following format:

            ```python
            {
                'segmentation': np.ndarray,  # (H, W) array where each pixel value is a segment ID
                'segments_info': [
                    {
                        'id': 1,  # segment ID (matches pixel values in segmentation array)
                        'label_id': 0,  # class label index
                        'label_text': 'person',  # human-readable class name
                        'score': 0.98,  # confidence score
                        'was_fused': False  # whether segment was fused from multiple instances
                    },
                    ...
                ]
            }
            ```

    Examples:
        Add a computed column that applies the model `facebook/detr-resnet-50-panoptic` to an existing
        Pixeltable column `image` of the table `tbl`:

        >>> tbl.add_computed_column(segmentation=detr_for_segmentation(
        ...     tbl.image,
        ...     model_id='facebook/detr-resnet-50-panoptic',
        ...     threshold=0.5
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import DetrForSegmentation, DetrImageProcessor

    model = _lookup_model(model_id, DetrForSegmentation.from_pretrained, device=device)
    processor = _lookup_processor(model_id, DetrImageProcessor.from_pretrained)
    normalized_images = [normalize_image_mode(img) for img in image]

    with torch.no_grad():
        inputs = processor(images=normalized_images, return_tensors='pt')
        outputs = model(**inputs.to(device))
        results = processor.post_process_panoptic_segmentation(
            outputs, threshold=threshold, target_sizes=[(img.height, img.width) for img in image]
        )

    output_list: list[dict] = []
    for result in results:
        seg_array = result['segmentation'].cpu().numpy()

        segments_info = [
            {
                'id': seg['id'],
                'label_id': seg['label_id'],
                'label_text': model.config.id2label[seg['label_id']],
                'score': seg['score'],
                'was_fused': seg['was_fused'],
            }
            for seg in result['segments_info']
        ]
        output_list.append({'segmentation': seg_array, 'segments_info': segments_info})

    return output_list


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
def speech2text_for_conditional_generation(audio: pxt.Audio, *, model_id: str, language: str | None = None) -> str:
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

    forced_bos_token_id: int | None = None if language is None else tokenizer.lang_code_to_id[language]

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


@pxt.udf
def text_generation(text: str, *, model_id: str, model_kwargs: dict[str, Any] | None = None) -> str:
    """
    Generates text using a pretrained language model. `model_id` should be a reference to a pretrained
    [text generation model](https://huggingface.co/models?pipeline_tag=text-generation).

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The input text to continue/complete.
        model_id: The pretrained model to use for text generation.
        model_kwargs: Additional keyword arguments to pass to the model's `generate` method, such as `max_length`,
            `temperature`, etc. See the
            [Hugging Face text_generation documentation](https://huggingface.co/docs/inference-providers/en/tasks/text-generation)
            for details.

    Returns:
        The generated text completion.

    Examples:
        Add a computed column that generates text completions using the `Qwen/Qwen3-0.6B` model:

        >>> tbl.add_computed_column(completion=text_generation(
        ...     tbl.prompt,
        ...     model_id='Qwen/Qwen3-0.6B',
        ...     model_kwargs={'temperature': 0.5, 'max_length': 150}
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_kwargs is None:
        model_kwargs = {}

    model = _lookup_model(model_id, AutoModelForCausalLM.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(**inputs.to(device), pad_token_id=tokenizer.eos_token_id, **model_kwargs)

    input_length = len(inputs['input_ids'][0])
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text


@pxt.udf(batch_size=16)
def text_classification(text: Batch[str], *, model_id: str, top_k: int = 5) -> Batch[list[dict[str, Any]]]:
    """
    Classifies text using a pretrained classification model. `model_id` should be a reference to a pretrained
    [text classification model](https://huggingface.co/models?pipeline_tag=text-classification)
    such as BERT, RoBERTa, or DistilBERT.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The text to classify.
        model_id: The pretrained model to use for classification.
        top_k: The number of top predictions to return.

    Returns:
        A dictionary containing classification results with scores, labels, and label text.

    Examples:
        Add a computed column for sentiment analysis:

        >>> tbl.add_computed_column(sentiment=text_classification(
        ...     tbl.review_text,
        ...     model_id='cardiffnlp/twitter-roberta-base-sentiment-latest'
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = _lookup_model(model_id, AutoModelForSequenceClassification.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs.to(device))
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)

    results = []
    for i in range(len(text)):
        # Return as list of individual classification items for HuggingFace compatibility
        classification_items = []
        for k in range(top_k_probs.shape[1]):
            classification_items.append(
                {
                    'label': top_k_indices[i, k].item(),
                    'label_text': model.config.id2label[top_k_indices[i, k].item()],
                    'score': top_k_probs[i, k].item(),
                }
            )
        results.append(classification_items)

    return results


@pxt.udf(batch_size=4)
def image_captioning(
    image: Batch[PIL.Image.Image], *, model_id: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[str]:
    """
    Generates captions for images using a pretrained image captioning model. `model_id` should be a reference to a
    pretrained [image-to-text model](https://huggingface.co/models?pipeline_tag=image-to-text) such as BLIP,
    Git, or LLaVA.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        image: The image to caption.
        model_id: The pretrained model to use for captioning.
        model_kwargs: Additional keyword arguments to pass to the model's `generate` method, such as `max_length`.

    Returns:
        The generated caption text.

    Examples:
        Add a computed column `caption` to an existing table `tbl` that generates captions using the
        `Salesforce/blip-image-captioning-base` model:

        >>> tbl.add_computed_column(caption=image_captioning(
        ...     tbl.image,
        ...     model_id='Salesforce/blip-image-captioning-base',
        ...     model_kwargs={'max_length': 30}
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if model_kwargs is None:
        model_kwargs = {}

    model = _lookup_model(model_id, AutoModelForVision2Seq.from_pretrained, device=device)
    processor = _lookup_processor(model_id, AutoProcessor.from_pretrained)
    normalized_images = [normalize_image_mode(img) for img in image]

    with torch.no_grad():
        inputs = processor(images=normalized_images, return_tensors='pt')
        outputs = model.generate(**inputs.to(device), **model_kwargs)

    captions = processor.batch_decode(outputs, skip_special_tokens=True)
    return captions


@pxt.udf(batch_size=8)
def summarization(text: Batch[str], *, model_id: str, model_kwargs: dict[str, Any] | None = None) -> Batch[str]:
    """
    Summarizes text using a pretrained summarization model. `model_id` should be a reference to a pretrained
    [summarization model](https://huggingface.co/models?pipeline_tag=summarization) such as BART, T5, or Pegasus.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The text to summarize.
        model_id: The pretrained model to use for summarization.
        model_kwargs: Additional keyword arguments to pass to the model's `generate` method, such as `max_length`.

    Returns:
        The generated summary text.

    Examples:
        Add a computed column that summarizes documents:

        >>> tbl.add_computed_column(summary=text_summarization(
        ...     tbl.document_text,
        ...     model_id='facebook/bart-large-cnn',
        ...     max_length=100
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if model_kwargs is None:
        model_kwargs = {}

    model = _lookup_model(model_id, AutoModelForSeq2SeqLM.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(**inputs.to(device), **model_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


@pxt.udf
def token_classification(
    text: str, *, model_id: str, aggregation_strategy: Literal['simple', 'first', 'average', 'max'] = 'simple'
) -> list[dict[str, Any]]:
    """
    Extracts named entities from text using a pretrained named entity recognition (NER) model.
    `model_id` should be a reference to a pretrained
    [token classification model](https://huggingface.co/models?pipeline_tag=token-classification) for NER.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The text to analyze for named entities.
        model_id: The pretrained model to use.
        aggregation_strategy: Method used to aggregate tokens.

    Returns:
        A list of dictionaries containing entity information (text, label, confidence, start, end).

    Examples:
        Add a computed column that extracts named entities:

        >>> tbl.add_computed_column(entities=token_classification(
        ...     tbl.text,
        ...     model_id='dbmdz/bert-large-cased-finetuned-conll03-english'
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    # Follow direct model loading pattern like other best practice functions
    model = _lookup_model(model_id, AutoModelForTokenClassification.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)

    # Validate aggregation strategy
    valid_strategies = {'simple', 'first', 'average', 'max'}
    if aggregation_strategy not in valid_strategies:
        raise excs.Error(
            f'Invalid aggregation_strategy {aggregation_strategy!r}. Must be one of: {", ".join(valid_strategies)}'
        )

    with torch.no_grad():
        # Tokenize with special tokens and return offsets for entity extraction
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        # Get model predictions
        outputs = model(**{k: v.to(device) for k, v in inputs.items() if k != 'offset_mapping'})
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the predicted labels and confidence scores
        predicted_token_classes = predictions.argmax(dim=-1).squeeze().tolist()
        confidence_scores = predictions.max(dim=-1).values.squeeze().tolist()

        # Handle single token case
        if not isinstance(predicted_token_classes, list):
            predicted_token_classes = [predicted_token_classes]
            confidence_scores = [confidence_scores]

        # Extract entities from predictions
        entities = []
        offset_mapping = inputs['offset_mapping'][0].tolist()

        current_entity = None

        for token_class, confidence, (start_offset, end_offset) in zip(
            predicted_token_classes, confidence_scores, offset_mapping
        ):
            # Skip special tokens (offset is (0, 0))
            if start_offset == 0 and end_offset == 0:
                continue

            label = model.config.id2label[token_class]

            # Skip 'O' (outside) labels
            if label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Parse BIO/BILOU tags
            if label.startswith('B-') or (label.startswith('I-') and current_entity is None):
                # Begin new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:] if label.startswith(('B-', 'I-')) else label
                current_entity = {
                    'word': text[start_offset:end_offset],
                    'entity_group': entity_type,
                    'score': float(confidence),
                    'start': start_offset,
                    'end': end_offset,
                }

            elif label.startswith('I-') and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if current_entity['entity_group'] == entity_type:
                    # Extend the current entity
                    current_entity['word'] = text[current_entity['start'] : end_offset]
                    current_entity['end'] = end_offset

                    # Update confidence based on aggregation strategy
                    if aggregation_strategy == 'average':
                        # Simple average (could be improved with token count weighting)
                        current_entity['score'] = (current_entity['score'] + float(confidence)) / 2
                    elif aggregation_strategy == 'max':
                        current_entity['score'] = max(current_entity['score'], float(confidence))
                    elif aggregation_strategy == 'first':
                        pass  # Keep first confidence
                    # 'simple' uses the same logic as 'first'
                else:
                    # Different entity type, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        'word': text[start_offset:end_offset],
                        'entity_group': entity_type,
                        'score': float(confidence),
                        'start': start_offset,
                        'end': end_offset,
                    }

        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        return entities


@pxt.udf
def question_answering(context: str, question: str, *, model_id: str) -> dict[str, Any]:
    """
    Answers questions based on provided context using a pretrained QA model. `model_id` should be a reference to a
    pretrained [question answering model](https://huggingface.co/models?pipeline_tag=question-answering) such as
    BERT or RoBERTa.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        context: The context text containing the answer.
        question: The question to answer.
        model_id: The pretrained QA model to use.

    Returns:
        A dictionary containing the answer, confidence score, and start/end positions.

    Examples:
        Add a computed column that answers questions based on document context:

        >>> tbl.add_computed_column(answer=question_answering(
        ...     tbl.document_text,
        ...     tbl.question,
        ...     model_id='deepset/roberta-base-squad2'
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    model = _lookup_model(model_id, AutoModelForQuestionAnswering.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)

    with torch.no_grad():
        # Tokenize the question and context
        inputs = tokenizer.encode_plus(
            question, context, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=512
        )

        # Get model predictions
        outputs = model(**inputs.to(device))
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Find the tokens with the highest start and end scores
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)

        # Ensure end_idx >= start_idx
        end_idx = torch.max(end_idx, start_idx)

        # Convert token positions to string
        input_ids = inputs['input_ids'][0]

        # Extract answer tokens
        answer_tokens = input_ids[start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Calculate confidence score
        start_probs = torch.softmax(start_scores, dim=1)
        end_probs = torch.softmax(end_scores, dim=1)
        confidence = float(start_probs[0][start_idx] * end_probs[0][end_idx])

        return {'answer': answer.strip(), 'score': confidence, 'start': int(start_idx), 'end': int(end_idx)}


@pxt.udf(batch_size=8)
def translation(
    text: Batch[str], *, model_id: str, src_lang: str | None = None, target_lang: str | None = None
) -> Batch[str]:
    """
    Translates text using a pretrained translation model. `model_id` should be a reference to a pretrained
    [translation model](https://huggingface.co/models?pipeline_tag=translation) such as MarianMT or T5.

    __Requirements:__

    - `pip install torch transformers sentencepiece`

    Args:
        text: The text to translate.
        model_id: The pretrained translation model to use.
        src_lang: Source language code (optional, can be inferred from model).
        target_lang: Target language code (optional, can be inferred from model).

    Returns:
        The translated text.

    Examples:
        Add a computed column that translates text:

        >>> tbl.add_computed_column(french_text=translation(
        ...     tbl.english_text,
        ...     model_id='Helsinki-NLP/opus-mt-en-fr',
        ...     src_lang='en',
        ...     target_lang='fr'
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = _lookup_model(model_id, AutoModelForSeq2SeqLM.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)
    lang_code_to_id: dict | None = getattr(tokenizer, 'lang_code_to_id', {})

    # Language validation - following speech2text_for_conditional_generation pattern
    if src_lang is not None and src_lang not in lang_code_to_id:
        raise excs.Error(
            f'Source language code {src_lang!r} is not supported by the model {model_id!r}. '
            f'Supported languages are: {list(lang_code_to_id.keys())}'
        )

    if target_lang is not None and target_lang not in lang_code_to_id:
        raise excs.Error(
            f'Target language code {target_lang!r} is not supported by the model {model_id!r}. '
            f'Supported languages are: {list(lang_code_to_id.keys())}'
        )

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Set forced_bos_token_id for target language if supported
        generate_kwargs = {'max_length': 512, 'num_beams': 4, 'early_stopping': True}

        if target_lang is not None:
            generate_kwargs['forced_bos_token_id'] = lang_code_to_id[target_lang]

        outputs = model.generate(**inputs.to(device), **generate_kwargs)

    # Decode all outputs at once
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


@pxt.udf
def text_to_image(
    prompt: str,
    *,
    model_id: str,
    height: int = 512,
    width: int = 512,
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> PIL.Image.Image:
    """
    Generates images from text prompts using a pretrained text-to-image model. `model_id` should be a reference to a
    pretrained [text-to-image model](https://huggingface.co/models?pipeline_tag=text-to-image) such as
    Stable Diffusion.

    __Requirements:__

    - `pip install torch transformers diffusers accelerate`

    Args:
        prompt: The text prompt describing the desired image.
        model_id: The pretrained text-to-image model to use.
        height: Height of the generated image in pixels.
        width: Width of the generated image in pixels.
        seed: Optional random seed for reproducibility.
        model_kwargs: Additional keyword arguments to pass to the model, such as `num_inference_steps`,
            `guidance_scale`, or `negative_prompt`.

    Returns:
        The generated Image.

    Examples:
        Add a computed column that generates images from text prompts:

        >>> tbl.add_computed_column(generated_image=text_to_image(
        ...     tbl.prompt,
        ...     model_id='stable-diffusion-v1.5/stable-diffusion-v1-5',
        ...     height=512,
        ...     width=512,
        ...     model_kwargs={'num_inference_steps': 25},
        ... ))
    """
    env.Env.get().require_package('transformers')
    env.Env.get().require_package('diffusers')
    env.Env.get().require_package('accelerate')
    device = resolve_torch_device('auto', allow_mps=False)
    import torch
    from diffusers import AutoPipelineForText2Image

    if model_kwargs is None:
        model_kwargs = {}

    # Parameter validation - following best practices pattern
    if height <= 0 or width <= 0:
        raise excs.Error(f'Height ({height}) and width ({width}) must be positive integers')

    if height % 8 != 0 or width % 8 != 0:
        raise excs.Error(f'Height ({height}) and width ({width}) must be divisible by 8 for most diffusion models')

    pipeline = _lookup_model(
        model_id,
        lambda x: AutoPipelineForText2Image.from_pretrained(
            x,
            dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            safety_checker=None,  # Disable safety checker for performance
            requires_safety_checker=False,
        ),
        device=device,
    )

    try:
        if device == 'cuda' and hasattr(pipeline, 'enable_model_cpu_offload'):
            pipeline.enable_model_cpu_offload()
        if hasattr(pipeline, 'enable_memory_efficient_attention'):
            pipeline.enable_memory_efficient_attention()
    except Exception:
        pass  # Ignore optimization failures

    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        result = pipeline(prompt, height=height, width=width, generator=generator, **model_kwargs)
        return result.images[0]


@pxt.udf
def text_to_speech(text: str, *, model_id: str, speaker_id: int | None = None, vocoder: str | None = None) -> pxt.Audio:
    """
    Converts text to speech using a pretrained TTS model. `model_id` should be a reference to a
    pretrained [text-to-speech model](https://huggingface.co/models?pipeline_tag=text-to-speech).

    __Requirements:__

    - `pip install torch transformers datasets soundfile`

    Args:
        text: The text to convert to speech.
        model_id: The pretrained TTS model to use.
        speaker_id: Speaker ID for multi-speaker models.
        vocoder: Optional vocoder model for higher quality audio.

    Returns:
        The generated audio file.

    Examples:
        Add a computed column that converts text to speech:

        >>> tbl.add_computed_column(audio=text_to_speech(
        ...     tbl.text_content,
        ...     model_id='microsoft/speecht5_tts',
        ...     speaker_id=0
        ... ))
    """
    env.Env.get().require_package('transformers')
    env.Env.get().require_package('datasets')
    env.Env.get().require_package('soundfile')
    device = resolve_torch_device('auto')
    import datasets  # type: ignore[import-untyped]
    import soundfile as sf  # type: ignore[import-untyped]
    import torch
    from transformers import (
        AutoModelForTextToWaveform,
        AutoProcessor,
        BarkModel,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
    )

    # Model loading with error handling - following best practices pattern
    if 'speecht5' in model_id.lower():
        model = _lookup_model(model_id, SpeechT5ForTextToSpeech.from_pretrained, device=device)
        processor = _lookup_processor(model_id, SpeechT5Processor.from_pretrained)
        vocoder_model_id = vocoder or 'microsoft/speecht5_hifigan'
        vocoder_model = _lookup_model(vocoder_model_id, SpeechT5HifiGan.from_pretrained, device=device)

    elif 'bark' in model_id.lower():
        model = _lookup_model(model_id, BarkModel.from_pretrained, device=device)
        processor = _lookup_processor(model_id, AutoProcessor.from_pretrained)
        vocoder_model = None

    else:
        model = _lookup_model(model_id, AutoModelForTextToWaveform.from_pretrained, device=device)
        processor = _lookup_processor(model_id, AutoProcessor.from_pretrained)
        vocoder_model = None

    # Load speaker embeddings once for SpeechT5 (following speech2text pattern)
    speaker_embeddings = None
    if 'speecht5' in model_id.lower():
        ds: datasets.Dataset
        if len(_speecht5_embeddings_dataset) == 0:
            ds = datasets.load_dataset(
                'Matthijs/cmu-arctic-xvectors', split='validation', revision='refs/convert/parquet'
            )
            _speecht5_embeddings_dataset.append(ds)
        else:
            assert len(_speecht5_embeddings_dataset) == 1
            ds = _speecht5_embeddings_dataset[0]
        speaker_embeddings = torch.tensor(ds[speaker_id or 7306]['xvector']).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate speech based on model type
        if 'speecht5' in model_id.lower():
            inputs = processor(text=text, return_tensors='pt').to(device)
            speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder_model)
            audio_np = speech.cpu().numpy()
            sample_rate = 16000

        elif 'bark' in model_id.lower():
            inputs = processor(text, return_tensors='pt').to(device)
            audio_array = model.generate(**inputs)
            audio_np = audio_array.cpu().numpy().squeeze()
            sample_rate = getattr(model.generation_config, 'sample_rate', 24000)

        else:
            # Generic approach for other TTS models
            inputs = processor(text, return_tensors='pt').to(device)
            audio_output = model(**inputs)
            audio_np = audio_output.waveform.cpu().numpy().squeeze()
            sample_rate = getattr(model.config, 'sample_rate', 22050)

        # Normalize audio - following consistent pattern
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.9

        # Create output file
        output_filename = str(TempStore.create_path(extension='.wav'))
        sf.write(output_filename, audio_np, sample_rate, format='WAV', subtype='PCM_16')
        return output_filename


@pxt.udf
def image_to_image(
    image: PIL.Image.Image,
    prompt: str,
    *,
    model_id: str,
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> PIL.Image.Image:
    """
    Transforms input images based on text prompts using a pretrained image-to-image model.
    `model_id` should be a reference to a pretrained
    [image-to-image model](https://huggingface.co/models?pipeline_tag=image-to-image) such as
    Stable Diffusion.

    __Requirements:__

    - `pip install torch transformers diffusers accelerate`

    Args:
        image: The input image to transform.
        prompt: The text prompt describing the desired transformation.
        model_id: The pretrained image-to-image model to use.
        seed: Random seed for reproducibility.
        model_kwargs: Additional keyword arguments to pass to the model, such as `strength`,
            `guidance_scale`, or `num_inference_steps`.

    Returns:
        The transformed image.

    Examples:
        Add a computed column that transforms images based on prompts:

        >>> tbl.add_computed_column(transformed=image_to_image(
        ...     tbl.source_image,
        ...     tbl.transformation_prompt,
        ...     model_id='stable-diffusion-v1-5/stable-diffusion-v1-5'
        ... ))

        With custom transformation strength:

        >>> tbl.add_computed_column(transformed=image_to_image(
        ...     tbl.source_image,
        ...     tbl.transformation_prompt,
        ...     model_id='stable-diffusion-v1-5/stable-diffusion-v1-5',
        ...     model_kwargs={'strength': 0.75, 'num_inference_steps': 50}
        ... ))
    """
    env.Env.get().require_package('transformers')
    env.Env.get().require_package('diffusers')
    env.Env.get().require_package('accelerate')
    device = resolve_torch_device('auto', allow_mps=False)
    import torch
    from diffusers import AutoPipelineForImage2Image

    if model_kwargs is None:
        model_kwargs = {}

    pipeline = _lookup_model(
        model_id,
        lambda x: AutoPipelineForImage2Image.from_pretrained(
            x,
            dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            safety_checker=None,  # Disable safety checker for performance
            requires_safety_checker=False,
        ),
        device=device,
    )

    try:
        if device == 'cuda' and hasattr(pipeline, 'enable_model_cpu_offload'):
            pipeline.enable_model_cpu_offload()
        if hasattr(pipeline, 'enable_memory_efficient_attention'):
            pipeline.enable_memory_efficient_attention()
    except Exception:
        pass  # Ignore optimization failures

    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    processed_image = image.convert('RGB')

    with torch.no_grad():
        result = pipeline(prompt=prompt, image=processed_image, generator=generator, **model_kwargs)
        return result.images[0]


@pxt.udf
def automatic_speech_recognition(
    audio: pxt.Audio,
    *,
    model_id: str,
    language: str | None = None,
    chunk_length_s: int | None = None,
    return_timestamps: bool = False,
) -> str:
    """
    Transcribes speech to text using a pretrained ASR model. `model_id` should be a reference to a
    pretrained [automatic-speech-recognition model](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition).

    This is a **generic function** that works with many ASR model families. For production use with
    specific models, consider specialized functions like `whisper.transcribe()` or
    `speech2text_for_conditional_generation()`.

    __Requirements:__

    - `pip install torch transformers torchaudio`

    __Recommended Models:__

    - **OpenAI Whisper**: `openai/whisper-tiny.en`, `openai/whisper-small`, `openai/whisper-base`
    - **Facebook Wav2Vec2**: `facebook/wav2vec2-base-960h`, `facebook/wav2vec2-large-960h-lv60-self`
    - **Microsoft SpeechT5**: `microsoft/speecht5_asr`
    - **Meta MMS (Multilingual)**: `facebook/mms-1b-all`

    Args:
        audio: The audio file(s) to transcribe.
        model_id: The pretrained ASR model to use.
        language: Language code for multilingual models (e.g., 'en', 'es', 'fr').
        chunk_length_s: Maximum length of audio chunks in seconds for long audio processing.
        return_timestamps: Whether to return word-level timestamps (model dependent).

    Returns:
        The transcribed text.

    Examples:
        Add a computed column that transcribes audio files:

        >>> tbl.add_computed_column(transcription=automatic_speech_recognition(
        ...     tbl.audio_file,
        ...     model_id='openai/whisper-tiny.en'  # Recommended
        ... ))

        Transcribe with language specification:

        >>> tbl.add_computed_column(transcription=automatic_speech_recognition(
        ...     tbl.audio_file,
        ...     model_id='facebook/mms-1b-all',
        ...     language='en'
        ... ))
    """
    env.Env.get().require_package('transformers')
    env.Env.get().require_package('torchaudio')
    device = resolve_torch_device('auto', allow_mps=False)  # Following speech2text pattern
    import torch
    import torchaudio

    # Try to load model and processor using direct model loading - following speech2text pattern
    # Handle different ASR model types
    if 'whisper' in model_id.lower():
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model = _lookup_model(model_id, WhisperForConditionalGeneration.from_pretrained, device=device)
        processor = _lookup_processor(model_id, WhisperProcessor.from_pretrained)

        # Language validation for Whisper - following speech2text pattern
        if language is not None and hasattr(processor.tokenizer, 'get_decoder_prompt_ids'):
            try:
                # Test if language is supported
                _ = processor.tokenizer.get_decoder_prompt_ids(language=language)
            except Exception:
                raise excs.Error(
                    f"Language code '{language}' is not supported by Whisper model '{model_id}'. "
                    f"Try common codes like 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'."
                ) from None

    elif 'wav2vec2' in model_id.lower():
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        model = _lookup_model(model_id, Wav2Vec2ForCTC.from_pretrained, device=device)
        processor = _lookup_processor(model_id, Wav2Vec2Processor.from_pretrained)

    elif 'speech_to_text' in model_id.lower() or 's2t' in model_id.lower():
        # Use the existing speech2text function for these models
        from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor

        model = _lookup_model(model_id, Speech2TextForConditionalGeneration.from_pretrained, device=device)
        processor = _lookup_processor(model_id, Speech2TextProcessor.from_pretrained)

    else:
        # Generic fallback using Auto classes
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        try:
            model = _lookup_model(model_id, AutoModelForSpeechSeq2Seq.from_pretrained, device=device)
            processor = _lookup_processor(model_id, AutoProcessor.from_pretrained)
        except Exception:
            # Fallback to CTC models
            from transformers import AutoModelForCTC

            model = _lookup_model(model_id, AutoModelForCTC.from_pretrained, device=device)
            processor = _lookup_processor(model_id, AutoProcessor.from_pretrained)

    # Get model's expected sampling rate - following speech2text pattern
    model_sampling_rate = getattr(model.config, 'sampling_rate', 16_000)

    # Load and preprocess audio - following speech2text pattern
    waveform, sampling_rate = torchaudio.load(audio)

    # Resample if necessary
    if sampling_rate != model_sampling_rate:
        waveform = torchaudio.transforms.Resample(sampling_rate, model_sampling_rate)(waveform)

    # Convert to mono if stereo
    if waveform.dim() == 2:
        waveform = torch.mean(waveform, dim=0)
    assert waveform.dim() == 1

    with torch.no_grad():
        # Process audio with the model
        inputs = processor(waveform, sampling_rate=model_sampling_rate, return_tensors='pt')

        # Handle different model types for generation
        if 'whisper' in model_id.lower():
            # Whisper-specific generation
            generate_kwargs = {}
            if language is not None:
                generate_kwargs['language'] = language
            if return_timestamps:
                generate_kwargs['return_timestamps'] = 'word' if return_timestamps else None

            generated_ids = model.generate(**inputs.to(device), **generate_kwargs)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elif hasattr(model, 'generate'):
            # Seq2Seq models (Speech2Text, etc.)
            generated_ids = model.generate(**inputs.to(device))
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        else:
            # CTC models (Wav2Vec2, etc.)
            logits = model(**inputs.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.strip()


@pxt.udf
def image_to_video(
    image: PIL.Image.Image,
    *,
    model_id: str,
    num_frames: int = 25,
    fps: int = 6,
    seed: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Generates videos from input images using a pretrained image-to-video model.
    `model_id` should be a reference to a pretrained
    [image-to-video model](https://huggingface.co/models?pipeline_tag=image-to-video).

    __Requirements:__

    - `pip install torch transformers diffusers accelerate`

    Args:
        image: The input image to animate into a video.
        model_id: The pretrained image-to-video model to use.
        num_frames: Number of video frames to generate.
        fps: Frames per second for the output video.
        seed: Random seed for reproducibility.
        model_kwargs: Additional keyword arguments to pass to the model, such as `num_inference_steps`,
            `motion_bucket_id`, or `guidance_scale`.

    Returns:
        The generated video file.

    Examples:
        Add a computed column that creates videos from images:

        >>> tbl.add_computed_column(video=image_to_video(
        ...     tbl.input_image,
        ...     model_id='stabilityai/stable-video-diffusion-img2vid-xt',
        ...     num_frames=25,
        ...     fps=7
        ... ))
    """
    env.Env.get().require_package('transformers')
    env.Env.get().require_package('diffusers')
    env.Env.get().require_package('accelerate')
    device = resolve_torch_device('auto', allow_mps=False)
    import numpy as np
    import torch
    from diffusers import StableVideoDiffusionPipeline

    if model_kwargs is None:
        model_kwargs = {}

    # Parameter validation - following best practices pattern
    if num_frames < 1:
        raise excs.Error(f'num_frames must be at least 1, got {num_frames}')

    if num_frames > 25:
        raise excs.Error(f'num_frames cannot exceed 25 for most video diffusion models, got {num_frames}')

    if fps < 1:
        raise excs.Error(f'fps must be at least 1, got {fps}')

    if fps > 60:
        raise excs.Error(f'fps should not exceed 60 for reasonable video generation, got {fps}')

    pipe = _lookup_model(
        model_id,
        lambda x: StableVideoDiffusionPipeline.from_pretrained(
            x,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            variant='fp16' if device == 'cuda' else None,
        ),
        device=device,
    )

    try:
        if device == 'cuda' and hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        if hasattr(pipe, 'enable_memory_efficient_attention'):
            pipe.enable_memory_efficient_attention()
    except Exception:
        pass  # Ignore optimization failures

    generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    # Ensure image is in RGB mode and proper size
    processed_image = image.convert('RGB')
    target_width, target_height = 512, 320
    processed_image = processed_image.resize((target_width, target_height), PIL.Image.Resampling.LANCZOS)

    # Generate video frames with proper error handling
    with torch.no_grad():
        result = pipe(image=processed_image, num_frames=num_frames, generator=generator, **model_kwargs)
        frames = result.frames[0]

    # Create output video file
    output_path = str(TempStore.create_path(extension='.mp4'))

    with av.open(output_path, mode='w') as container:
        stream = container.add_stream('h264', rate=fps)
        stream.width = target_width
        stream.height = target_height
        stream.pix_fmt = 'yuv420p'

        # Set codec options for better compatibility
        stream.codec_context.options = {'crf': '23', 'preset': 'medium'}

        for frame_pil in frames:
            # Convert PIL to numpy array
            frame_array = np.array(frame_pil)
            # Create av VideoFrame
            av_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
            # Encode and mux
            for packet in stream.encode(av_frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

    return output_path


def _lookup_model(
    model_id: str, create: Callable[..., T], device: str | None = None, pass_device_to_create: bool = False
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


_model_cache: dict[tuple[str, Callable, str | None], Any] = {}
_speecht5_embeddings_dataset: list[Any] = []  # contains only the speecht5 embeddings loaded by text_to_speech()
_processor_cache: dict[tuple[str, Callable], Any] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
