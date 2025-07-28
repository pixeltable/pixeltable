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


@pxt.udf(batch_size=8)
def text_generation(
    text: Batch[str], *, model_id: str, max_length: int = 100, temperature: float = 0.7, do_sample: bool = True
) -> Batch[str]:
    """
    Generates text using a pretrained language model. `model_id` should be a reference to a pretrained
    [text generation model](https://huggingface.co/models?pipeline_tag=text-generation) such as GPT-2, GPT-J, or T5.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The input text to continue/complete.
        model_id: The pretrained model to use for text generation.
        max_length: Maximum length of the generated text.
        temperature: Controls randomness in generation (lower = more deterministic).
        do_sample: Whether to use sampling for generation.

    Returns:
        The generated text completion.

    Examples:
        Add a computed column that generates text completions using GPT-2:

        >>> tbl.add_computed_column(completion=text_generation(
        ...     tbl.prompt,
        ...     model_id='gpt2-medium',
        ...     max_length=150
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = _lookup_model(model_id, AutoModelForCausalLM.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(
            **inputs.to(device),
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        
    results = []
    for i, output in enumerate(outputs):
        input_length = len(inputs['input_ids'][i])
        generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        results.append(generated_text)
    
    return results


@pxt.udf(batch_size=16)
def text_classification(
    text: Batch[str], *, model_id: str, top_k: int = 5
) -> Batch[dict[str, Any]]:
    """
    Classifies text using a pretrained classification model. `model_id` should be a reference to a pretrained
    [text classification model](https://huggingface.co/models?pipeline_tag=text-classification) such as BERT, RoBERTa, or DistilBERT.

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
    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, logits.shape[-1]), dim=-1)

    results = []
    for i in range(len(text)):
        results.append({
            'scores': [top_k_probs[i, k].item() for k in range(top_k_probs.shape[1])],
            'labels': [top_k_indices[i, k].item() for k in range(top_k_indices.shape[1])],
            'label_text': [model.config.id2label[top_k_indices[i, k].item()] for k in range(top_k_indices.shape[1])],
        })
    
    return results


@pxt.udf(batch_size=4)
def image_captioning(
    image: Batch[PIL.Image.Image], *, model_id: str, max_length: int = 50
) -> Batch[str]:
    """
    Generates captions for images using a pretrained image captioning model. `model_id` should be a reference to a
    pretrained [image-to-text model](https://huggingface.co/models?pipeline_tag=image-to-text) such as BLIP or Git.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        image: The image to caption.
        model_id: The pretrained model to use for captioning.
        max_length: Maximum length of the generated caption.

    Returns:
        The generated caption text.

    Examples:
        Add a computed column that generates captions for images:

        >>> tbl.add_computed_column(caption=image_captioning(
        ...     tbl.image,
        ...     model_id='Salesforce/blip-image-captioning-base'
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model = _lookup_model(model_id, BlipForConditionalGeneration.from_pretrained, device=device)
    processor = _lookup_processor(model_id, BlipProcessor.from_pretrained)
    normalized_images = [normalize_image_mode(img) for img in image]

    with torch.no_grad():
        inputs = processor(images=normalized_images, return_tensors='pt')
        outputs = model.generate(**inputs.to(device), max_length=max_length)
        
    captions = processor.batch_decode(outputs, skip_special_tokens=True)
    return captions


@pxt.udf(batch_size=8)
def text_summarization(
    text: Batch[str], *, model_id: str, max_length: int = 150, min_length: int = 30
) -> Batch[str]:
    """
    Summarizes text using a pretrained summarization model. `model_id` should be a reference to a pretrained
    [summarization model](https://huggingface.co/models?pipeline_tag=summarization) such as BART, T5, or Pegasus.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The text to summarize.
        model_id: The pretrained model to use for summarization.
        max_length: Maximum length of the summary.
        min_length: Minimum length of the summary.

    Returns:
        The generated summary text.

    Examples:
        Add a computed column that summarizes documents:

        >>> tbl.add_computed_column(summary=text_summarization(
        ...     tbl.document_text,
        ...     model_id='facebook/bart-base-cnn',
        ...     max_length=100
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = _lookup_model(model_id, AutoModelForSeq2SeqLM.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        outputs = model.generate(
            **inputs.to(device),
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
    summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return summaries


@pxt.udf(batch_size=16)
def named_entity_recognition(
    text: Batch[str], *, model_id: str, aggregation_strategy: str = 'simple'
) -> Batch[list[dict[str, Any]]]:
    """
    Extracts named entities from text using a pretrained NER model. `model_id` should be a reference to a pretrained
    [token classification model](https://huggingface.co/models?pipeline_tag=token-classification) for NER.

    __Requirements:__

    - `pip install torch transformers`

    Args:
        text: The text to analyze for named entities.
        model_id: The pretrained NER model to use.
        aggregation_strategy: How to aggregate tokens ('simple', 'first', 'average', 'max').

    Returns:
        A list of dictionaries containing entity information (text, label, confidence, start, end).

    Examples:
        Add a computed column that extracts named entities:

        >>> tbl.add_computed_column(entities=named_entity_recognition(
        ...     tbl.text,
        ...     model_id='dbmdz/bert-large-cased-finetuned-conll03-english'
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    # For NER, we use the pipeline for easier entity aggregation
    if not hasattr(named_entity_recognition, '_pipeline_cache'):
        named_entity_recognition._pipeline_cache = {}
    
    if model_id not in named_entity_recognition._pipeline_cache:
        named_entity_recognition._pipeline_cache[model_id] = pipeline(
            'ner',
            model=model_id,
            tokenizer=model_id,
            aggregation_strategy=aggregation_strategy,
            device=0 if device == 'cuda' else -1
        )
    
    ner_pipeline = named_entity_recognition._pipeline_cache[model_id]
    results = []
    
    for txt in text:
        entities = ner_pipeline(txt)
        # Convert to consistent format
        formatted_entities = []
        for entity in entities:
            # Convert numpy types to Python types for JSON serialization
            confidence = entity.get('score', 0.0)
            if hasattr(confidence, 'item'):  # numpy scalar
                confidence = confidence.item()
            else:
                confidence = float(confidence)
            
            start = entity.get('start', 0)
            if hasattr(start, 'item'):  # numpy scalar
                start = start.item()
            else:
                start = int(start)
                
            end = entity.get('end', 0)
            if hasattr(end, 'item'):  # numpy scalar
                end = end.item()
            else:
                end = int(end)
            
            formatted_entities.append({
                'text': str(entity.get('word', '')),
                'label': str(entity.get('entity_group', entity.get('entity', ''))),
                'confidence': confidence,
                'start': start,
                'end': end
            })
        results.append(formatted_entities)
    
    return results


@pxt.udf(batch_size=8)
def question_answering(
    context: Batch[str], question: Batch[str], *, model_id: str
) -> Batch[dict[str, Any]]:
    """
    Answers questions based on provided context using a pretrained QA model. `model_id` should be a reference to a
    pretrained [question answering model](https://huggingface.co/models?pipeline_tag=question-answering) such as BERT or RoBERTa.

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

    results = []
    
    with torch.no_grad():
        for ctx, q in zip(context, question):
            inputs = tokenizer(q, ctx, return_tensors='pt', truncation=True, max_length=512)
            outputs = model(**inputs.to(device))
            
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
            
            if end_idx >= start_idx:
                answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                confidence = float(torch.softmax(start_logits, dim=-1)[0][start_idx] * 
                                 torch.softmax(end_logits, dim=-1)[0][end_idx])
            else:
                answer = ""
                confidence = 0.0
            
            results.append({
                'answer': answer,
                'confidence': confidence,
                'start': int(start_idx),
                'end': int(end_idx)
            })
    
    return results


@pxt.udf(batch_size=8)
def translation(
    text: Batch[str], *, model_id: str, src_lang: Optional[str] = None, tgt_lang: Optional[str] = None
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
        tgt_lang: Target language code (optional, can be inferred from model).

    Returns:
        The translated text.

    Examples:
        Add a computed column that translates text:

        >>> tbl.add_computed_column(french_text=translation(
        ...     tbl.english_text,
        ...     model_id='Helsinki-NLP/opus-mt-en-fr',
        ...     src_lang='en',
        ...     tgt_lang='fr'
        ... ))
    """
    env.Env.get().require_package('transformers')
    device = resolve_torch_device('auto')
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = _lookup_model(model_id, AutoModelForSeq2SeqLM.from_pretrained, device=device)
    tokenizer = _lookup_processor(model_id, AutoTokenizer.from_pretrained)

    results = []
    
    with torch.no_grad():
        for txt in text:
            inputs = tokenizer(txt, return_tensors='pt', truncation=True, max_length=512)
            outputs = model.generate(**inputs.to(device), max_length=512, num_beams=4, early_stopping=True)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(translated)
    
    return results


@pxt.udf(batch_size=2)
def text_to_image(
    prompt: Batch[str], *, model_id: str, height: int = 512, width: int = 512, 
    num_inference_steps: int = 20, guidance_scale: float = 7.5, seed: Optional[int] = None
) -> Batch[PIL.Image.Image]:
    """
    Generates images from text prompts using a pretrained text-to-image model. `model_id` should be a reference to a
    pretrained [text-to-image model](https://huggingface.co/models?pipeline_tag=text-to-image) such as Stable Diffusion or FLUX.

    __Requirements:__

    - `pip install torch transformers diffusers accelerate`

    Args:
        prompt: The text prompt describing the desired image.
        model_id: The pretrained text-to-image model to use.
        height: Height of the generated image in pixels.
        width: Width of the generated image in pixels.
        num_inference_steps: Number of denoising steps (more steps = higher quality, slower).
        guidance_scale: How closely to follow the prompt (higher = more adherence).
        seed: Random seed for reproducible generation.

    Returns:
        The generated PIL Image.

    Examples:
        Add a computed column that generates images from text prompts:

        >>> tbl.add_computed_column(generated_image=text_to_image(
        ...     tbl.prompt,
        ...     model_id='runwayml/stable-diffusion-v1-5',
        ...     height=512,
        ...     width=512,
        ...     num_inference_steps=25
        ... ))
    """
    env.Env.get().require_package('transformers')
    env.Env.get().require_package('diffusers')
    env.Env.get().require_package('accelerate')
    device = resolve_torch_device('auto')
    import torch
    from diffusers import AutoPipelineForText2Image
    import random

    # Cache the pipeline to avoid reloading
    if not hasattr(text_to_image, '_pipeline_cache'):
        text_to_image._pipeline_cache = {}
    
    if model_id not in text_to_image._pipeline_cache:
        text_to_image._pipeline_cache[model_id] = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None
        )
        if device == 'cuda':
            text_to_image._pipeline_cache[model_id].enable_model_cpu_offload()
    
    pipeline = text_to_image._pipeline_cache[model_id]
    results = []
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    for prompt_text in prompt:
        try:
            # Generate image
            image = pipeline(
                prompt_text,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            results.append(image)
            
        except Exception as e:
            # Return a blank image on error to maintain batch consistency
            import PIL.Image
            blank_image = PIL.Image.new('RGB', (width, height), (128, 128, 128))
            results.append(blank_image)
            print(f"Warning: Text-to-image generation failed for prompt '{prompt_text}': {e}")
    
    return results


@pxt.udf(batch_size=4)
def text_to_speech(
    text: Batch[str], *, model_id: str, speaker_id: Optional[int] = None, 
    vocoder: Optional[str] = None
) -> Batch[bytes]:
    """
    Converts text to speech using a pretrained TTS model. `model_id` should be a reference to a
    pretrained [text-to-speech model](https://huggingface.co/models?pipeline_tag=text-to-speech).

    __Requirements:__

    - `pip install torch transformers datasets soundfile librosa`

    Args:
        text: The text to convert to speech.
        model_id: The pretrained TTS model to use.
        speaker_id: Speaker ID for multi-speaker models.
        vocoder: Optional vocoder model for higher quality audio.

    Returns:
        The generated audio as bytes (WAV format).

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
    import torch
    from transformers import pipeline
    import soundfile as sf
    import io

    # Cache the pipeline to avoid reloading
    if not hasattr(text_to_speech, '_pipeline_cache'):
        text_to_speech._pipeline_cache = {}
    
    if model_id not in text_to_speech._pipeline_cache:
        text_to_speech._pipeline_cache[model_id] = pipeline(
            'text-to-speech',
            model=model_id,
            device=0 if device == 'cuda' else -1
        )
    
    tts_pipeline = text_to_speech._pipeline_cache[model_id]
    results = []
    
    for text_input in text:
        try:
            # Generate speech
            speech_output = tts_pipeline(text_input)
            
            # Convert to bytes (WAV format)
            audio_data = speech_output['audio']
            sample_rate = speech_output['sampling_rate']
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
            
            results.append(audio_bytes)
            
        except Exception as e:
            # Return empty bytes on error
            results.append(b'')
            print(f"Warning: Text-to-speech generation failed for text '{text_input[:50]}...': {e}")
    
    return results


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
