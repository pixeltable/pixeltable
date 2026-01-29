"""
Pixeltable UDFs that wrap the VLM Run API.

In order to use them, you must first `pip install vlmrun` and configure your VLM Run API key,
as described in the [VLM Run documentation](https://docs.vlm.run/).
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from vlmrun.client import VLMRun


@register_client('vlmrun')
def _(api_key: str) -> 'VLMRun':
    from vlmrun.client import VLMRun

    return VLMRun(api_key=api_key, base_url='https://agent.vlm.run/v1')


def _vlmrun_client() -> 'VLMRun':
    return Env.get().get_client('vlmrun')


_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf', '.mp4', '.mov', '.avi', '.mkv', '.webm'}


@pxt.udf(resource_pool='request-rate:vlmrun')
async def upload_file(path: str) -> str:
    """
    Uploads a file to VLM Run and returns the file_id.

    Supports images, documents, and videos. The returned file_id can be used
    in messages for `chat_completions()`.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        path: The local file path to upload. For Image columns, use `.localpath`.
            For Video and Document columns, pass the column directly (they are already file paths).

    Returns:
        The file_id string for use in chat completion messages.

    Raises:
        ValueError: If the file format is not supported.

    Examples:
        Upload an image (requires .localpath since Image is PIL.Image):

        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.image.localpath))

        Upload a video or document (already file paths):

        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.video))
        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.document))
    """
    Env.get().require_package('vlmrun')

    def _call_api() -> str:
        file_path = Path(path)
        ext = file_path.suffix.lower()

        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        client = _vlmrun_client()
        uploaded = client.files.upload(file=file_path)
        return uploaded.id

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def upload_image(image: pxt.Image) -> str:
    """
    Uploads an image to VLM Run and returns the file_id.

    This is a convenience function that accepts a Pixeltable Image directly,
    without needing to use `.localpath`.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        image: The image to upload.

    Returns:
        The file_id string for use in chat completion messages.

    Examples:
        Upload an image directly:

        >>> t.add_computed_column(file_id=vlmrun.upload_image(t.image))
    """
    from pixeltable.utils.local_store import TempStore

    Env.get().require_package('vlmrun')

    def _call_api() -> str:
        # Save PIL Image to temp file
        local_path = TempStore.create_path(extension='.png')
        image.save(local_path)

        client = _vlmrun_client()
        uploaded = client.files.upload(file=Path(local_path))
        return uploaded.id

    result = await asyncio.to_thread(_call_api)
    return result


def _build_artifact_schema(output_artifacts: list[str]) -> dict | None:
    """Build a JSON schema for the requested artifact types.

    Supports both singular and plural artifact types:
    - Singular: 'image', 'video', 'audio', 'document', 'recon' -> single artifact
    - Plural: 'images', 'videos', 'audios', 'documents', 'recons' -> list of artifacts
    """
    from pydantic import BaseModel, Field
    from vlmrun.types.refs import AudioRef, DocumentRef, ImageRef, ReconRef, VideoRef

    # Map for singular artifact types
    singular_ref_map = {
        'image': ImageRef,
        'video': VideoRef,
        'audio': AudioRef,
        'document': DocumentRef,
        'recon': ReconRef,
    }

    # Map for plural artifact types (list versions)
    plural_ref_map = {
        'images': ('List[ImageRef]', ImageRef, 'List of images'),
        'videos': ('List[VideoRef]', VideoRef, 'List of videos'),
        'audios': ('List[AudioRef]', AudioRef, 'List of audio files'),
        'documents': ('List[DocumentRef]', DocumentRef, 'List of documents'),
        'recons': ('List[ReconRef]', ReconRef, 'List of 3D reconstructions'),
    }

    annotations = {}
    field_defaults = {}

    for artifact_type in output_artifacts:
        if artifact_type in singular_ref_map:
            annotations[artifact_type] = singular_ref_map[artifact_type]
        elif artifact_type in plural_ref_map:
            _, ref_class, description = plural_ref_map[artifact_type]
            annotations[artifact_type] = list[ref_class]
            field_defaults[artifact_type] = Field(..., description=description)

    if not annotations:
        return None

    # Create dynamic Pydantic model with field definitions
    namespace = {'__annotations__': annotations}
    namespace.update(field_defaults)
    DynamicModel = type('ArtifactResponse', (BaseModel,), namespace)
    return DynamicModel.model_json_schema()


@pxt.udf(resource_pool='request-rate:vlmrun')
async def chat_completions(
    messages: list[dict[str, Any]],
    *,
    model: str = 'vlmrun-orion-1:auto',
    output_artifacts: list[str] | None = None,
    toolsets: list[str] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> pxt.Json:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the VLM Run chat completions API endpoint.
    For additional details, see: <https://docs.vlm.run/>

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: The model to use. Options: `vlmrun-orion-1:fast`, `vlmrun-orion-1:auto`,
            `vlmrun-orion-1:pro`. Defaults to `vlmrun-orion-1:auto`.
        output_artifacts: List of artifact types to generate.
            Singular options: 'image', 'video', 'audio', 'document', 'recon' -> returns single artifact ID.
            Plural options: 'images', 'videos', 'audios', 'documents', 'recons' -> returns list of artifact IDs.
            When specified, auto-generates the response schema. The response will include
            object_ids (e.g., 'img_abc123') or lists of object_ids that can be retrieved
            using get_image_artifact, get_video_artifact, get_recon_artifact, etc.
        toolsets: List of tool categories to enable. Options: 'core', 'image', 'image-gen',
            'world_gen', 'viz', 'document', 'video', 'web'. If not specified, all tools are available.
        model_kwargs: Additional keyword args for the VLM Run chat completions API.

    Returns:
        A JSON object containing the response, session_id, and artifact object_ids.

    Examples:
        Text response (no artifacts):

        >>> messages = [{'role': 'user', 'content': [
        ...     {'type': 'text', 'text': 'Describe this image'},
        ...     {'type': 'input_file', 'file_id': t.file_id}
        ... ]}]
        >>> t.add_computed_column(response=vlmrun.chat_completions(messages))

        Generate image artifact:

        >>> messages = [{'role': 'user', 'content': [
        ...     {'type': 'text', 'text': 'Blur all faces in this image'},
        ...     {'type': 'input_file', 'file_id': t.file_id}
        ... ]}]
        >>> t.add_computed_column(response=vlmrun.chat_completions(
        ...     messages, output_artifacts=['image']
        ... ))
        >>> t.add_computed_column(blurred=vlmrun.get_image_artifact(
        ...     t.response['image'], session_id=t.response['session_id']
        ... ))

        Generate multiple image artifacts (list):

        >>> messages = [{'role': 'user', 'content': [
        ...     {'type': 'text', 'text': 'Extract all frames with faces'},
        ...     {'type': 'input_file', 'file_id': t.file_id}
        ... ]}]
        >>> t.add_computed_column(response=vlmrun.chat_completions(
        ...     messages, output_artifacts=['images']  # plural for list
        ... ))
        >>> # response['images'] contains a list of image IDs like ['img_1', 'img_2', ...]

        Use specific toolsets for 3D reconstruction:

        >>> t.add_computed_column(response=vlmrun.chat_completions(
        ...     messages, output_artifacts=['recon'], toolsets=['world_gen']
        ... ))

        Use document processing tools only:

        >>> t.add_computed_column(response=vlmrun.chat_completions(
        ...     messages, toolsets=['document']
        ... ))
    """
    Env.get().require_package('vlmrun')

    if model_kwargs is None:
        model_kwargs = {}

    # Auto-generate schema if output_artifacts specified
    if output_artifacts:
        schema = _build_artifact_schema(output_artifacts)
        if schema:
            model_kwargs['response_format'] = {'type': 'json_schema', 'schema': schema}

    def _call_api() -> dict:
        client = _vlmrun_client()

        # Build extra_body for VLM Run-specific parameters
        extra_body = model_kwargs.pop('extra_body', {}) if model_kwargs else {}
        if toolsets:
            extra_body['toolsets'] = toolsets

        response = client.agent.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body if extra_body else None,
            **model_kwargs,
        )
        result = response.model_dump()
        # Include session_id at top level for artifact retrieval
        # session_id may be a direct attribute or in model_extra (Pydantic extra fields)
        if hasattr(response, 'session_id') and response.session_id is not None:
            result['session_id'] = response.session_id
        elif hasattr(response, 'model_extra') and 'session_id' in response.model_extra:
            result['session_id'] = response.model_extra['session_id']

        # Parse artifact object_ids from content if output_artifacts was used
        if output_artifacts and result.get('choices'):
            import json
            try:
                content = result['choices'][0]['message']['content']
                parsed = json.loads(content)
                # Add parsed artifact IDs to top level for easy access
                # Singular: {"image": {"id": "img_abc123"}} -> extract ID string
                # Plural: {"images": [{"id": "img_1"}, {"id": "img_2"}]} -> extract list of IDs
                for artifact_type in output_artifacts:
                    if artifact_type in parsed:
                        artifact_data = parsed[artifact_type]
                        # Handle list of artifacts (plural types like 'images', 'videos')
                        if isinstance(artifact_data, list):
                            ids = []
                            for item in artifact_data:
                                if isinstance(item, dict) and 'id' in item:
                                    ids.append(item['id'])
                                elif isinstance(item, str):
                                    ids.append(item)
                            result[artifact_type] = ids
                        # Handle single artifact
                        elif isinstance(artifact_data, dict) and 'id' in artifact_data:
                            result[artifact_type] = artifact_data['id']
                        elif isinstance(artifact_data, str):
                            # Already a string ID
                            result[artifact_type] = artifact_data
                        else:
                            # Store as-is for debugging
                            result[artifact_type] = artifact_data
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        return result

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def get_image_artifact(
    object_id: str,
    *,
    session_id: str | None = None,
    execution_id: str | None = None,
) -> pxt.Image:
    """
    Downloads an image artifact from VLM Run.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        object_id: The object_id from the structured response (e.g., 'img_abc123').
        session_id: The session_id from chat_completions.
        execution_id: The execution_id from agent executions.

    Returns:
        The image as a PIL Image.

    Examples:
        >>> t.add_computed_column(image=vlmrun.get_image_artifact(
        ...     t.response['image'], session_id=t.response['session_id']
        ... ))
    """
    import PIL.Image

    Env.get().require_package('vlmrun')

    if not session_id and not execution_id:
        raise ValueError("Either session_id or execution_id must be provided")

    def _call_api() -> PIL.Image.Image:
        client = _vlmrun_client()
        if session_id:
            return client.artifacts.get(object_id=object_id, session_id=session_id)
        return client.artifacts.get(object_id=object_id, execution_id=execution_id)

    return await asyncio.to_thread(_call_api)


@pxt.udf(resource_pool='request-rate:vlmrun')
async def get_video_artifact(
    object_id: str,
    *,
    session_id: str | None = None,
    execution_id: str | None = None,
) -> pxt.Video:
    """
    Downloads a video artifact from VLM Run.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        object_id: The object_id from the structured response (e.g., 'vid_abc123').
        session_id: The session_id from chat_completions.
        execution_id: The execution_id from agent executions.

    Returns:
        The local file path to the video.

    Examples:
        >>> t.add_computed_column(video=vlmrun.get_video_artifact(
        ...     t.response['video'], session_id=t.response['session_id']
        ... ))
    """
    Env.get().require_package('vlmrun')

    if not session_id and not execution_id:
        raise ValueError("Either session_id or execution_id must be provided")

    def _call_api() -> str:
        client = _vlmrun_client()
        if session_id:
            return str(client.artifacts.get(object_id=object_id, session_id=session_id))
        return str(client.artifacts.get(object_id=object_id, execution_id=execution_id))

    return await asyncio.to_thread(_call_api)


@pxt.udf(resource_pool='request-rate:vlmrun')
async def get_audio_artifact(
    object_id: str,
    *,
    session_id: str | None = None,
    execution_id: str | None = None,
) -> pxt.Audio:
    """
    Downloads an audio artifact from VLM Run.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        object_id: The object_id from the structured response (e.g., 'aud_abc123').
        session_id: The session_id from chat_completions.
        execution_id: The execution_id from agent executions.

    Returns:
        The local file path to the audio.

    Examples:
        >>> t.add_computed_column(audio=vlmrun.get_audio_artifact(
        ...     t.response['audio'], session_id=t.response['session_id']
        ... ))
    """
    Env.get().require_package('vlmrun')

    if not session_id and not execution_id:
        raise ValueError("Either session_id or execution_id must be provided")

    def _call_api() -> str:
        client = _vlmrun_client()
        if session_id:
            return str(client.artifacts.get(object_id=object_id, session_id=session_id))
        return str(client.artifacts.get(object_id=object_id, execution_id=execution_id))

    return await asyncio.to_thread(_call_api)


@pxt.udf(resource_pool='request-rate:vlmrun')
async def get_document_artifact(
    object_id: str,
    *,
    session_id: str | None = None,
    execution_id: str | None = None,
) -> pxt.Document:
    """
    Downloads a document artifact from VLM Run.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        object_id: The object_id from the structured response (e.g., 'doc_abc123').
        session_id: The session_id from chat_completions.
        execution_id: The execution_id from agent executions.

    Returns:
        The local file path to the document.

    Examples:
        >>> t.add_computed_column(document=vlmrun.get_document_artifact(
        ...     t.response['document'], session_id=t.response['session_id']
        ... ))
    """
    Env.get().require_package('vlmrun')

    if not session_id and not execution_id:
        raise ValueError("Either session_id or execution_id must be provided")

    def _call_api() -> str:
        client = _vlmrun_client()
        if session_id:
            return str(client.artifacts.get(object_id=object_id, session_id=session_id))
        return str(client.artifacts.get(object_id=object_id, execution_id=execution_id))

    return await asyncio.to_thread(_call_api)


@pxt.udf(resource_pool='request-rate:vlmrun')
async def get_recon_artifact(
    object_id: str,
    *,
    session_id: str | None = None,
    execution_id: str | None = None,
) -> str:
    """
    Downloads a 3D reconstruction artifact from VLM Run.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        object_id: The object_id from the structured response (e.g., 'recon_abc123').
        session_id: The session_id from chat_completions.
        execution_id: The execution_id from agent executions.

    Returns:
        The local file path to the 3D reconstruction file (e.g., .glb, .obj).

    Examples:
        >>> t.add_computed_column(recon=vlmrun.get_recon_artifact(
        ...     t.response['recon'], session_id=t.response['session_id']
        ... ))
    """
    Env.get().require_package('vlmrun')

    if not session_id and not execution_id:
        raise ValueError("Either session_id or execution_id must be provided")

    def _call_api() -> str:
        client = _vlmrun_client()
        if session_id:
            return str(client.artifacts.get(object_id=object_id, session_id=session_id))
        return str(client.artifacts.get(object_id=object_id, execution_id=execution_id))

    return await asyncio.to_thread(_call_api)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
