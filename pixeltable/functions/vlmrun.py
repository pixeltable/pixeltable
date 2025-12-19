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


@pxt.udf(resource_pool='request-rate:vlmrun')
async def chat_completions(
    messages: list[dict[str, Any]],
    *,
    model: str = 'vlmrun-orion-1:auto',
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
        model_kwargs: Additional keyword args for the VLM Run chat completions API.

    Returns:
        A JSON object containing the response and other metadata.

    Examples:
        Add a computed column that analyzes an uploaded file:

        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.media.localpath))
        >>> messages = [{'role': 'user', 'content': [
        ...     {'type': 'text', 'text': 'Describe this file'},
        ...     {'type': 'input_file', 'file_id': t.file_id}
        ... ]}]
        >>> t.add_computed_column(response=vlmrun.chat_completions(messages, model='vlmrun-orion-1:auto'))
    """
    Env.get().require_package('vlmrun')

    if model_kwargs is None:
        model_kwargs = {}

    def _call_api() -> dict:
        client = _vlmrun_client()
        response = client.agent.completions.create(
            model=model,
            messages=messages,
            **model_kwargs,
        )
        return response.model_dump()

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def get_artifact(artifact_id: str) -> str:
    """
    Downloads an artifact from VLM Run and returns the local file path.

    Use this to retrieve generated files (e.g., redacted documents, generated images)
    from VLM Run's artifact store.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        artifact_id: The artifact ID or URL returned from a chat completion.

    Returns:
        The local file path to the downloaded artifact.

    Examples:
        Download an artifact from VLM Run:

        >>> t.add_computed_column(artifact_path=vlmrun.get_artifact(t.artifact_id))
    """
    import urllib.request
    from urllib.parse import urlparse

    from pixeltable.utils.local_store import TempStore

    Env.get().require_package('vlmrun')

    def _call_api() -> str:
        # If it's a URL, download directly
        if artifact_id.startswith('http'):
            parsed = urlparse(artifact_id)
            ext = Path(parsed.path).suffix or '.png'
            local_path = TempStore.create_path(extension=ext)
            urllib.request.urlretrieve(artifact_id, local_path)
            return str(local_path)

        # Otherwise, use the artifacts API
        client = _vlmrun_client()
        artifact = client.artifacts.get(artifact_id)

        # Download from the artifact URL
        parsed = urlparse(artifact.url)
        ext = Path(parsed.path).suffix or '.png'
        local_path = TempStore.create_path(extension=ext)
        urllib.request.urlretrieve(artifact.url, local_path)
        return str(local_path)

    result = await asyncio.to_thread(_call_api)
    return result


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
