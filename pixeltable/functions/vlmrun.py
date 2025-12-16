"""
Pixeltable UDFs that wrap the VLM Run Orion chat completions API.

In order to use them, you must first `pip install vlmrun` and configure your VLM Run API key,
as described in the [VLM Run documentation](https://docs.vlm.run/).
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from vlmrun.client import VLMRun


@env.register_client('vlmrun')
def _(api_key: str) -> 'VLMRun':
    from vlmrun.client import VLMRun

    return VLMRun(api_key=api_key, base_url='https://agent.vlm.run/v1')


def _vlmrun_client() -> 'VLMRun':
    return env.Env.get().get_client('vlmrun')


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf', '.mp4', '.mov', '.avi', '.mkv'}


@pxt.udf(resource_pool='request-rate:vlmrun')
async def upload_file(path: str) -> str:
    """
    Uploads a file to VLM Run and returns the file_id.

    Supports images, documents, and videos. The returned file_id can be used
    with `chat_completions()` to run analyses without re-uploading.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        path: The file path to upload.

    Returns:
        The file_id string for use with `chat_completions()`.

    Raises:
        ValueError: If the file format is not supported.

    Examples:
        Upload a video and run multiple analyses:

        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.video.localpath))
        >>> t.add_computed_column(
        ...     description=vlmrun.chat_completions(t.file_id, 'Describe this video')
        ... )
    """

    def _call_api() -> str:
        file_path = Path(path)
        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        client = _vlmrun_client()
        try:
            uploaded = client.files.upload(file=file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to upload file '{file_path.name}': {e}") from e
        return uploaded.id

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def chat_completions(
    file_id: str,
    prompt: str,
    *,
    model: str = 'vlmrun-orion-1:auto',
    response_format: Optional[dict] = None,
) -> dict:
    """
    Runs chat completions on an uploaded file using VLM Run's Orion API.

    Takes a file_id (from `upload_file()`) and a prompt, and returns the full response
    dictionary. Supports images, documents, and videos.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        file_id: The file_id returned from `upload_file()`.
        prompt: The text prompt describing what to analyze or extract.
        model: The model to use. Options: `vlmrun-orion-1:fast`, `vlmrun-orion-1:auto`,
            `vlmrun-orion-1:pro`. Defaults to `vlmrun-orion-1:auto`.
        response_format: Optional response format specification for structured output.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Analyze an uploaded video:

        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.video.localpath))
        >>> t.add_computed_column(
        ...     response=vlmrun.chat_completions(t.file_id, 'Describe this video')
        ... )
        >>> t.add_computed_column(text=t.response['choices'][0]['message']['content'])

        Run multiple analyses on the same file:

        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.image.localpath))
        >>> t.add_computed_column(
        ...     objects=vlmrun.chat_completions(t.file_id, 'List all objects')
        ... )
        >>> t.add_computed_column(
        ...     colors=vlmrun.chat_completions(t.file_id, 'What colors are present?')
        ... )
    """

    def _call_api() -> dict:
        client = _vlmrun_client()

        kwargs = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'input_file', 'file_id': file_id},
                    ],
                }
            ],
            'temperature': 0,
        }

        if response_format is not None:
            kwargs['response_format'] = response_format

        try:
            response = client.agent.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Chat completion failed for file_id '{file_id}': {e}") from e
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
        Redact a document and download the result:

        >>> t.add_computed_column(file_id=vlmrun.upload_file(t.doc.localpath))
        >>> t.add_computed_column(
        ...     redacted_url=vlmrun.chat_completions(
        ...         t.file_id,
        ...         'Redact all PII. Return the artifact URL.'
        ...     )
        ... )
        >>> t.add_computed_column(
        ...     redacted_path=vlmrun.get_artifact(t.redacted_url)
        ... )
    """
    import urllib.request
    from urllib.parse import urlparse

    from pixeltable.utils.local_store import TempStore

    def _call_api() -> str:
        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve artifact '{artifact_id}': {e}") from e

    result = await asyncio.to_thread(_call_api)
    return result


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
