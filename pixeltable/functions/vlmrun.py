"""
Pixeltable UDFs that wrap the VLM Run Orion API for multimodal chat completions.

In order to use them, you must first `pip install vlmrun` and configure your VLM Run API key,
as described in the [VLM Run documentation](https://docs.vlm.run/).
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import PIL.Image

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    from vlmrun.client import VLMRun


@env.register_client('vlmrun')
def _(api_key: str) -> 'VLMRun':
    from vlmrun.client import VLMRun

    return VLMRun(api_key=api_key, base_url='https://agent.vlm.run/v1')


def _vlmrun_client() -> 'VLMRun':
    return env.Env.get().get_client('vlmrun')


@pxt.udf(resource_pool='request-rate:vlmrun')
async def chat_image(
    image: PIL.Image.Image,
    prompt: str,
    *,
    model: str = 'vlmrun-orion-1:auto',
) -> str:
    """
    Analyzes images using VLM Run's chat completions API.

    This UDF accepts a PIL Image, uploads it to VLM Run, sends a chat completion
    request with the specified prompt, and returns the text response.

    For running multiple analyses on the same image, consider using `upload()` +
    `chat_with_file()` to avoid re-uploading.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        image: The image to analyze.
        prompt: The text prompt describing what to analyze or extract from the image.
        model: The name of the model to use. Options: `vlmrun-orion-1:fast`, `vlmrun-orion-1:auto`,
            `vlmrun-orion-1:pro`. Defaults to `vlmrun-orion-1:auto`.

    Returns:
        The text response from the model.

    Examples:
        Analyze images stored in a Pixeltable column:

        >>> tbl.add_computed_column(
        ...     image_description=vlmrun.chat_image(tbl.image, "Describe this image in detail")
        ... )

        Extract specific information from receipts:

        >>> tbl.add_computed_column(
        ...     total=vlmrun.chat_image(tbl.receipt, "What is the total amount on this receipt?")
        ... )
    """

    def _call_api() -> str:
        import os

        client = _vlmrun_client()

        # Save image to temp file for upload
        temp_path = TempStore.create_path(extension='.png')
        try:
            image.save(temp_path, format='PNG')

            # Upload file to VLM Run object store
            uploaded = client.files.upload(file=Path(temp_path))

            # Chat completion using file_id
            response = client.agent.completions.create(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {'type': 'input_file', 'file_id': uploaded.id},
                        ],
                    }
                ],
                temperature=0,
            )
            return response.choices[0].message.content
        finally:
            # Clean up temp file after upload
            if os.path.exists(temp_path):
                os.remove(temp_path)

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def chat_document(
    document: str,
    prompt: str,
    *,
    model: str = 'vlmrun-orion-1:auto',
) -> str:
    """
    Analyzes PDF documents using VLM Run's chat completions API.

    This UDF accepts a document file path, uploads it to VLM Run, sends a chat completion
    request with the specified prompt, and returns the text response.

    For running multiple analyses on the same document, consider using `upload()` +
    `chat_with_file()` to avoid re-uploading.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        document: The file path to the PDF document to analyze.
        prompt: The text prompt describing what to analyze or extract from the document.
        model: The name of the model to use. Options: `vlmrun-orion-1:fast`, `vlmrun-orion-1:auto`,
            `vlmrun-orion-1:pro`. Defaults to `vlmrun-orion-1:auto`.

    Returns:
        The text response from the model.

    Examples:
        Summarize PDF documents:

        >>> tbl.add_computed_column(
        ...     doc_summary=vlmrun.chat_document(tbl.document.localpath, "Summarize the key points")
        ... )

        Extract specific information from contracts:

        >>> tbl.add_computed_column(
        ...     parties=vlmrun.chat_document(tbl.contract.localpath, "Who are the parties in this contract?")
        ... )
    """

    def _call_api() -> str:
        client = _vlmrun_client()

        # Upload file to VLM Run object store
        uploaded = client.files.upload(file=Path(document))

        # Chat completion using file_id
        response = client.agent.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'input_file', 'file_id': uploaded.id},
                    ],
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def upload(media: str) -> str:
    """
    Uploads a file to VLM Run and returns the file_id for later use.

    This UDF supports images, videos, and PDF documents. It is designed for a two-stage
    workflow where files are uploaded once and then analyzed multiple times using the
    stored file_id. This avoids redundant uploads when running multiple analyses.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        media: The file path to upload (supports images, videos, and PDFs).

    Returns:
        The file_id string that can be used with `chat_with_file()`.

    Examples:
        Upload video and run multiple analyses:

        >>> tbl.add_computed_column(file_id=vlmrun.upload(tbl.video.localpath))
        >>> tbl.add_computed_column(
        ...     description=vlmrun.chat_with_file(tbl.file_id, "Describe this video")
        ... )

        Upload image for multiple analyses:

        >>> tbl.add_computed_column(file_id=vlmrun.upload(tbl.image.localpath))
        >>> tbl.add_computed_column(
        ...     objects=vlmrun.chat_with_file(tbl.file_id, "What objects are visible?")
        ... )

        Upload PDF for multiple queries:

        >>> tbl.add_computed_column(file_id=vlmrun.upload(tbl.document.localpath))
        >>> tbl.add_computed_column(
        ...     summary=vlmrun.chat_with_file(tbl.file_id, "Summarize this document")
        ... )
    """

    def _call_api() -> str:
        client = _vlmrun_client()
        uploaded = client.files.upload(file=Path(media))
        return uploaded.id

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def chat_with_file(
    file_id: str,
    prompt: str,
    *,
    model: str = 'vlmrun-orion-1:auto',
) -> str:
    """
    Analyzes an already-uploaded file using VLM Run's chat completions API.

    This UDF uses a previously obtained file_id (from `upload()`) to run chat
    completions without re-uploading the file. Works with any file type that was
    uploaded: images, videos, or PDFs. This enables efficient workflows where a
    single upload supports multiple analysis queries.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        file_id: The file_id returned from `upload()`.
        prompt: The text prompt describing what to analyze or extract.
        model: The name of the model to use. Options: `vlmrun-orion-1:fast`, `vlmrun-orion-1:auto`,
            `vlmrun-orion-1:pro`. Defaults to `vlmrun-orion-1:auto`.

    Returns:
        The text response from the model.

    Examples:
        Analyze an uploaded video:

        >>> tbl.add_computed_column(file_id=vlmrun.upload(tbl.video.localpath))
        >>> tbl.add_computed_column(
        ...     description=vlmrun.chat_with_file(tbl.file_id, "What happens in this video?")
        ... )

        Analyze an uploaded image:

        >>> tbl.add_computed_column(file_id=vlmrun.upload(tbl.image.localpath))
        >>> tbl.add_computed_column(
        ...     description=vlmrun.chat_with_file(tbl.file_id, "Describe this image")
        ... )

        Analyze an uploaded PDF:

        >>> tbl.add_computed_column(file_id=vlmrun.upload(tbl.document.localpath))
        >>> tbl.add_computed_column(
        ...     summary=vlmrun.chat_with_file(tbl.file_id, "Summarize this document")
        ... )
    """

    def _call_api() -> str:
        client = _vlmrun_client()
        response = client.agent.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'input_file', 'file_id': file_id},
                    ],
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    result = await asyncio.to_thread(_call_api)
    return result


@pxt.udf(resource_pool='request-rate:vlmrun')
async def redact(
    document: str,
    *,
    instructions: str = 'Redact all PII from this document.',
    model: str = 'vlmrun-orion-1:auto',
) -> str:
    """
    Redacts PII from documents using VLM Run's chat completions API.

    This UDF accepts a document or image file path, uploads it to VLM Run, sends a chat
    completion request with redaction instructions, downloads the redacted file, and
    returns the local file path.

    Request throttling:
    Applies the rate limit set in the config (section `vlmrun`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install vlmrun`

    Args:
        document: The file path to the document or image to redact.
        instructions: Custom redaction instructions. Defaults to redacting common PII.
        model: The name of the model to use. Defaults to `vlmrun-orion-1:auto`.

    Returns:
        The local file path to the redacted document.

    Examples:
        Redact PII from documents:

        >>> tbl.add_computed_column(
        ...     redacted=vlmrun.redact(tbl.document.localpath)
        ... )

        Custom redaction instructions:

        >>> tbl.add_computed_column(
        ...     redacted=vlmrun.redact(
        ...         tbl.document.localpath,
        ...         instructions="Redact all patient names and medical record numbers"
        ...     )
        ... )
    """
    import urllib.request
    from urllib.parse import urlparse

    from pydantic import BaseModel, Field

    class RedactedUrlResponse(BaseModel):
        url: str = Field(..., description='The presigned URL to the redacted image')

    def _call_api() -> str:
        client = _vlmrun_client()

        # Upload file to VLM Run object store
        uploaded = client.files.upload(file=Path(document))

        # Build prompt with redaction instructions
        prompt = f'{instructions} Return the presigned URL to the redacted image.'

        # Chat completion with redaction prompt and structured response
        response = client.agent.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'input_file', 'file_id': uploaded.id},
                    ],
                }
            ],
            response_format={
                'type': 'json_schema',
                'schema': RedactedUrlResponse.model_json_schema(),
            },
            temperature=0,
        )

        # Parse response to get URL
        content = response.choices[0].message.content
        result = RedactedUrlResponse.model_validate_json(content)

        # Get file extension from URL path
        parsed = urlparse(result.url)
        ext = Path(parsed.path).suffix or '.png'

        # Download to temp file
        local_path = TempStore.create_path(extension=ext)
        urllib.request.urlretrieve(result.url, local_path)

        return str(local_path)

    result = await asyncio.to_thread(_call_api)
    return result


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
