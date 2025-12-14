"""
Pixeltable UDF for converting media file URIs to presigned HTTP URLs.
"""

from pixeltable import exceptions as excs
from pixeltable.func.udf import udf
from pixeltable.utils.code import local_public_names
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, StorageTarget


@udf
def presigned_url(uri: str, expiration_seconds: int) -> str:
    """
    Convert a blob storage URI to a presigned HTTP URL for direct access.

    Generates a time-limited, publicly accessible URL from cloud storage URIs
    (S3, GCS, Azure, etc.) that can be used to serve media files over HTTP.

    Note:
        This function uses presigned URLs from storage providers. Provider-specific
        limitations apply:

        - Google Cloud Storage: maximum 7-day expiration
        - AWS S3: requires proper region configuration
        - Azure: subject to storage account access policies

    Args:
        uri: The media file URI (e.g., `s3://bucket/path`, `gs://bucket/path`, `azure://container/path`)
        expiration_seconds: How long the URL remains valid

    Returns:
        A presigned HTTP URL for accessing the file

    Raises:
        Error: If the URI is a local file:// path

    Examples:
        Generate a presigned URL for a video column with 1-hour expiration:

        >>> tbl.select(
        ...     original_url=tbl.video.fileurl,
        ...     presigned_url=pxtf.net.presigned_url(tbl.video.fileurl, 3600)
        ... ).collect()
    """
    if not uri:
        return uri

    # Parse the object storage address from the URI
    soa = ObjectPath.parse_object_storage_addr(uri, allow_obj_name=True)

    # HTTP/HTTPS URLs are already publicly accessible
    if soa.storage_target == StorageTarget.HTTP_STORE:
        return uri

    # For file:// URLs, we can't generate presigned URLs
    if soa.storage_target == StorageTarget.LOCAL_STORE:
        raise excs.Error(
            'Cannot generate presigned URL for local file:// URLs. '
            'Please use cloud storage (S3, GCS, Azure) for presigned URLs.'
        )

    store = ObjectOps.get_store(soa, allow_obj_name=True)
    return store.create_presigned_url(soa, expiration_seconds)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
