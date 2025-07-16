"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `AudioType`.
"""

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf(is_method=True)
def get_metadata(audio: pxt.Audio) -> dict:
    """
    Gets various metadata associated with an audio file and returns it as a dictionary.

    Args:
        audio: The audio to get metadata for.

    Returns:
        A `dict` such as the following:

            ```json
            {
                'size': 2568827,
                'streams': [
                    {
                        'type': 'audio',
                        'frames': 0,
                        'duration': 2646000,
                        'metadata': {},
                        'time_base': 2.2675736961451248e-05,
                        'codec_context': {
                            'name': 'flac',
                            'profile': None,
                            'channels': 1,
                            'codec_tag': '\\x00\\x00\\x00\\x00',
                        },
                        'duration_seconds': 60.0,
                    }
                ],
                'bit_rate': 342510,
                'metadata': {'encoder': 'Lavf61.1.100'},
                'bit_exact': False,
            }
            ```

    Examples:
        Extract metadata for files in the `audio_col` column of the table `tbl`:

        >>> tbl.select(tbl.audio_col.get_metadata()).collect()
    """
    return pxt.functions.video._get_metadata(audio)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
