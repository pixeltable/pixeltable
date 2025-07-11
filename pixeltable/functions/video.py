"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `VideoType`.
"""

import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import av
import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names

_format_defaults = {  # format -> (codec, ext)
    'wav': ('pcm_s16le', 'wav'),
    'mp3': ('libmp3lame', 'mp3'),
    'flac': ('flac', 'flac'),
    # 'mp4': ('aac', 'm4a'),
}

# for mp4:
# - extract_audio() fails with
#   "Application provided invalid, non monotonically increasing dts to muxer in stream 0: 1146 >= 290"
# - chatgpt suggests this can be fixed in the following manner
#     for packet in container.demux(audio_stream):
#         packet.pts = None  # Reset the PTS and DTS to allow FFmpeg to set them automatically
#         packet.dts = None
#         for frame in packet.decode():
#             frame.pts = None
#             for packet in output_stream.encode(frame):
#                 output_container.mux(packet)
#
#     # Flush remaining packets
#     for packet in output_stream.encode():
#         output_container.mux(packet)


@pxt.uda(requires_order_by=True)
class make_video(pxt.Aggregator):
    """
    Aggregator that creates a video from a sequence of images.
    """

    container: Optional[av.container.OutputContainer]
    stream: Optional[av.video.stream.VideoStream]
    fps: int

    def __init__(self, fps: int = 25):
        """follows https://pyav.org/docs/develop/cookbook/numpy.html#generating-video"""
        self.container = None
        self.stream = None
        self.fps = fps

    def update(self, frame: PIL.Image.Image) -> None:
        if frame is None:
            return
        if self.container is None:
            (_, output_filename) = tempfile.mkstemp(suffix='.mp4', dir=str(env.Env.get().tmp_dir))
            self.out_file = Path(output_filename)
            self.container = av.open(str(self.out_file), mode='w')
            self.stream = self.container.add_stream('h264', rate=self.fps)
            self.stream.pix_fmt = 'yuv420p'
            self.stream.width = frame.width
            self.stream.height = frame.height

        av_frame = av.VideoFrame.from_ndarray(np.array(frame.convert('RGB')), format='rgb24')
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)

    def value(self) -> pxt.Video:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        return str(self.out_file)


@pxt.udf(is_method=True)
def extract_audio(
    video_path: pxt.Video, stream_idx: int = 0, format: str = 'wav', codec: Optional[str] = None
) -> pxt.Audio:
    """
    Extract an audio stream from a video.

    Args:
        stream_idx: Index of the audio stream to extract.
        format: The target audio format. (`'wav'`, `'mp3'`, `'flac'`).
        codec: The codec to use for the audio stream. If not provided, a default codec will be used.

    Returns:
        The extracted audio.

    Examples:
        Add a computed column to a table `tbl` that extracts audio from an existing column `video_col`:

        >>> tbl.add_computed_column(
        ...     extracted_audio=tbl.video_col.extract_audio(format='flac')
        ... )
    """
    if format not in _format_defaults:
        raise ValueError(f'extract_audio(): unsupported audio format: {format}')
    default_codec, ext = _format_defaults[format]

    with av.open(video_path) as container:
        if len(container.streams.audio) <= stream_idx:
            return None
        audio_stream = container.streams.audio[stream_idx]
        # create this in our tmp directory, so it'll get cleaned up if it's being generated as part of a query
        output_filename = str(env.Env.get().tmp_dir / f'{uuid.uuid4()}.{ext}')

        with av.open(output_filename, 'w', format=format) as output_container:
            output_stream = output_container.add_stream(codec or default_codec)
            assert isinstance(output_stream, av.audio.stream.AudioStream)
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    output_container.mux(output_stream.encode(frame))  # type: ignore[arg-type]

        return output_filename


@pxt.udf(is_method=True)
def get_metadata(video: pxt.Video) -> dict:
    """
    Gets various metadata associated with a video file and returns it as a dictionary.

    Args:
        video: The video to get metadata for.

    Returns:
        A `dict` such as the following:

            ```json
            {
                'bit_exact': False,
                'bit_rate': 967260,
                'size': 2234371,
                'metadata': {
                    'encoder': 'Lavf60.16.100',
                    'major_brand': 'isom',
                    'minor_version': '512',
                    'compatible_brands': 'isomiso2avc1mp41',
                },
                'streams': [
                    {
                        'type': 'video',
                        'width': 640,
                        'height': 360,
                        'frames': 462,
                        'time_base': 1.0 / 12800,
                        'duration': 236544,
                        'duration_seconds': 236544.0 / 12800,
                        'average_rate': 25.0,
                        'base_rate': 25.0,
                        'guessed_rate': 25.0,
                        'metadata': {
                            'language': 'und',
                            'handler_name': 'L-SMASH Video Handler',
                            'vendor_id': '[0][0][0][0]',
                            'encoder': 'Lavc60.31.102 libx264',
                        },
                        'codec_context': {'name': 'h264', 'codec_tag': 'avc1', 'profile': 'High', 'pix_fmt': 'yuv420p'},
                    }
                ],
            }
            ```

    Examples:
        Extract metadata for files in the `video_col` column of the table `tbl`:

        >>> tbl.select(tbl.video_col.get_metadata()).collect()
    """
    return _get_metadata(video)


def _get_metadata(path: str) -> dict:
    with av.open(path) as container:
        assert isinstance(container, av.container.InputContainer)
        streams_info = [__get_stream_metadata(stream) for stream in container.streams]
        result = {
            'bit_exact': getattr(container, 'bit_exact', False),
            'bit_rate': container.bit_rate,
            'size': container.size,
            'metadata': container.metadata,
            'streams': streams_info,
        }
    return result


def __get_stream_metadata(stream: av.stream.Stream) -> dict:
    if stream.type not in ('audio', 'video'):
        return {'type': stream.type}  # Currently unsupported

    codec_context = stream.codec_context
    codec_context_md: dict[str, Any] = {
        'name': codec_context.name,
        'codec_tag': codec_context.codec_tag.encode('unicode-escape').decode('utf-8'),
        'profile': codec_context.profile,
    }
    metadata = {
        'type': stream.type,
        'duration': stream.duration,
        'time_base': float(stream.time_base) if stream.time_base is not None else None,
        'duration_seconds': float(stream.duration * stream.time_base)
        if stream.duration is not None and stream.time_base is not None
        else None,
        'frames': stream.frames,
        'metadata': stream.metadata,
        'codec_context': codec_context_md,
    }

    if stream.type == 'audio':
        # Additional metadata for audio
        channels = getattr(stream.codec_context, 'channels', None)
        codec_context_md['channels'] = int(channels) if channels is not None else None
    else:
        assert stream.type == 'video'
        assert isinstance(stream, av.video.stream.VideoStream)
        # Additional metadata for video
        codec_context_md['pix_fmt'] = getattr(stream.codec_context, 'pix_fmt', None)
        metadata.update(
            **{
                'width': stream.width,
                'height': stream.height,
                'frames': stream.frames,
                'average_rate': float(stream.average_rate) if stream.average_rate is not None else None,
                'base_rate': float(stream.base_rate) if stream.base_rate is not None else None,
                'guessed_rate': float(stream.guessed_rate) if stream.guessed_rate is not None else None,
            }
        )

    return metadata


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
