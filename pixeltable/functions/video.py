import uuid
from typing import Optional

import av

import pixeltable.env as env
import pixeltable.func as func
import pixeltable.type_system as ts

_format_defaults = {  # format -> (codec, ext)
    'wav': ('pcm_s16le', 'wav'),
    'mp3': ('libmp3lame', 'mp3'),
    'flac': ('flac', 'flac'),
    #'mp4': ('aac', 'm4a'),
}

# for mp4:
# - extract_audio() fails with "Application provided invalid, non monotonically increasing dts to muxer in stream 0: 1146 >= 290"
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


_extract_audio_param_types = [
    ts.VideoType(nullable=False),
    ts.IntType(nullable=False),
    ts.StringType(nullable=False),
    ts.StringType(nullable=True),
]


@func.udf(return_type=ts.AudioType(nullable=True), param_types=_extract_audio_param_types)
def extract_audio(
    video_path: str, stream_idx: int = 0, format: str = 'wav', codec: Optional[str] = None
) -> Optional[str]:
    """Extract an audio stream from a video file, save it as a media file and return its path"""
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
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    output_container.mux(output_stream.encode(frame))

        return output_filename


@func.udf(return_type=ts.JsonType(nullable=False), param_types=[ts.VideoType(nullable=False)])
def get_metadata(video: str) -> dict:
    """Gets various metadata associated with a video file.

    Args:
        video (str): Path to the video file.

    Returns:
        A dictionary containing the associated metadata.
    """
    with av.open(video) as container:
        assert isinstance(container, av.container.InputContainer)
        video_streams_info = [
            {
                'duration': stream.duration,
                'frames': stream.frames,
                'language': stream.language,
                'average_rate': float(stream.average_rate) if stream.average_rate is not None else None,
                'base_rate': float(stream.base_rate) if stream.base_rate is not None else None,
                'guessed_rate': float(stream.guessed_rate) if stream.guessed_rate is not None else None,
                'pix_fmt': getattr(stream.codec_context, 'pix_fmt', None),
                'width': stream.width,
                'height': stream.height,
            }
            for stream in container.streams
            if isinstance(stream, av.video.stream.VideoStream)
        ]
        result = {
            'bit_exact': container.bit_exact,
            'bit_rate': container.bit_rate,
            'size': container.size,
            'metadata': container.metadata,
            'streams': video_streams_info,  # TODO: Audio streams?
        }
    return result
