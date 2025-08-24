from typing import Any

import av
import av.stream


def get_metadata(path: str) -> dict:
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


def get_video_duration(path: str) -> float | None:
    """Return video duration in seconds."""
    with av.open(path) as container:
        video_stream = container.streams.video[0]
        if video_stream is None:
            return None
        if video_stream.duration is not None:
            return float(video_stream.duration * video_stream.time_base)

        # if duration is not in the header, look for it in the last packet
        last_pts: int | None = None
        for packet in container.demux(video_stream):
            if packet.pts is not None:
                last_pts = packet.pts
        if last_pts is not None:
            return float(last_pts * video_stream.time_base)

        return None


def has_audio_stream(path: str) -> bool:
    """Check if video has audio stream using PyAV."""
    md = get_metadata(path)
    return any(stream['type'] == 'audio' for stream in md['streams'])


def ffmpeg_clip_cmd(input_path: str, output_path: str, start_time: float, duration: float | None = None) -> list[str]:
    # the order of arguments is critical: -ss <start> -t <duration> -i <input>
    cmd = ['ffmpeg', '-ss', str(start_time)]
    if duration is not None:
        cmd.extend(['-t', str(duration)])
    cmd.extend(
        [
            '-i',  # Input file
            input_path,
            '-y',  # Overwrite output file
            '-loglevel',
            'error',  # Only show errors
            '-c',
            'copy',  # Stream copy (no re-encoding)
            '-map',
            '0',  # Copy all streams from input
            output_path,
        ]
    )
    return cmd
