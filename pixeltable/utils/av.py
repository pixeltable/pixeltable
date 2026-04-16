from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, NoReturn

import av
import av.stream
import PIL.Image
from typing_extensions import Self

import pixeltable as pxt
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')

# format -> (codec, extension)
AUDIO_FORMATS: dict[str, tuple[str, str]] = {
    'wav': ('pcm_s16le', 'wav'),
    'mp3': ('libmp3lame', 'mp3'),
    'flac': ('flac', 'flac'),
    'mp4': ('aac', 'm4a'),
}


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

    # Compute duration_seconds from stream-level duration.
    # We intentionally don't fall back to container.duration here because it's ambiguous —
    # it may reflect a different stream's duration (e.g. audio vs video).
    duration_seconds: float | None = None
    if stream.duration is not None and stream.time_base is not None:
        duration_seconds = float(stream.duration * stream.time_base)

    metadata = {
        'type': stream.type,
        'duration': stream.duration,
        'time_base': float(stream.time_base) if stream.time_base is not None else None,
        'duration_seconds': duration_seconds,
        'frames': stream.frames,
        'metadata': stream.metadata,
        'codec_context': codec_context_md,
    }

    if stream.type == 'audio':
        # Additional metadata for audio
        assert isinstance(stream.codec_context, av.AudioCodecContext)
        channels = stream.codec_context.channels
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
                'average_rate': float(stream.average_rate) if stream.average_rate is not None else None,
                'base_rate': float(stream.base_rate) if stream.base_rate is not None else None,
                'guessed_rate': float(stream.guessed_rate) if stream.guessed_rate is not None else None,
            }
        )

    return metadata


def get_video_duration(path: str) -> float | None:
    """Return video duration in seconds."""
    with av.open(path) as container:
        if len(container.streams.video) == 0:
            return None
        video_stream = container.streams.video[0]

        # Prefer stream-level duration from the header
        if video_stream.duration is not None:
            return float(video_stream.duration * video_stream.time_base)

        # use container duration if we don't have audio streams (which might be longer than the video stream)
        if len(container.streams.audio) == 0 and container.duration is not None:
            return container.duration / 1_000_000

        # Fall back to scanning packets to find the latest presentation timestamp.
        # We track the maximum PTS rather than the last packet's PTS because B-frame reordering
        # (common in h264/h265) means packets are demuxed in decode order, not presentation order.
        # The last demuxed packet may be a B-frame that presents before the final I/P frame.
        max_pts: int | None = None  # max observed packet.pts
        max_pts_duration: int | None = None  # duration of that packet
        for packet in container.demux(video_stream):
            if packet.pts is not None and (max_pts is None or packet.pts > max_pts):
                max_pts = packet.pts
                max_pts_duration = packet.duration
        if max_pts is not None:
            end_pts = max_pts + (max_pts_duration or 0)
            result = float(end_pts * video_stream.time_base)
            # the video stream can't be longer than the container, but some demuxers
            # (e.g. MPEG) emit trailing packets past the real end
            if container.duration is not None:
                result = min(result, container.duration / 1_000_000)
            return result

        return None


def get_audio_duration(path: str) -> float | None:
    """
    Return audio duration in seconds, or None if there is no audio stream or the duration cannot be determined.
    """
    with av.open(path) as container:
        if len(container.streams.audio) == 0:
            return None
        audio_stream = container.streams.audio[0]

        # prefer stream-level duration from the header
        if audio_stream.duration is not None:
            return float(audio_stream.duration * audio_stream.time_base)

        # use container duration if we have no video streams that might be longer than the audio
        if len(container.streams.video) == 0 and container.duration is not None:
            return container.duration / 1_000_000

        # Fall back to decoding all audio frames and counting samples. This handles formats like
        # FLAC that don't always expose duration in the stream header.
        sample_rate = audio_stream.codec_context.sample_rate
        if sample_rate is None or sample_rate == 0:
            return None
        n_samples = 0
        for frame in container.decode(audio=0):
            n_samples += frame.samples
        return n_samples / sample_rate if n_samples > 0 else None


_MAX_VOLUME_PATTERN = re.compile(r'max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB')


def get_max_volume_db(path: str) -> float | None:
    """
    Measure the peak amplitude of an audio clip in dBFS.

    Returns the max volume as a non-positive number (0 dBFS = full scale), or None if the clip
    is silent or has no audio stream.
    """
    Env.get().require_binary('ffmpeg')
    # -vn/-sn/-dn: ignore non-audio streams; -f null -: discard the output, we only want stderr.
    # Explicit -loglevel info so volumedetect's report reaches us regardless of the build default.
    cmd = ['ffmpeg', '-i', path, '-vn', '-sn', '-dn', '-af', 'volumedetect', '-f', 'null', '-loglevel', 'info', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        handle_ffmpeg_error(e)
    # Stderr lines look like: "[Parsed_volumedetect_0 @ 0x...] max_volume: -6.0 dB"
    m = re.search(_MAX_VOLUME_PATTERN, result.stderr)
    return float(m.group(1)) if m is not None else None


def has_audio_stream(path: str) -> bool:
    """Check if video has audio stream using PyAV."""
    md = get_metadata(path)
    return any(stream['type'] == 'audio' for stream in md['streams'])


# bytes of memory per pixel of decoded frames, by pixel format
BYTES_PER_PIXEL = {
    # 8-bit 4:2:0: chroma planes are quarter size
    'yuv420p': 1.5,
    'yuvj420p': 1.5,
    # 8-bit 4:2:2: chroma planes are half size
    'yuv422p': 2.0,
    'yuvj422p': 2.0,
    # 8-bit 4:4:4: all planes full size
    'yuv444p': 3.0,
    'yuvj444p': 3.0,
    # packed 4:2:0 variants (Android/camera common)
    'nv12': 1.5,
    'nv21': 1.5,
    # 10-bit variants (HDR content)
    'yuv420p10le': 3.0,
    'yuv420p10be': 3.0,
    'yuv422p10le': 4.0,
    'yuv422p10be': 4.0,
    'yuv444p10le': 6.0,
    'yuv444p10be': 6.0,
    'p010le': 3.0,
    'p010be': 3.0,
    # 12-bit variants
    'yuv420p12le': 3.0,
    'yuv420p12be': 3.0,
    'yuv422p12le': 4.0,
    'yuv422p12be': 4.0,
    'yuv444p12le': 6.0,
    'yuv444p12be': 6.0,
    # RGB/RGBA
    'rgb24': 3.0,
    'bgr24': 3.0,
    'rgba': 4.0,
    'bgra': 4.0,
    'rgb48le': 6.0,
    'rgb48be': 6.0,
    'rgba64le': 8.0,
    'rgba64be': 8.0,
}

DEFAULT_BYTES_PER_PIXEL = 3.0  # a conservative fallback for unknown formats


def estimate_segment_duration(path: str, approx_decoded_bytes: int) -> float | None:
    """
    Return the length of a segment for which the combined in-memory size of all its decoded frames is roughly
    approx_decoded_bytes.

    Returns None f the frame rate or dimensions cannot be determined.
    """

    with av.open(path) as container:
        if len(container.streams.video) == 0:
            return None
        video_stream = container.streams.video[0]
        width = video_stream.width
        height = video_stream.height
        pix_fmt = video_stream.codec_context.pix_fmt

        if width <= 0 or height <= 0:
            return None

        if video_stream.average_rate is None or video_stream.average_rate == 0:
            return None
        fps = float(video_stream.average_rate)

    bpp = BYTES_PER_PIXEL.get(pix_fmt, DEFAULT_BYTES_PER_PIXEL)
    bytes_per_frame = width * height * bpp
    frames_per_segment = approx_decoded_bytes / bytes_per_frame
    return frames_per_segment / fps


def handle_ffmpeg_error(e: subprocess.CalledProcessError) -> NoReturn:
    error_msg = f'ffmpeg failed with return code {e.returncode}'
    if e.stderr is not None:
        error_msg += f':\n{e.stderr.strip()}'
    raise pxt.Error(error_msg) from e


def run_ffmpeg_cmdline(
    ffmpeg_args: list[str],
    output_path: str,
    encode_video: bool = False,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> str:
    """
    Create and run an ffmpeg commandline, verify the output file, and return its path.

    If encode_video==True, command is expected to encode video and requires video encoder args
    """
    cmd = ['ffmpeg', *ffmpeg_args]
    if encode_video:
        append_video_encoder(cmd, video_encoder, video_encoder_args)
    # loglevel=error: avoid excessive logging
    cmd += ['-loglevel', 'error', output_path]

    try:
        _logger.debug(f'running ffmpeg commandline: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_file = Path(output_path)
        if not output_file.exists() or output_file.stat().st_size == 0:
            stderr_output = result.stderr.strip() if result.stderr is not None else ''
            raise pxt.Error(f'ffmpeg failed to create output file for commandline: {" ".join(cmd)}\n{stderr_output}')
        return output_path
    except subprocess.CalledProcessError as e:
        handle_ffmpeg_error(e)


def append_video_encoder(
    cmd: list[str], video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> None:
    """Append video encoder-related args to ffmpeg cmdline."""
    if video_encoder is None:
        video_encoder = Env.get().default_video_encoder
    if video_encoder is not None:
        cmd.extend(['-c:v', video_encoder])
    if video_encoder_args is not None:
        for k, v in video_encoder_args.items():
            cmd.extend([f'-{k}', str(v)])


def ffmpeg_clip_args(
    input_path: str,
    start_time: float,
    duration: float | None = None,
    fast: bool = True,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> list[str]:
    """Construct args for an ffmpeg commandline that creates a clip."""
    cmd: list[str] = []
    if fast:
        # fast: -ss before -i
        cmd.extend(
            [
                '-ss',
                str(start_time),
                '-i',
                input_path,
                '-map',
                '0',  # Copy all streams from input
                '-c',
                'copy',  # Stream copy (no re-encoding)
            ]
        )
    else:
        if video_encoder is None:
            video_encoder = Env.get().default_video_encoder

        # accurate: -ss after -i
        cmd.extend(
            [
                '-i',
                input_path,
                '-ss',
                str(start_time),
                '-map',
                '0',  # Copy all streams from input
                '-c:a',
                'copy',  # audio copy
                '-c:s',
                'copy',  # subtitle copy
                '-c:v',
                video_encoder,  # re-encode video
            ]
        )
        if video_encoder_args is not None:
            for k, v in video_encoder_args.items():
                cmd.extend([f'-{k}', str(v)])

    # critical: duration is added *after* encoding flags
    if duration is not None:
        cmd.extend(['-t', str(duration)])
    return cmd


def ffmpeg_clip_cmd(
    input_path: str,
    output_path: str,
    start_time: float,
    duration: float | None = None,
    fast: bool = True,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> list[str]:
    return [
        'ffmpeg',
        *ffmpeg_clip_args(input_path, start_time, duration, fast, video_encoder, video_encoder_args),
        '-loglevel',
        'error',
        output_path,
    ]


def ffmpeg_segment_cmd(
    input_path: str,
    output_pattern: str,
    segment_duration: float | None = None,
    segment_times: list[float] | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> list[str]:
    """Commandline for frame-accurate segmentation"""
    assert (segment_duration is None) != (segment_times is None)
    if video_encoder is None:
        video_encoder = Env.get().default_video_encoder

    cmd = [
        'ffmpeg',
        '-i',
        input_path,
        '-map',
        '0',  # Copy all streams from input
        '-c:a',
        'copy',  # don't re-encode audio
        '-c:v',
        video_encoder,  # re-encode video
    ]
    if video_encoder_args is not None:
        for k, v in video_encoder_args.items():
            cmd.extend([f'-{k}', str(v)])

    # -force_key_frames must precede -f segment
    if segment_duration is not None:
        cmd.extend(
            [
                '-force_key_frames',
                f'expr:gte(t,n_forced*{segment_duration})',  # Force keyframe at each segment boundary
                '-f',
                'segment',
                '-segment_time',
                str(segment_duration),
            ]
        )
    else:
        assert segment_times is not None
        times_str = ','.join([str(t) for t in segment_times])
        cmd.extend(['-force_key_frames', times_str, '-f', 'segment', '-segment_times', times_str])

    cmd.extend(
        [
            '-break_non_keyframes',
            '1',  # Allow splitting at non-keyframe boundaries for accurate duration
            '-reset_timestamps',
            '1',  # Reset timestamps for each segment
            '-loglevel',
            'error',  # Only show errors
            output_pattern,
        ]
    )
    return cmd


class VideoFrames:
    """
    Context manager for iterating over video frames at a specified frame rate.

    Args:
        path: Path to the video file
        fps: Number of frames to extract per second. If None or 0.0, extracts all frames.
    """

    path: Path
    fps: float
    container: av.container.input.InputContainer | None
    video_framerate: Fraction | None
    video_time_base: Fraction | None
    video_start_time: int | None

    @dataclass
    class Item:
        frame_idx: int
        pts: int
        dts: int
        time: float
        is_corrupt: bool
        key_frame: bool
        pict_type: int
        interlaced_frame: bool
        frame: PIL.Image.Image

    def __init__(self, path: Path, fps: float | None = None) -> None:
        self.path = path
        self.fps = 0.0 if fps is None else fps
        self.container = None
        self.video_framerate = None
        self.video_time_base = None
        self.video_start_time = None

    def __enter__(self) -> Self:
        self.container = av.open(self.path)
        stream = self.container.streams.video[0]
        self.video_framerate = stream.average_rate
        self.video_time_base = stream.time_base
        self.video_start_time = stream.start_time or 0
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        # Clean up
        if self.container:
            self.container.close()

    def __iter__(self) -> Iterator[Item]:
        num_returned = 0
        frame_idx = -1
        while True:
            try:
                frame = next(self.container.decode(video=0))
            except (StopIteration, EOFError):
                return

            frame_idx += 1
            if self.fps == 0.0 or (num_returned <= frame.time * self.fps):
                img = frame.to_image()
                assert isinstance(img, PIL.Image.Image)
                yield VideoFrames.Item(
                    frame_idx=frame_idx,
                    pts=frame.pts,
                    dts=frame.dts,
                    time=frame.time,
                    is_corrupt=frame.is_corrupt,
                    key_frame=frame.key_frame,
                    pict_type=frame.pict_type,
                    interlaced_frame=frame.interlaced_frame,
                    frame=img,
                )
                num_returned += 1
