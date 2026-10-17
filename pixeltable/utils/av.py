import logging
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, NoReturn, Self

import av
import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env

_logger = logging.getLogger(__name__)

# format -> (codec, extension)
AUDIO_FORMATS: dict[str, tuple[str, str]] = {
    'wav': ('pcm_s16le', 'wav'),
    'mp3': ('libmp3lame', 'mp3'),
    'flac': ('flac', 'flac'),
    'mp4': ('aac', 'm4a'),
}


def _format_name_in(format_name: str, names: frozenset[str]) -> bool:
    return any(part.strip().lower() in names for part in format_name.split(','))


def _parse_clock_duration(value: str) -> float | None:
    """Parse an `HH:MM:SS[.fraction]` duration into seconds."""
    parts = value.strip().split(':')
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    if hours < 0 or minutes < 0 or seconds < 0:
        return None
    return hours * 3600 + minutes * 60 + seconds


# These formats store an absolute end instead of a length. NUT's container duration is the
# last presentation timestamp, not a length measured from the start.
_ABSOLUTE_END_FORMATS = frozenset({'asf', 'matroska', 'webm', 'nut'})


def _presentation_end(stored_s: float, start_s: float, *, absolute: bool) -> float | None:
    if stored_s <= 0:
        return None
    return stored_s if absolute else start_s + stored_s


def _sole_video_stream(container: av.container.input.InputContainer) -> bool:
    streams = container.streams
    return (
        len(streams.video) == 1 and len(streams.audio) == 0 and len(streams.subtitles) == 0 and len(streams.data) == 0
    )


def video_time_range(container: av.container.input.InputContainer) -> tuple[float, float] | None:
    """Return the first video stream's presentation start and end, in seconds."""
    if len(container.streams.video) == 0:
        return None
    stream = container.streams.video[0]
    if stream.time_base is None:
        return None
    start = float((stream.start_time or 0) * stream.time_base)
    format_name = container.format.name or ''
    absolute = _format_name_in(format_name, _ABSOLUTE_END_FORMATS)

    if stream.duration is not None:
        # AVStream.duration is an absolute end for ASF and a length for every other format.
        # Zero is an empty stream. It is not an unknown duration.
        if stream.duration == 0:
            return start, start
        stream_absolute = _format_name_in(format_name, frozenset({'asf'}))
        end = _presentation_end(float(stream.duration * stream.time_base), start, absolute=stream_absolute)
        if end is not None:
            return start, end

    tag = stream.metadata.get('DURATION') if stream.metadata else None
    if isinstance(tag, str):
        parsed = _parse_clock_duration(tag)
        if parsed is not None:
            end = _presentation_end(parsed, start, absolute=absolute)
            if end is not None:
                return start, end

    if container.duration is not None and _sole_video_stream(container):
        # Zero is an empty container, matching a zero stream duration above.
        if container.duration <= 0:
            return start, start
        end = _presentation_end(container.duration / 1_000_000, start, absolute=absolute)
        if end is not None:
            return start, end
    return None


def get_video_duration(path: str) -> float | None:
    """Return video duration in seconds."""
    with av.open(path) as container:
        assert isinstance(container, av.container.input.InputContainer)
        time_range = video_time_range(container)
        if time_range is not None:
            start, end = time_range
            return end - start if end >= start else None

        if len(container.streams.video) == 0:
            return None
        video_stream = container.streams.video[0]
        if video_stream.time_base is None:
            return None
        start = float((video_stream.start_time or 0) * video_stream.time_base)

        # Fall back to scanning packets. They are demuxed in decode order, so with B-frames
        # the last packet can present before an earlier one. The end is that latest PTS plus
        # the packet's duration.
        max_pts: int | None = None  # max observed packet.pts
        max_pts_duration: int | None = None  # duration of that packet
        for packet in container.demux(video_stream):
            if packet.pts is not None and (max_pts is None or packet.pts > max_pts):
                max_pts = packet.pts
                max_pts_duration = packet.duration
        if max_pts is None:
            return None
        end = float((max_pts + (max_pts_duration or 0)) * video_stream.time_base)
        # Some demuxers, such as MPEG, emit trailing packets past the container's end.
        if container.duration is not None:
            stored = container.duration / 1_000_000
            absolute = _format_name_in(container.format.name or '', _ABSOLUTE_END_FORMATS)
            end = min(end, stored if absolute else start + stored)
        return end - start if end >= start else None


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
    with av.open(path) as container:
        assert isinstance(container, av.container.InputContainer)
        return any(stream.type == 'audio' for stream in container.streams)


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
    raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, error_msg) from e


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
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_DATA_FORMAT,
                f'ffmpeg failed to create output file for commandline: {" ".join(cmd)}\n{stderr_output}',
            )
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
    # Presentation end recorded by the container, in seconds.
    _video_end: float | None
    # Presentation time of the latest decoded frame, in seconds.
    last_frame_time: float | None
    # Duration of that frame, in seconds. None when PyAV omits it or reports zero.
    last_frame_duration: float | None

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
        self._video_end = None
        self.last_frame_time = None
        self.last_frame_duration = None

    def __enter__(self) -> Self:
        self.container = av.open(self.path)
        assert isinstance(self.container, av.container.input.InputContainer)
        stream = self.container.streams.video[0]
        self.video_framerate = stream.average_rate
        time_range = video_time_range(self.container)
        self._video_end = None if time_range is None else time_range[1]
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        # Clean up
        if self.container:
            self.container.close()

    def end_time(self) -> float:
        """The time at which the video ends, in seconds. Valid after iteration is exhausted."""
        assert self.last_frame_time is not None
        if self._video_end is not None and self._video_end > self.last_frame_time:
            return self._video_end
        # The stored end is missing or not past the last frame. That frame's own duration is the
        # end; one average frame interval is only a guess.
        if self.last_frame_duration is not None:
            return self.last_frame_time + self.last_frame_duration
        assert self.video_framerate is not None and self.video_framerate != 0
        return self.last_frame_time + float(1 / self.video_framerate)

    def __iter__(self) -> Iterator[Item]:
        assert self.container is not None
        self.last_frame_time = None
        self.last_frame_duration = None
        latest_pts: int | None = None
        num_returned = 0
        frame_idx = -1
        try:
            for frame in self.container.decode(video=0):
                frame_idx += 1
                # The latest presentation time wins, including a frame fps sampling does not yield.
                # A missing duration must not keep an earlier frame's end.
                if frame.pts is not None and frame.time is not None and (latest_pts is None or frame.pts >= latest_pts):
                    latest_pts = frame.pts
                    self.last_frame_time = float(frame.time)
                    # Frame.duration was added in PyAV 14.4. av>=14.1 has no such attribute,
                    # and a zero duration is not an end.
                    frame_duration = getattr(frame, 'duration', None)
                    time_base = frame.time_base
                    if isinstance(frame_duration, int) and frame_duration > 0 and time_base is not None:
                        self.last_frame_duration = float(frame_duration * time_base)
                    else:
                        self.last_frame_duration = None
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
        except EOFError:
            return
