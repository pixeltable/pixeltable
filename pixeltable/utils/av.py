from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import PIL.Image
import av
import av.stream
import pandas as pd

from pixeltable.env import Env

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


def ffmpeg_clip_cmd(
    input_path: str,
    output_path: str,
    start_time: float,
    duration: float | None = None,
    fast: bool = True,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> list[str]:
    cmd = ['ffmpeg']
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

    if duration is not None:
        cmd.extend(['-t', str(duration)])
    cmd.extend(['-loglevel', 'error', output_path])
    return cmd


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
    cmd.extend(['-f', 'segment'])

    # -force_key_frames needs to precede -f segment
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
    fps: float | None
    container: av.container.input.InputContainer | None
    video_framerate: Fraction | None
    video_time_base: Fraction | None
    video_start_time: int | None
    video_frame_count: int | None
    frames_to_extract: list[int] | None

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
        self.fps = fps
        self.container = None
        self.video_framerate = None
        self.video_time_base = None
        self.video_start_time = None
        self.video_frame_count = None
        self.frames_to_extract = None

    def __enter__(self) -> VideoFrames:
        self.container = av.open(self.path)
        video_stream = self.container.streams.video[0]
        self.video_framerate = video_stream.average_rate
        self.video_time_base = video_stream.time_base
        self.video_start_time = video_stream.start_time or 0

        self.video_frame_count = video_stream.frames
        if self.video_frame_count == 0:
            # The video codec does not provide a frame count in the standard `frames` field
            metadata = video_stream.metadata
            if 'NUMBER_OF_FRAMES' in metadata:
                self.video_frame_count = int(metadata['NUMBER_OF_FRAMES'])
            elif 'DURATION' in metadata:
                # Calculate the frame count from the stream duration
                duration = metadata['DURATION']
                assert isinstance(duration, str)
                seconds = pd.to_timedelta(duration).total_seconds()
                self.video_frame_count = round(seconds * self.video_framerate)
            else:
                raise ValueError(f'Video {self.path}: failed to get number of frames')

        # Calculate which frames to extract based on fps
        if self.fps is None:
            # Extract all frames
            self.frames_to_extract = None
        elif self.fps > float(self.video_framerate):
            raise ValueError(
                f'Video {self.path}: requested fps ({self.fps}) exceeds video framerate ({float(self.video_framerate)})'
            )
        else:
            # Extract frames at the specified frequency
            freq = self.fps / float(self.video_framerate)
            n = math.ceil(self.video_frame_count * freq)  # number of frames to extract
            self.frames_to_extract = [round(i / freq) for i in range(n)]

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        # Clean up
        if self.container:
            self.container.close()
        return False

    def __iter__(self) -> Iterator[Item]:
        import PIL.Image

        next_pos = next_video_idx = 0
        while True:
            # Determine the frame index in the video corresponding to next_pos
            if self.frames_to_extract is None:
                next_video_idx = next_pos  # extracting all frames
            elif next_pos >= len(self.frames_to_extract):
                return
            else:
                next_video_idx = self.frames_to_extract[next_pos]

            # decode frames until we hit next_video_idx
            while True:
                try:
                    frame = next(self.container.decode(video=0))
                except StopIteration:
                    return
                except EOFError:
                    return

                pts = frame.pts - self.video_start_time
                video_idx = round(pts * self.video_time_base * self.video_framerate)
                if video_idx < next_video_idx:
                    # Haven't reached the desired frame yet
                    continue
                # Sanity check that we're at the right frame
                if video_idx != next_video_idx:
                    raise ValueError(f'Frame {next_video_idx} is missing from the video (video file may be corrupt)')

                img = frame.to_image()
                assert isinstance(img, PIL.Image.Image)

                # Build result dict with frame and all frame attributes
                result = VideoFrames.Item(
                    frame_idx=next_pos,
                    pts=frame.pts,
                    dts=frame.dts,
                    time=frame.time,
                    is_corrupt=frame.is_corrupt,
                    key_frame=frame.key_frame,
                    pict_type=frame.pict_type,
                    interlaced_frame=frame.interlaced_frame,
                    frame=img,
                )

                next_pos += 1
                yield result
                break
