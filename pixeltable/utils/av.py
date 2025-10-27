from typing import Any

import av
import av.stream

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

    def __init__(self, path: str, fps: float | None = None):
        self.path = path
        self.fps = fps if fps is not None and fps > 0.0 else None
        self.container = None
        self.video_framerate = None
        self.video_time_base = None
        self.video_start_time = None
        self.video_frame_count = None
        self.frames_to_extract = None
        self.next_pos = 0

    def __enter__(self):
        import math
        import pandas as pd
        from fractions import Fraction

        # Open the video file
        self.container = av.open(self.path)
        video_stream = self.container.streams.video[0]

        # Extract video metadata
        self.video_framerate = video_stream.average_rate
        self.video_time_base = video_stream.time_base
        self.video_start_time = video_stream.start_time or 0

        # Determine the number of frames in the video
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

        self.next_pos = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up
        if self.container:
            self.container.close()
        return False

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        import PIL.Image

        # Determine the frame index in the video corresponding to the iterator index `next_pos`
        if self.frames_to_extract is None:
            next_video_idx = self.next_pos  # extracting all frames
        elif self.next_pos >= len(self.frames_to_extract):
            raise StopIteration
        else:
            next_video_idx = self.frames_to_extract[self.next_pos]

        # Step through the video until we find the frame we're looking for
        while True:
            try:
                frame = next(self.container.decode(video=0))
            except StopIteration:
                raise
            except EOFError:
                raise StopIteration from None

            # Compute the index of the current frame based on pts
            pts = frame.pts - self.video_start_time
            video_idx = round(pts * self.video_time_base * self.video_framerate)
            assert isinstance(video_idx, int)

            if video_idx < next_video_idx:
                # Haven't reached the desired frame yet
                continue

            # Sanity check that we're at the right frame
            if video_idx != next_video_idx:
                raise ValueError(f'Frame {next_video_idx} is missing from the video (video file may be corrupt)')

            # Convert frame to PIL Image
            img = frame.to_image()
            assert isinstance(img, PIL.Image.Image)

            # Build result dict with frame and all frame attributes
            result = {
                'frame': img,
                'frame_attrs': {
                    'index': video_idx,
                    'pts': frame.pts,
                    'dts': frame.dts,
                    'time': frame.time,
                    'is_corrupt': frame.is_corrupt,
                    'key_frame': frame.key_frame,
                    'pict_type': frame.pict_type,
                    'interlaced_frame': frame.interlaced_frame,
                }
            }

            self.next_pos += 1
            return result
