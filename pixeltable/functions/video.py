"""
Pixeltable UDFs for `VideoType`.
"""

import glob
import logging
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, NamedTuple, TypedDict

import av
import av.container
import numpy as np
import pandas as pd
import PIL.Image
from av.container import InputContainer

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    from scenedetect.detectors import SceneDetector  # type: ignore[import-untyped]

_logger = logging.getLogger('pixeltable')


@pxt.uda(requires_order_by=True)
class make_video(pxt.Aggregator):
    """
    Aggregate function that creates a video from a sequence of images, using the default video encoder and
    yuv420p pixel format.

    Args:
        fps: Frames per second for the output video.

    Returns:
        The video obtained by combining the input frames at the specified `fps`.

    Examples:
        Combine the images in the `img` column of the table `tbl` into a video:

        >>> tbl.select(make_video(tbl.img, fps=30)).collect()

        Combine a sequence of rotated images into a video:

        >>> tbl.select(make_video(tbl.img.rotate(45), fps=30)).collect()

        For a more extensive example, see the
        [Object Detection in Videos](https://docs.pixeltable.com/howto/cookbooks/video/object-detection-in-videos)
        cookbook.
    """

    # Based on: https://pyav.org/docs/develop/cookbook/numpy.html#generating-video

    # TODO: provide parameters for video_encoder and pix_fmt

    container: av.container.OutputContainer | None
    stream: av.VideoStream | None
    fps: int

    def __init__(self, fps: int = 25):
        self.container = None
        self.stream = None
        self.fps = fps

    def update(self, frame: PIL.Image.Image) -> None:
        if frame is None:
            return
        if self.container is None:
            self.out_file = TempStore.create_path(extension='.mp4')
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
    video_path: pxt.Video, stream_idx: int = 0, format: str = 'wav', codec: str | None = None
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
    if format not in av_utils.AUDIO_FORMATS:
        raise ValueError(f'extract_audio(): unsupported audio format: {format}')
    default_codec, ext = av_utils.AUDIO_FORMATS[format]

    with av.open(video_path) as container:
        if len(container.streams.audio) <= stream_idx:
            return None
        audio_stream = container.streams.audio[stream_idx]
        # create this in our tmp directory, so it'll get cleaned up if it's being generated as part of a query
        output_path = str(TempStore.create_path(extension=f'.{ext}'))

        with av.open(output_path, 'w', format=format) as output_container:
            output_stream = output_container.add_stream(codec or default_codec)
            assert isinstance(output_stream, av.audio.stream.AudioStream)
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    output_container.mux(output_stream.encode(frame))  # type: ignore[arg-type]

        return output_path


@pxt.udf(is_method=True)
def get_metadata(video: pxt.Video) -> dict:
    """
    Gets various metadata associated with a video file and returns it as a dictionary.

    Args:
        video: The video for which to get metadata.

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
    return av_utils.get_metadata(video)


@pxt.udf(is_method=True)
def get_duration(video: pxt.Video) -> float | None:
    """
    Get video duration in seconds.

    Args:
        video: The video for which to get the duration.

    Returns:
        The duration in seconds, or None if the duration cannot be determined.
    """
    return av_utils.get_video_duration(video)


@pxt.udf(is_method=True)
def extract_frame(video: pxt.Video, *, timestamp: float) -> PIL.Image.Image | None:
    """
    Extract a single frame from a video at a specific timestamp.

    Args:
        video: The video from which to extract the frame.
        timestamp: Extract frame at this timestamp (in seconds).

    Returns:
        The extracted frame as a PIL Image, or None if the timestamp is beyond the video duration.

    Examples:
        Extract the first frame from each video in the `video` column of the table `tbl`:

        >>> tbl.select(tbl.video.extract_frame(0.0)).collect()

        Extract a frame close to the end of each video in the `video` column of the table `tbl`:

        >>> tbl.select(
        ...     tbl.video.extract_frame(
        ...         tbl.video.get_metadata().streams[0].duration_seconds - 0.1
        ...     )
        ... ).collect()
    """
    if timestamp < 0:
        raise ValueError("'timestamp' must be non-negative")

    try:
        with av.open(str(video)) as container:
            video_stream = container.streams.video[0]
            time_base = float(video_stream.time_base)
            start_time = video_stream.start_time or 0
            duration = video_stream.duration

            # Check if timestamp is beyond video duration
            if duration is not None:
                duration_seconds = float(duration * time_base)
                if timestamp > duration_seconds:
                    return None

            # Convert timestamp to stream time base units
            target_pts = int(timestamp / time_base) + start_time

            # Seek to the nearest keyframe *before* our target timestamp
            container.seek(target_pts, backward=True, stream=video_stream)

            # Decode frames until we reach or pass the target timestamp
            for frame in container.decode(video=0):
                frame_pts = frame.pts
                if frame_pts is None:
                    continue
                frame_timestamp = (frame_pts - start_time) * time_base
                if frame_timestamp >= timestamp:
                    return frame.to_image()

            return None

    except Exception as e:
        raise pxt.Error(f'extract_frame(): failed to extract frame: {e}') from e


@pxt.udf(is_method=True)
def clip(
    video: pxt.Video,
    *,
    start_time: float,
    end_time: float | None = None,
    duration: float | None = None,
    mode: Literal['fast', 'accurate'] = 'accurate',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video | None:
    """
    Extract a clip from a video, specified by `start_time` and either `end_time` or `duration` (in seconds).

    If `start_time` is beyond the end of the video, returns None. Can only specify one of `end_time` and `duration`.
    If both `end_time` and `duration` are None, the clip goes to the end of the video.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        duration: Duration of the clip in seconds
        mode:

            - `'fast'`: avoids re-encoding but starts the clip at the nearest keyframes and as a result, the clip
                duration will be slightly longer than requested
            - `'accurate'`: extracts a frame-accurate clip, but requires re-encoding
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
            Only available for `mode='accurate'`.
        video_encoder_args: Additional arguments to pass to the video encoder. Only available for `mode='accurate'`.

    Returns:
        New video containing only the specified time range or None if start_time is beyond the end of the video.
    """
    Env.get().require_binary('ffmpeg')
    if start_time < 0:
        raise pxt.Error(f'start_time must be non-negative, got {start_time}')
    if end_time is not None and end_time <= start_time:
        raise pxt.Error(f'end_time ({end_time}) must be greater than start_time ({start_time})')
    if duration is not None and duration <= 0:
        raise pxt.Error(f'duration must be positive, got {duration}')
    if end_time is not None and duration is not None:
        raise pxt.Error('end_time and duration cannot both be specified')
    if mode == 'fast':
        if video_encoder is not None:
            raise pxt.Error("video_encoder is not supported for mode='fast'")
        if video_encoder_args is not None:
            raise pxt.Error("video_encoder_args is not supported for mode='fast'")

    video_duration = av_utils.get_video_duration(video)
    if video_duration is not None and start_time > video_duration:
        return None

    if end_time is not None:
        duration = end_time - start_time
    ffmpeg_args = av_utils.ffmpeg_clip_args(
        str(video),
        start_time,
        duration,
        fast=(mode == 'fast'),
        video_encoder=video_encoder,
        video_encoder_args=video_encoder_args,
    )
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(ffmpeg_args, output_path)


@pxt.udf(is_method=True)
def segment_video(
    video: pxt.Video,
    *,
    duration: float | None = None,
    segment_times: list[float] | None = None,
    mode: Literal['fast', 'accurate'] = 'accurate',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> list[str]:
    """
    Split a video into segments.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video file to segment
        duration: Duration of each segment (in seconds). For `mode='fast'`, this is approximate;
            for `mode='accurate'`, segments will have exact durations. Cannot be specified together with
            `segment_times`.
        segment_times: List of timestamps (in seconds) in video where segments should be split. Note that these are not
            segment durations. If all segment times are less than the duration of the video, produces exactly
            `len(segment_times) + 1` segments. Cannot be empty or be specified together with `duration`.
        mode: Segmentation mode:

            - `'fast'`: Quick segmentation using stream copy (splits only at keyframes, approximate durations)
            - `'accurate'`: Precise segmentation with re-encoding (exact durations, slower)
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
            Only available for `mode='accurate'`.
        video_encoder_args: Additional arguments to pass to the video encoder. Only available for `mode='accurate'`.

    Returns:
        List of file paths for the generated video segments.

    Raises:
        pxt.Error: If the video is missing timing information.

    Examples:
        Split a video at 1 minute intervals using fast mode:

        >>> tbl.select
        ...     segment_paths=tbl.video.segment_video(
        ...         duration=60, mode='fast'
        ...     )
        ... ).collect()

        Split video into exact 10-second segments with default accurate mode, using the libx264 encoder with a CRF of 23
        and slow preset (for smaller output files):

        >>> tbl.select(
        ...     segment_paths=tbl.video.segment_video(
        ...         duration=10,
        ...         video_encoder='libx264',
        ...         video_encoder_args={'crf': 23, 'preset': 'slow'},
        ...     )
        ... ).collect()

        Split video into two parts at the midpoint:

        >>> duration = tbl.video.get_duration()
        >>> tbl.select(
        ...     segment_paths=tbl.video.segment_video(segment_times=[duration / 2])
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if duration is not None and segment_times is not None:
        raise pxt.Error('duration and segment_times cannot both be specified')
    if duration is not None and duration <= 0:
        raise pxt.Error(f'duration must be positive, got {duration}')
    if segment_times is not None and len(segment_times) == 0:
        raise pxt.Error('segment_times cannot be empty')
    if mode == 'fast':
        if video_encoder is not None:
            raise pxt.Error("video_encoder is not supported for mode='fast'")
        if video_encoder_args is not None:
            raise pxt.Error("video_encoder_args is not supported for mode='fast'")

    base_path = TempStore.create_path(extension='')

    output_paths: list[str] = []
    if mode == 'accurate':
        # Use ffmpeg -f segment for accurate segmentation with re-encoding
        output_pattern = f'{base_path}_segment_%04d.mp4'
        cmd = av_utils.ffmpeg_segment_cmd(
            str(video),
            output_pattern,
            segment_duration=duration,
            segment_times=segment_times,
            video_encoder=video_encoder,
            video_encoder_args=video_encoder_args,
        )

        try:
            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_paths = sorted(glob.glob(f'{base_path}_segment_*.mp4'))
            # TODO: is this actually an error?
            # if len(output_paths) == 0:
            #     stderr_output = result.stderr.strip() if result.stderr is not None else ''
            #     raise pxt.Error(
            #         f'ffmpeg failed to create output files for commandline: {" ".join(cmd)}\n{stderr_output}'
            #     )
            return output_paths

        except subprocess.CalledProcessError as e:
            av_utils.handle_ffmpeg_error(e)

    else:
        # Fast mode: extract consecutive clips using stream copy (no re-encoding)
        # This is faster but can only split at keyframes, leading to approximate durations
        start_time = 0.0
        segment_idx = 0
        try:
            while True:
                target_duration: float | None
                if duration is not None:
                    target_duration = duration
                elif segment_idx < len(segment_times):
                    target_duration = segment_times[segment_idx] - start_time
                else:
                    target_duration = None  # the rest
                segment_path = f'{base_path}_segment_{len(output_paths)}.mp4'
                cmd = av_utils.ffmpeg_clip_cmd(str(video), segment_path, start_time, target_duration)

                _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
                segment_duration = av_utils.get_video_duration(segment_path)
                if segment_duration == 0.0:
                    # we're done
                    Path(segment_path).unlink()
                    return output_paths
                output_paths.append(segment_path)
                start_time += segment_duration  # use the actual segment duration here, it won't match duration exactly

                segment_idx += 1
                if segment_times is not None and segment_idx > len(segment_times):
                    break

            return output_paths

        except subprocess.CalledProcessError as e:
            # clean up partial results
            for segment_path in output_paths:
                Path(segment_path).unlink()
            av_utils.handle_ffmpeg_error(e)


def _concat_videos(
    videos: list[str],
    error_prefix: str,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> str | None:
    """Concatenate videos and return the path to the output video, or None for an empty list"""
    Env.get().require_binary('ffmpeg')
    if len(videos) == 0:
        return None

    # Check that all videos have the same resolution
    resolutions: list[tuple[int, int]] = []
    for video in videos:
        metadata = av_utils.get_metadata(str(video))
        video_stream = next((stream for stream in metadata['streams'] if stream['type'] == 'video'), None)
        if video_stream is None:
            raise pxt.Error(f'{error_prefix}: file {video!r} has no video stream')
        resolutions.append((video_stream['width'], video_stream['height']))

    # check for divergence
    x0, y0 = resolutions[0]
    for i, (x, y) in enumerate(resolutions[1:], start=1):
        if (x0, y0) != (x, y):
            raise pxt.Error(
                f'{error_prefix}: requires that all videos have the same resolution, but:'
                f'\n  video 0 ({videos[0]!r}): {x0}x{y0}'
                f'\n  video {i} ({videos[i]!r}): {x}x{y}.'
            )

    # ffmpeg -f concat needs an input file list
    filelist_path = TempStore.create_path(extension='.txt')
    with filelist_path.open('w', encoding='utf-8') as f:
        for video in videos:
            f.write(f'file {video!r}\n')

    output_path = TempStore.create_path(extension='.mp4')

    try:
        # First attempt: fast copy without re-encoding (works for compatible formats)
        cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(filelist_path), '-c', 'copy', '-y', str(output_path)]
        _logger.debug(f'_concat_videos(): {" ".join(cmd)}')
        try:
            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return str(output_path)
        except subprocess.CalledProcessError:
            # Expected for mixed formats - continue to fallback
            pass

        # we might have some corrupted output
        if output_path.exists():
            output_path.unlink()

        # general approach: re-encode with -f filter_complex
        #
        # example: 2 videos with audio:
        #   ffmpeg -i video1.mp4 -i video2.mp4
        #     -filter_complex "[0:v:0][1:v:0]concat=n=2:v=1:a=0[outv];[0:a:0][1:a:0]concat=n=2:v=0:a=1[outa]"
        #     -map "[outv]" -map "[outa]"
        #     ...
        # breakdown:
        # - [0:v:0][1:v:0] - video stream 0 from inputs 0 and 1
        # - concat=n=2:v=1:a=0[outv] - concat 2 inputs, 1 video stream, 0 audio, output to [outv]
        # - [0:a:0][1:a:0] - audio stream 0 from inputs 0 and 1
        # - concat=n=2:v=0:a=1[outa] - concat 2 inputs, 0 video, 1 audio stream, output to [outa]

        cmd = ['ffmpeg']
        for video in videos:
            cmd.extend(['-i', video])

        all_have_audio = all(av_utils.has_audio_stream(str(video)) for video in videos)
        video_inputs = ''.join([f'[{i}:v:0]' for i in range(len(videos))])
        # concat video streams
        filter_str = f'{video_inputs}concat=n={len(videos)}:v=1:a=0[outv]'
        if all_have_audio:
            # also concat audio streams
            audio_inputs = ''.join([f'[{i}:a:0]' for i in range(len(videos))])
            filter_str += f';{audio_inputs}concat=n={len(videos)}:v=0:a=1[outa]'
        cmd.extend(['-filter_complex', filter_str, '-map', '[outv]'])
        if all_have_audio:
            cmd.extend(['-map', '[outa]'])

        av_utils.append_video_encoder(cmd, video_encoder, video_encoder_args)
        if all_have_audio:
            cmd.extend(['-c:a', 'aac'])
        cmd.extend(['-pix_fmt', 'yuv420p', str(output_path)])

        _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return str(output_path)

    except subprocess.CalledProcessError as e:
        av_utils.handle_ffmpeg_error(e)
    finally:
        filelist_path.unlink()


@pxt.udf(is_method=True)
def concat_videos(
    videos: list[pxt.Video], *, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> pxt.Video | None:
    """
    Merge multiple videos into a single video.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        videos: List of videos to merge.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video containing the merged videos, or None if the input list is empty.
    """
    Env.get().require_binary('ffmpeg')
    videos = [v for v in videos if v is not None]
    return _concat_videos(
        videos, error_prefix='concat_videos()', video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.uda(requires_order_by=True)
class concat_videos_agg(pxt.Aggregator):
    """
    Aggregate function that merges videos into a single video.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH
    - All videos must have the same resolution

    Args:
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video containing all input videos concatenated in order, or None if all inputs are None.

    Examples:
        Concatenate all videos in a table, ordered by timestamp:

        >>> tbl.select(concat_videos_agg(tbl.timestamp, tbl.video)).collect()
    """

    videos: list[str]
    video_encoder: str | None
    video_encoder_args: dict[str, Any] | None

    def __init__(self, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None) -> None:
        Env.get().require_binary('ffmpeg')
        self.videos = []
        self.video_encoder = video_encoder
        self.video_encoder_args = video_encoder_args

    def update(self, video: pxt.Video) -> None:
        if video is not None:
            self.videos.append(str(video))

    def value(self) -> pxt.Video | None:
        return _concat_videos(
            self.videos,
            error_prefix='concat_videos_agg()',
            video_encoder=self.video_encoder,
            video_encoder_args=self.video_encoder_args,
        )


@pxt.udf
def with_audio(
    video: pxt.Video,
    audio: pxt.Audio,
    *,
    video_start_time: float = 0.0,
    video_duration: float | None = None,
    audio_start_time: float = 0.0,
    audio_duration: float | None = None,
) -> pxt.Video:
    """
    Creates a new video that combines the video stream from `video` and the audio stream from `audio`.
    The `start_time` and `duration` parameters can be used to select a specific time range from each input.
    If the audio input (or selected time range) is longer than the video, the audio will be truncated.


    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        audio: Input audio.
        video_start_time: Start time in the video input (in seconds).
        video_duration: Duration of video segment (in seconds). If None, uses the remainder of the video after
            `video_start_time`. `video_duration` determines the duration of the output video.
        audio_start_time: Start time in the audio input (in seconds).
        audio_duration: Duration of audio segment (in seconds). If None, uses the remainder of the audio after
            `audio_start_time`. If the audio is longer than the output video, it will be truncated.

    Returns:
        A new video file with the audio track added.

    Examples:
        Add background music to a video:

        >>> tbl.select(tbl.video.with_audio(tbl.music_track)).collect()

        Add audio starting 5 seconds into both files:

        >>> tbl.select(
        ...     tbl.video.with_audio(
        ...         tbl.music_track, video_start_time=5.0, audio_start_time=5.0
        ...     )
        ... ).collect()

        Use a 10-second clip from the middle of both files:

        >>> tbl.select(
        ...     tbl.video.with_audio(
        ...         tbl.music_track,
        ...         video_start_time=30.0,
        ...         video_duration=10.0,
        ...         audio_start_time=15.0,
        ...         audio_duration=10.0,
        ...     )
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if video_start_time < 0:
        raise pxt.Error(f'video_offset must be non-negative, got {video_start_time}')
    if audio_start_time < 0:
        raise pxt.Error(f'audio_offset must be non-negative, got {audio_start_time}')
    if video_duration is not None and video_duration <= 0:
        raise pxt.Error(f'video_duration must be positive, got {video_duration}')
    if audio_duration is not None and audio_duration <= 0:
        raise pxt.Error(f'audio_duration must be positive, got {audio_duration}')

    output_path = str(TempStore.create_path(extension='.mp4'))

    cmd: list[str] = []
    if video_start_time > 0:
        # fast seek, must precede -i
        cmd.extend(['-ss', str(video_start_time)])
    if video_duration is not None:
        cmd.extend(['-t', str(video_duration)])
    else:
        video_duration = av_utils.get_video_duration(video)
    cmd.extend(['-i', str(video)])

    if audio_start_time > 0:
        cmd.extend(['-ss', str(audio_start_time)])
    if audio_duration is not None:
        cmd.extend(['-t', str(audio_duration)])
    cmd.extend(['-i', str(audio)])

    cmd.extend(
        [
            '-map',
            '0:v:0',  # video from first input
            '-map',
            '1:a:0',  # audio from second input
            '-c:v',
            'copy',  # avoid re-encoding
            '-c:a',
            'copy',  # avoid re-encoding
            '-t',
            str(video_duration),  # limit output duration to video duration
        ]
    )
    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


@pxt.udf(is_method=True)
def mix_audio(
    video: pxt.Video,
    audio: pxt.Audio,
    *,
    audio_volume: float = 1.0,
    original_volume: float = 1.0,
    audio_start_time: float = 0.0,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Mix an audio track into a video's existing audio, blending both tracks together. Volume levels for each track can
    be controlled independently.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video (must have an existing audio stream).
        audio: Audio track to mix in.
        audio_volume: Volume multiplier for the added audio track. 1.0 is original volume.
        original_volume: Volume multiplier for the video's existing audio track. 1.0 is original volume.
        audio_start_time: Time in seconds at which the added audio begins playing in the output.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with both audio tracks mixed together.

    Examples:
        Add background music at 30% volume:

        >>> tbl.select(tbl.video.mix_audio(tbl.music, audio_volume=0.3)).collect()

        Mix audio starting at second 5, with the original audio reduced:

        >>> tbl.select(
        ...     tbl.video.mix_audio(
        ...         tbl.music,
        ...         audio_volume=0.5,
        ...         original_volume=0.7,
        ...         audio_start_time=5.0,
        ...     )
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if audio_volume < 0:
        raise pxt.Error(f'audio_volume must be non-negative, got {audio_volume}')
    if original_volume < 0:
        raise pxt.Error(f'original_volume must be non-negative, got {original_volume}')
    if audio_start_time < 0:
        raise pxt.Error(f'audio_start_time must be non-negative, got {audio_start_time}')
    if not av_utils.has_audio_stream(str(video)):
        raise pxt.Error('mix_audio() requires a video with an audio stream')

    output_path = str(TempStore.create_path(extension='.mp4'))

    delay_filter = ''
    if audio_start_time > 0:
        delay_ms = int(audio_start_time * 1000)
        delay_filter = f'adelay={delay_ms}|{delay_ms},'
    filter_complex = (
        f'[0:a]volume={original_volume}[a0];'
        f'[1:a]{delay_filter}volume={audio_volume}[a1];'
        f'[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]'
    )
    cmd = ['-i', str(video), '-i', str(audio), '-filter_complex', filter_complex, '-map', '0:v:0', '-map', '[aout]']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def overlay_text(
    video: pxt.Video,
    text: str,
    *,
    font: str | None = None,
    font_size: int = 24,
    color: str = 'white',
    opacity: float = 1.0,
    horizontal_align: Literal['left', 'center', 'right'] = 'center',
    horizontal_margin: int = 0,
    vertical_align: Literal['top', 'center', 'bottom'] = 'center',
    vertical_margin: int = 0,
    box: bool = False,
    box_color: str = 'black',
    box_opacity: float = 1.0,
    box_border: list[int] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Overlay text on a video with customizable positioning and styling.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video to overlay text on.
        text: The text string to overlay on the video.
        font: Font family or path to font file. If None, uses the system default.
        font_size: Size of the text in points.
        color: Text color (e.g., `'white'`, `'red'`, `'#FF0000'`).
        opacity: Text opacity from 0.0 (transparent) to 1.0 (opaque).
        horizontal_align: Horizontal text alignment (`'left'`, `'center'`, `'right'`).
        horizontal_margin: Horizontal margin in pixels from the alignment edge.
        vertical_align: Vertical text alignment (`'top'`, `'center'`, `'bottom'`).
        vertical_margin: Vertical margin in pixels from the alignment edge.
        box: Whether to draw a background box behind the text.
        box_color: Background box color as a string.
        box_opacity: Background box opacity from 0.0 to 1.0.
        box_border: Padding around text in the box in pixels.

            - `[10]`: 10 pixels on all sides
            - `[10, 20]`: 10 pixels on top/bottom, 20 on left/right
            - `[10, 20, 30]`: 10 pixels on top, 20 on left/right, 30 on bottom
            - `[10, 20, 30, 40]`: 10 pixels on top, 20 on right, 30 on bottom, 40 on left
        start_time: Time in seconds when the text appears. If None, the text is visible from the start.
        end_time: Time in seconds when the text disappears. If None, the text is visible until the end.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the text overlay applied.

    Examples:
        Add a simple text overlay to videos in a table:

        >>> tbl.select(tbl.video.overlay_text('Sample Text')).collect()

        Add a YouTube-style caption:

        >>> tbl.select(
        ...     tbl.video.overlay_text(
        ...         'Caption text',
        ...         font_size=32,
        ...         color='white',
        ...         opacity=1.0,
        ...         box=True,
        ...         box_color='black',
        ...         box_opacity=0.8,
        ...         box_border=[6, 14],
        ...         horizontal_margin=10,
        ...         vertical_align='bottom',
        ...         vertical_margin=70,
        ...     )
        ... ).collect()

        Add text with a semi-transparent background box:

        >>> tbl.select(
        ...     tbl.video.overlay_text(
        ...         'Important Message',
        ...         font_size=32,
        ...         color='yellow',
        ...         box=True,
        ...         box_color='black',
        ...         box_opacity=0.6,
        ...         box_border=[20, 10],
        ...     )
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if font_size <= 0:
        raise pxt.Error(f'font_size must be positive, got {font_size}')
    if opacity < 0.0 or opacity > 1.0:
        raise pxt.Error(f'opacity must be between 0.0 and 1.0, got {opacity}')
    if horizontal_margin < 0:
        raise pxt.Error(f'horizontal_margin must be non-negative, got {horizontal_margin}')
    if vertical_margin < 0:
        raise pxt.Error(f'vertical_margin must be non-negative, got {vertical_margin}')
    if box_opacity < 0.0 or box_opacity > 1.0:
        raise pxt.Error(f'box_opacity must be between 0.0 and 1.0, got {box_opacity}')
    if box_border is not None and not (
        isinstance(box_border, (list, tuple))
        and len(box_border) >= 1
        and len(box_border) <= 4
        and all(isinstance(x, int) for x in box_border)
        and all(x >= 0 for x in box_border)
    ):
        raise pxt.Error(f'box_border must be a list or tuple of 1-4 non-negative ints, got {box_border!s} instead')

    output_path = str(TempStore.create_path(extension='.mp4'))

    if start_time is not None and start_time < 0:
        raise pxt.Error(f'start_time must be non-negative, got {start_time}')
    if end_time is not None and end_time < 0:
        raise pxt.Error(f'end_time must be non-negative, got {end_time}')
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise pxt.Error(f'start_time must be less than end_time, got start_time={start_time}, end_time={end_time}')

    drawtext_params = _create_drawtext_params(
        text,
        font,
        font_size,
        color,
        opacity,
        horizontal_align,
        horizontal_margin,
        vertical_align,
        vertical_margin,
        box,
        box_color,
        box_opacity,
        box_border,
    )

    if start_time is not None or end_time is not None:
        st = start_time if start_time is not None else 0
        et = end_time if end_time is not None else 99999999
        drawtext_params.append(f'enable=between(t\\,{st}\\,{et})')

    cmd = [
        '-i',
        str(video),
        '-vf',
        'drawtext=' + ':'.join(drawtext_params),
        '-c:a',
        'copy',  # Copy audio stream unchanged
    ]
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


def _create_drawtext_params(
    text: str,
    font: str | None,
    font_size: int,
    color: str,
    opacity: float,
    horizontal_align: str,
    horizontal_margin: int,
    vertical_align: str,
    vertical_margin: int,
    box: bool,
    box_color: str,
    box_opacity: float,
    box_border: list[int] | None,
) -> list[str]:
    """Construct parameters for the ffmpeg drawtext filter"""
    drawtext_params: list[str] = []
    escaped_text = text.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")
    drawtext_params.append(f"text='{escaped_text}'")
    drawtext_params.append(f'fontsize={font_size}')

    if font is not None:
        if Path(font).exists():
            drawtext_params.append(f"fontfile='{font}'")
        else:
            drawtext_params.append(f"font='{font}'")
    if opacity < 1.0:
        drawtext_params.append(f'fontcolor={color}@{opacity}')
    else:
        drawtext_params.append(f'fontcolor={color}')

    if horizontal_align == 'left':
        x_expr = str(horizontal_margin)
    elif horizontal_align == 'center':
        x_expr = '(w-text_w)/2'
    else:  # right
        x_expr = f'w-text_w-{horizontal_margin}' if horizontal_margin != 0 else 'w-text_w'
    if vertical_align == 'top':
        y_expr = str(vertical_margin)
    elif vertical_align == 'center':
        y_expr = '(h-text_h)/2'
    else:  # bottom
        y_expr = f'h-text_h-{vertical_margin}' if vertical_margin != 0 else 'h-text_h'
    drawtext_params.extend([f'x={x_expr}', f'y={y_expr}'])

    if box:
        drawtext_params.append('box=1')
        if box_opacity < 1.0:
            drawtext_params.append(f'boxcolor={box_color}@{box_opacity}')
        else:
            drawtext_params.append(f'boxcolor={box_color}')
        if box_border is not None:
            drawtext_params.append(f'boxborderw={"|".join(map(str, box_border))}')

    return drawtext_params


@pxt.udf(is_method=True)
def overlay_image(
    video: pxt.Video,
    image: pxt.Image,
    *,
    horizontal_align: Literal['left', 'center', 'right'] = 'center',
    horizontal_margin: int = 0,
    vertical_align: Literal['top', 'center', 'bottom'] = 'center',
    vertical_margin: int = 0,
    scale: float | None = None,
    opacity: float = 1.0,
    start_time: float | None = None,
    end_time: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Overlay an image on a video with customizable positioning, scaling, opacity, and timing.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video to overlay the image on.
        image: Image to overlay.
        horizontal_align: Horizontal alignment of the overlay (`'left'`, `'center'`, `'right'`).
        horizontal_margin: Horizontal margin in pixels from the alignment edge.
        vertical_align: Vertical alignment of the overlay (`'top'`, `'center'`, `'bottom'`).
        vertical_margin: Vertical margin in pixels from the alignment edge.
        scale: Scale factor for the overlay image relative to the video height. For example, 0.1 scales the
            image to 10% of the video height while preserving aspect ratio. If None, uses the original size.
        opacity: Overlay opacity from 0.0 (transparent) to 1.0 (opaque).
        start_time: Time in seconds when the overlay appears. If None, the overlay is visible from the start.
        end_time: Time in seconds when the overlay disappears. If None, the overlay is visible until the end.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the image overlay applied.

    Examples:
        Add a logo to the top-right corner:

        >>> tbl.select(
        ...     tbl.video.overlay_image(
        ...         tbl.logo_img, horizontal_align='right', vertical_align='top'
        ...     )
        ... ).collect()

        Add a watermark at 50% opacity, scaled to 10% of video height:

        >>> tbl.select(
        ...     tbl.video.overlay_image(tbl.watermark_img, scale=0.1, opacity=0.5)
        ... ).collect()

        Show an image only between seconds 2 and 8:

        >>> tbl.select(
        ...     tbl.video.overlay_image(
        ...         tbl.img, start_time=2.0, end_time=8.0, horizontal_align='right'
        ...     )
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if horizontal_margin < 0:
        raise pxt.Error(f'horizontal_margin must be non-negative, got {horizontal_margin}')
    if vertical_margin < 0:
        raise pxt.Error(f'vertical_margin must be non-negative, got {vertical_margin}')
    if opacity < 0.0 or opacity > 1.0:
        raise pxt.Error(f'opacity must be between 0.0 and 1.0, got {opacity}')
    if scale is not None and scale <= 0:
        raise pxt.Error(f'scale must be positive, got {scale}')
    if start_time is not None and start_time < 0:
        raise pxt.Error(f'start_time must be non-negative, got {start_time}')
    if end_time is not None and end_time < 0:
        raise pxt.Error(f'end_time must be non-negative, got {end_time}')
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise pxt.Error(f'start_time must be less than end_time, got start_time={start_time}, end_time={end_time}')

    output_path = str(TempStore.create_path(extension='.mp4'))

    # ffmpeg needs file input
    image_path = str(TempStore.create_path(extension='.png'))
    image.convert('RGBA').save(image_path)

    x_expr: str
    if horizontal_align == 'left':
        x_expr = str(horizontal_margin)
    elif horizontal_align == 'center':
        x_expr = '(W-w)/2'
    else:  # right
        x_expr = f'W-w-{horizontal_margin}' if horizontal_margin != 0 else 'W-w'

    y_expr: str
    if vertical_align == 'top':
        y_expr = str(vertical_margin)
    elif vertical_align == 'center':
        y_expr = '(H-h)/2'
    else:  # bottom
        y_expr = f'H-h-{vertical_margin}' if vertical_margin != 0 else 'H-h'

    filters: list[str] = []

    overlay_label: str
    if scale is not None:
        md = av_utils.get_metadata(str(video))
        video_height = next(s for s in md['streams'] if s['type'] == 'video')['height']
        filters.append(f'[1:v]scale=-2:trunc({video_height}*{scale}/2)*2[ovr_scaled]')
        overlay_label = '[ovr_scaled]'
    else:
        overlay_label = '[1:v]'

    # apply opacity to the overlay if not fully opaque
    if opacity < 1.0:
        out_label = '[ovr_alpha]'
        filters.append(f'{overlay_label}format=rgba,colorchannelmixer=aa={opacity}{out_label}')
        overlay_label = out_label

    # Build enable clause for timed overlay
    enable_clause = ''
    if start_time is not None or end_time is not None:
        st = start_time if start_time is not None else 0
        et = end_time if end_time is not None else 99999999
        enable_clause = f":enable='between(t,{st},{et})'"

    filters.append(f'[0:v]{overlay_label}overlay={x_expr}:{y_expr}{enable_clause}[vout]')
    filter_complex = ';'.join(filters)

    cmd = [
        '-i',
        str(video),
        '-i',
        image_path,
        '-filter_complex',
        filter_complex,
        '-map',
        '[vout]',
        '-map',
        # 0:a?: make the audio stream optional
        '0:a?',
        '-c:a',
        'copy',
    ]
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def crop(
    video: pxt.Video,
    bbox: list[int],
    *,
    bbox_format: Literal['xyxy', 'xywh', 'cxcywh'] = 'xywh',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Crop a rectangular region from a video using ffmpeg's crop filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        bbox: Crop region as a list of 4 integers.
        bbox_format: Format of the `bbox` coordinates:

            - `'xyxy'`: `[x1, y1, x2, y2]` where (x1, y1) is top-left and (x2, y2) is bottom-right
            - `'xywh'`: `[x, y, width, height]` where (x, y) is top-left corner
            - `'cxcywh'`: `[cx, cy, width, height]` where (cx, cy) is the center
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        Video containing the cropped region.

    Examples:
        Crop using default xywh format:

        >>> tbl.select(tbl.video.crop2([100, 50, 320, 240])).collect()

        Crop using xyxy format (common in object detection):

        >>> tbl.select(
        ...     tbl.video.crop2([100, 50, 420, 290], bbox_format='xyxy')
        ... ).collect()

        Crop using center format:

        >>> tbl.select(
        ...     tbl.video.crop2([260, 170, 320, 240], bbox_format='cxcywh')
        ... ).collect()

        Use with yolox object detection output:

        >>> tbl.add_computed_column(
        ...     cropped=tbl.video.crop2(tbl.detections.bboxes[0], bbox_format='xyxy')
        ... )
    """
    Env.get().require_binary('ffmpeg')

    if len(bbox) != 4 or not all(isinstance(x, int) for x in bbox) or not all(x >= 0 for x in bbox):
        raise pxt.Error(f'bbox must have exactly 4 non-negative integers, got {bbox}')
    if bbox_format == 'xyxy' and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
        raise pxt.Error(f'x2 must be greater than x1 and y2 must be greater than y1 for xyxy format, got {bbox}')

    # normalize to xywh
    x: int
    y: int
    w: int
    h: int
    if bbox_format == 'xyxy':
        x1, y1, x2, y2 = bbox
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
    elif bbox_format == 'xywh':
        x, y, w, h = bbox
    elif bbox_format == 'cxcywh':
        cx, cy, w, h = bbox
        x = cx - w // 2
        y = cy - h // 2
    else:
        raise pxt.Error(f"bbox_format must be one of ['xyxy', 'xywh', 'cxcywh'], got {bbox_format!r}")

    cmd = ['-i', str(video), '-vf', f'crop={w}:{h}:{x}:{y}', '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def resize(
    video: pxt.Video,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Resize a video using ffmpeg's scale filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        width: Width of the output video. Maintains the existing aspect ratio if no `height` is provided.
        height: Height of the output video. Maintains the existing aspect ratio if no `width` is provided.
        scale: Scale factor. Mutually exclusive with `width` and `height`.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        The resized video.

    Examples:
        Resize to a specific width, preserving aspect ratio:

        >>> tbl.select(tbl.video.resize(width=640)).collect()

        Resize to exact dimensions:

        >>> tbl.select(tbl.video.resize(width=1280, height=720)).collect()

        Scale down by half:

        >>> tbl.select(tbl.video.resize(scale=0.5)).collect()
    """
    Env.get().require_binary('ffmpeg')

    if scale is not None and (width is not None or height is not None):
        raise pxt.Error('`scale` is mutually exclusive with `width` and `height`')
    if scale is not None:
        if scale <= 0:
            raise pxt.Error(f'`scale` must be positive, got {scale}')
        scale_filter = f'scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2'
    elif width is not None or height is not None:
        if width is not None and width <= 0:
            raise pxt.Error(f'`width` must be positive, got {width}')
        if height is not None and height <= 0:
            raise pxt.Error(f'`height` must be positive, got {height}')

        # Use -2 for the unspecified dimension: like -1 (preserve aspect ratio),
        # but rounds to the nearest even value (required by most codecs)
        w_expr = str(width) if width is not None else '-2'
        h_expr = str(height) if height is not None else '-2'
        scale_filter = f'scale={w_expr}:{h_expr}'
    else:
        raise pxt.Error('At least one of `width`, `height`, or `scale` must be specified')

    output_path = str(TempStore.create_path(extension='.mp4'))
    cmd = ['-i', str(video), '-vf', scale_filter, '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def reverse(
    video: pxt.Video,
    audio: Literal['reverse', 'drop', 'keep'] = 'drop',
    *,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Reverse a video using ffmpeg's reverse filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        audio: Specifies what to do with audio streams

            - `'drop'`: drop the audio streams
            - `'reverse'`: also reverse the audio streams
            - `'keep'`: keep the audio streams
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        The reversed video.

    Examples:
        Reverse a video, dropping audio:

        >>> tbl.select(tbl.video.reverse()).collect()

        Reverse a video along with its audio:

        >>> tbl.select(tbl.video.reverse(audio='reverse')).collect()
    """

    Env.get().require_binary('ffmpeg')

    # ffmpeg's reverse filter requires all frames to be decoded into memory at once, which can exhaust RAM on
    # long or high-resolution videos. To avoid this, we split the video into segments whose decoded frames fit
    # within ~1 GB, reverse each segment independently, then concatenate the reversed segments in reverse order.
    segment_bytes = 2**30
    segment_duration = av_utils.estimate_segment_duration(video, segment_bytes)
    if segment_duration is None:
        raise pxt.Error(f'not a valid video: {video}')

    duration = av_utils.get_video_duration(video)
    if duration is None:
        raise excs.Error(f'reverse(): could not determine video duration: {video}')

    with av.open(video) as container:
        has_audio = any(s.type == 'audio' for s in container.streams)

    starts = [segment_duration * i for i in range(math.ceil(duration / segment_duration))]

    # Build the filtergraph. For a 25s video with segment_duration=10, starts=[0, 10, 20] and the filtergraph is:
    #
    #   [0:v]trim=start=0:end=10,setpts=PTS-STARTPTS,reverse[v0];
    #   [0:v]trim=start=10:end=20,setpts=PTS-STARTPTS,reverse[v1];
    #   [0:v]trim=start=20,setpts=PTS-STARTPTS,reverse[v2];
    #   [v2][v1][v0]concat=n=3:v=1:a=0[v]
    #
    # Each segment is: trim to time range -> reset timestamps -> reverse.
    # The last segment omits :end= so it runs to the end of the stream.
    # The concat inputs are listed in reverse order ([v2][v1][v0]) so the last segment of the original
    # video becomes the first segment of the output.
    n = len(starts)
    filter_parts: list[str] = []

    for i, start in enumerate(starts):
        is_last = i == n - 1
        end_clause = '' if is_last else f':end={start + segment_duration}'

        filter_parts.append(f'[0:v]trim=start={start}{end_clause},setpts=PTS-STARTPTS,reverse[v{i}]')
        if audio == 'reverse' and has_audio:
            filter_parts.append(f'[0:a]atrim=start={start}{end_clause},asetpts=PTS-STARTPTS,areverse[a{i}]')

    v_inputs = ''.join(f'[v{i}]' for i in range(n - 1, -1, -1))
    filter_parts.append(f'{v_inputs}concat=n={n}:v=1:a=0[v]')

    if audio == 'reverse' and has_audio:
        a_inputs = ''.join(f'[a{i}]' for i in range(n - 1, -1, -1))
        filter_parts.append(f'{a_inputs}concat=n={n}:v=0:a=1[a]')

    filtergraph = '; '.join(filter_parts)

    # Example commandline (audio='reverse'):
    #   ffmpeg -i input.mp4 -filter_complex "<filtergraph>" -map [v] -map [a] -loglevel error out.mp4
    # audio='keep': -map 0:a -c:a copy (passes original audio through without the filtergraph)
    # audio='drop': no audio mapping, so ffmpeg omits audio from the output
    cmd = ['-i', str(video), '-filter_complex', filtergraph, '-map', '[v]']
    # we need to add the video encoder args at this point (not later)
    av_utils.append_video_encoder(cmd, video_encoder, video_encoder_args)

    if audio == 'reverse' and has_audio:
        cmd.extend(['-map', '[a]'])
    elif audio == 'keep' and has_audio:
        cmd.extend(['-map', '0:a', '-c:a', 'copy'])

    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


def _fade(
    video: str,
    direction: Literal['in', 'out'],
    duration: float,
    color: str,
    video_duration: float | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> str:
    Env.get().require_binary('ffmpeg')
    if duration <= 0:
        raise pxt.Error(f'duration must be positive, got {duration}')

    if direction == 'in':
        start_time = 0.0
    else:
        assert video_duration is not None
        start_time = max(0, video_duration - duration)

    output_path = str(TempStore.create_path(extension='.mp4'))
    fade_filter = f'fade={direction}:st={start_time}:d={duration}:color={color}'
    cmd = ['-i', str(video), '-vf', fade_filter, '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def fade_in(
    video: pxt.Video,
    *,
    duration: float = 1.0,
    color: str = 'black',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a fade-in effect from a solid color at the start of a video using ffmpeg's fade filter.
    The video transitions from a solid `color` to the full video content over `duration` seconds.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        duration: Duration of the fade-in effect in seconds.
        color: Color to fade from (e.g., `'black'`, `'white'`, `'#FF0000'`).
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the fade-in effect applied.

    Examples:
        Apply a 1-second fade from black (default):

        >>> tbl.select(tbl.video.fade_in()).collect()

        Apply a 2-second fade from white:

        >>> tbl.select(tbl.video.fade_in(duration=2.0, color='white')).collect()
    """
    return _fade(video, 'in', duration, color, video_encoder=video_encoder, video_encoder_args=video_encoder_args)


@pxt.udf(is_method=True)
def fade_out(
    video: pxt.Video,
    *,
    duration: float = 1.0,
    color: str = 'black',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a fade-out effect to a solid color at the end of a video using ffmpeg's fade filter.
    The video transitions from the full video content to a solid `color` over the final `duration` seconds.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        duration: Duration of the fade-out effect in seconds.
        color: Color to fade to (e.g., `'black'`, `'white'`, `'#FF0000'`).
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the fade-out effect applied.

    Examples:
        Apply a 1-second fade to black (default):

        >>> tbl.select(tbl.video.fade_out()).collect()

        Apply a 3-second fade to white:

        >>> tbl.select(tbl.video.fade_out(duration=3.0, color='white')).collect()
    """
    video_duration = av_utils.get_video_duration(video)
    if video_duration is None:
        raise pxt.Error('fade_out(): could not determine video duration')
    return _fade(
        video,
        'out',
        duration,
        color,
        video_duration=video_duration,
        video_encoder=video_encoder,
        video_encoder_args=video_encoder_args,
    )


@pxt.udf
def transition(
    video1: pxt.Video,
    video2: pxt.Video,
    *,
    effect: Literal[
        'fade',
        'wipeleft',
        'wiperight',
        'wipeup',
        'wipedown',
        'slideleft',
        'slideright',
        'slideup',
        'slidedown',
        'dissolve',
        'smoothleft',
        'smoothright',
        'smoothup',
        'smoothdown',
    ] = 'fade',
    duration: float = 1.0,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Join two video clips with a transition effect using ffmpeg's xfade filter.

    Applies a crossfade or other transition effect between the end of the first clip and the beginning of
    the second clip. The transition overlaps the last `duration` seconds of `video1` with the first `duration`
    seconds of `video2`, so the total output duration is `len(video1) + len(video2) - duration`.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video1: First video clip.
        video2: Second video clip.
        effect: Transition effect type. Common options:

            - `'fade'`: Classic crossfade (default).
            - `'dissolve'`: Dissolve transition.
            - `'wipeleft'`, `'wiperight'`, `'wipeup'`, `'wipedown'`: Wipe transitions.
            - `'slideleft'`, `'slideright'`, `'slideup'`, `'slidedown'`: Slide transitions.
            - `'smoothleft'`, `'smoothright'`, `'smoothup'`, `'smoothdown'`: Smooth transitions.
        duration: Duration of the transition in seconds.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the transition applied between the two clips.

    Examples:
        Join two clips with a 1-second crossfade:

        >>> tbl.select(transition(tbl.clip1, tbl.clip2)).collect()

        Join with a 2-second wipe-left transition:

        >>> tbl.select(
        ...     transition(tbl.clip1, tbl.clip2, effect='wipeleft', duration=2.0)
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if duration <= 0:
        raise pxt.Error(f'duration must be positive, got {duration}')

    # xfade requires both inputs to have the same resolution
    md1 = av_utils.get_metadata(str(video1))
    v1_stream = next(s for s in md1['streams'] if s['type'] == 'video')
    w1, h1 = v1_stream['width'], v1_stream['height']
    md2 = av_utils.get_metadata(str(video2))
    v2_stream = next(s for s in md2['streams'] if s['type'] == 'video')
    w2, h2 = v2_stream['width'], v2_stream['height']
    if (w1, h1) != (w2, h2):
        raise pxt.Error(f'video1 and video2 must have the same resolution, got {w1}x{h1} and {w2}x{h2}')

    video1_duration = av_utils.get_video_duration(video1)
    if video1_duration is None:
        raise pxt.Error(f'Could not determine duration of {video1}')
    if duration > video1_duration:
        raise pxt.Error(f'transition duration ({duration}s) exceeds duration ({video1_duration}s) of {video1}')
    video2_duration = av_utils.get_video_duration(video2)
    if video2_duration is None:
        raise pxt.Error(f'Could not determine duration of {video2}')
    if duration > video2_duration:
        raise pxt.Error(f'transition duration ({duration}s) exceeds duration ({video2_duration}s) of {video2}')

    offset = video1_duration - duration
    output_path = str(TempStore.create_path(extension='.mp4'))

    # build xfade filter; handle audio with acrossfade if both clips have audio
    has_audio1 = av_utils.has_audio_stream(video1)
    has_audio2 = av_utils.has_audio_stream(video2)

    filter_complex = f'[0:v][1:v]xfade=transition={effect}:duration={duration}:offset={offset}[vout]'
    if has_audio1 and has_audio2:
        filter_complex += f';[0:a][1:a]acrossfade=d={duration}[aout]'
    cmd = ['-i', str(video1), '-i', str(video2), '-filter_complex', filter_complex, '-map', '[vout]']
    if has_audio1 and has_audio2:
        cmd.extend(['-map', '[aout]', '-c:a', 'aac'])
    elif has_audio1:
        cmd.extend(['-map', '0:a', '-c:a', 'copy'])
    elif has_audio2:
        cmd.extend(['-map', '1:a', '-c:a', 'copy'])

    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def speed(
    video: pxt.Video,
    *,
    factor: float,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Change the playback speed of a video using ffmpeg's setpts filter.

    A factor of 2.0 doubles the speed (halves the duration); a factor of 0.5 halves the speed (doubles the duration).
    Audio pitch is preserved using ffmpeg's `atempo` filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        factor: Speed multiplier. Must be positive. Values > 1.0 speed up, values < 1.0 slow down.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the adjusted playback speed.

    Examples:
        Double the speed:

        >>> tbl.select(tbl.video.speed(factor=2.0)).collect()

        Half speed (slow motion):

        >>> tbl.select(tbl.video.speed(factor=0.5)).collect()
    """
    Env.get().require_binary('ffmpeg')
    if factor <= 0:
        raise pxt.Error(f'factor must be positive, got {factor}')

    output_path = str(TempStore.create_path(extension='.mp4'))
    # setpts=PTS/<factor> adjusts video timing; atempo=<factor> adjusts audio speed (preserving pitch).
    # atempo only accepts values in [0.5, 100.0]; for slower speeds, chain multiple atempo filters.
    video_filter = f'setpts=PTS/{factor}'
    has_audio = av_utils.has_audio_stream(video)

    cmd = ['-i', str(video), '-vf', video_filter]
    # add video encoder args here
    av_utils.append_video_encoder(cmd, video_encoder, video_encoder_args)

    if has_audio:
        # Chain atempo filters for factors outside [0.5, 100.0]
        atempo_parts = []
        remaining = factor
        while remaining < 0.5:
            atempo_parts.append('atempo=0.5')
            remaining /= 0.5
        while remaining > 100.0:
            atempo_parts.append('atempo=100.0')
            remaining /= 100.0
        atempo_parts.append(f'atempo={remaining}')
        cmd.extend(['-af', ','.join(atempo_parts)])
    else:
        cmd.append('-an')

    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


def _flip(
    video: str,
    orientation: Literal['h', 'v'],
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> str:
    Env.get().require_binary('ffmpeg')
    flip_filter = 'hflip' if orientation == 'h' else 'vflip'
    cmd = ['-i', str(video), '-vf', flip_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def mirror_x(
    video: pxt.Video, *, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> pxt.Video:
    """
    Flip a video horizontally using ffmpeg's hflip filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A horizontally flipped video.

    Examples:
        >>> tbl.select(tbl.video.mirror_x()).collect()
    """
    return _flip(video, 'h', video_encoder, video_encoder_args)


@pxt.udf(is_method=True)
def mirror_y(
    video: pxt.Video, *, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> pxt.Video:
    """
    Flip a video vertically using ffmpeg's vflip filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A vertically flipped video.

    Examples:
        >>> tbl.select(tbl.video.mirror_y()).collect()
    """
    return _flip(video, 'v', video_encoder, video_encoder_args)


@pxt.udf(is_method=True)
def rotate(
    video: pxt.Video,
    *,
    angle: float,
    unit: Literal['deg', 'rad'] = 'deg',
    expand: bool = False,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Rotate a video by a fixed angle using ffmpeg's rotate filter.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        angle: Rotation angle. Positive values rotate counter-clockwise.
        unit: Unit of the angle: `'deg'` for degrees or `'rad'` for radians.
        expand: If True, the output frame is enlarged to contain the entire rotated frame (no cropping).
            If False (default), the output frame keeps the original dimensions, cropping corners.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video rotated by the specified angle.

    Examples:
        Rotate 90 degrees counter-clockwise:

        >>> tbl.select(tbl.video.rotate(angle=90)).collect()

        Rotate 45 degrees with frame expansion to avoid cropping:

        >>> tbl.select(tbl.video.rotate(angle=45, expand=True)).collect()

        Rotate by pi/2 radians:

        >>> tbl.select(tbl.video.rotate(angle=1.5708, unit='rad')).collect()
    """
    Env.get().require_binary('ffmpeg')

    # Convert to radians for ffmpeg's rotate filter
    angle_rad = angle if unit == 'rad' else angle * math.pi / 180

    if expand:
        # Expand output to fit the rotated frame: compute new dimensions from the rotation angle
        # For a WxH frame rotated by A: new_w = W*|cos(A)| + H*|sin(A)|, new_h = W*|sin(A)| + H*|cos(A)|
        # Use ffmpeg's rotate filter with out_w/out_h expressions
        rotate_filter = (
            f'rotate={angle_rad}'
            f":ow='ceil((iw*abs(cos({angle_rad}))+ih*abs(sin({angle_rad})))/2)*2'"
            f":oh='ceil((iw*abs(sin({angle_rad}))+ih*abs(cos({angle_rad})))/2)*2'"
            f':fillcolor=black'
        )
    else:
        rotate_filter = f'rotate={angle_rad}:fillcolor=black'

    cmd = ['-i', str(video), '-vf', rotate_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def grayscale(
    video: pxt.Video, *, video_encoder: str | None = None, video_encoder_args: dict[str, Any] | None = None
) -> pxt.Video:
    """
    Convert a video to grayscale

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A grayscale version of the video.

    Examples:
        >>> tbl.select(tbl.video.grayscale()).collect()
    """
    Env.get().require_binary('ffmpeg')

    output_path = str(TempStore.create_path(extension='.mp4'))

    # Convert to grayscale via hue filter (set saturation to 0), which keeps the yuv420p pixel format
    # compatible with most encoders. Using format=gray would produce a single-channel output that
    # many players and encoders don't handle well.
    cmd = ['-i', str(video), '-vf', 'hue=s=0', '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def adjust_brightness(
    video: pxt.Video,
    *,
    factor: float,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Adjust the brightness of a video by a multiplicative factor using ffmpeg's lutrgb filter.

    A factor of 1.0 leaves the video unchanged; values below 1.0 dim the video (e.g., 0.5 for 50% brightness),
    and values above 1.0 brighten it (e.g., 1.5 for 150% brightness).

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        factor: Brightness multiplier. 0.0 produces a black video, 1.0 is unchanged, values > 1.0 brighten.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with adjusted brightness.

    Examples:
        Dim a video to 50% brightness:

        >>> tbl.select(tbl.video.adjust_brightness(factor=0.5)).collect()

        Brighten a video by 20%:

        >>> tbl.select(tbl.video.adjust_brightness(factor=1.2)).collect()
    """
    Env.get().require_binary('ffmpeg')
    if factor < 0:
        raise pxt.Error(f'factor must be non-negative, got {factor}')

    # FFmpeg eq filter: brightness is additive (-1.0 to 1.0), gamma is multiplicative.
    # Using curves filter with a master curve for true multiplicative brightness.
    # For the eq filter, we use gamma_r/gamma_g/gamma_b which are multiplicative.
    # However, the simplest approach: use the lut filter to multiply pixel values.
    output_path = str(TempStore.create_path(extension='.mp4'))
    # Clamp to 0-255: min(val*factor, 255)
    lut_expr = f"'min(val*{factor},255)'"
    cmd = ['-i', str(video), '-vf', f'lutrgb=r={lut_expr}:g={lut_expr}:b={lut_expr}', '-c:a', 'copy']
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def ffmpeg_filter(
    video: pxt.Video,
    *,
    vf: str,
    af: str | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply an arbitrary FFmpeg filter expression to a video.

    The `vf` string is passed directly as the `-vf` argument to FFmpeg. If `af` is
    also provided, it is passed as the `-af` argument; otherwise the audio stream is copied unchanged.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        vf: FFmpeg video filter string, passed as `-vf`.
        af: Optional FFmpeg audio filter string, passed as `-af`. If None, the audio stream is copied
            unchanged. The input video must have an audio stream when `af` is provided.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the filter(s) applied.

    Examples:
        Apply a sepia tone:

        >>> tbl.select(
        ...     tbl.video.ffmpeg_filter(
        ...         vf='colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131'
        ...     )
        ... ).collect()

        Sharpen a video:

        >>> tbl.select(tbl.video.ffmpeg_filter(vf='unsharp=5:5:1.5')).collect()

        Add a vignette with audio normalization:

        >>> tbl.select(
        ...     tbl.video.ffmpeg_filter(vf='vignette', af='loudnorm')
        ... ).collect()

        Chain multiple video filters:

        >>> tbl.select(
        ...     tbl.video.ffmpeg_filter(vf='eq=brightness=0.1,hue=h=30')
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')

    output_path = str(TempStore.create_path(extension='.mp4'))
    cmd = ['-i', str(video), '-vf', vf]
    if af is not None:
        cmd.extend(['-af', af])
    else:
        cmd.extend(['-c:a', 'copy'])
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


def pan(video: Any, *, direction: Literal['left', 'right', 'up', 'down'] = 'right', crop_pct: float = 0.2) -> Any:
    """
    Apply a smooth pan effect across a video. Convenience function for
    [`scroll()`][pixeltable.functions.video.scroll] that automatically computes viewport size and speed from the
    video's dimensions and duration, panning across the full available range.

    The effect crops a viewport that is `(1 - crop_pct)` of the original dimension in the pan direction and smoothly
    slides it across the full available range over the video's duration.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: A pixeltable Video expression (e.g., `tbl.video`).
        direction: Pan direction: `'left'`, `'right'`, `'up'`, or `'down'`.
        crop_pct: Fraction of the dimension (width for left/right, height for up/down) used as panning range,
            between 0.0 (exclusive) and 1.0 (exclusive). Larger values produce more pronounced panning but a
            narrower output. Default is 0.2 (viewport is 80% of the original dimension).

    Returns:
        A panned video.

    Examples:
        Pan rightward (default):

        >>> tbl.select(pan(tbl.video)).collect()

        Pan leftward with a wider range:

        >>> tbl.select(pan(tbl.video, direction='left', crop_pct=0.4)).collect()

        Pan downward:

        >>> tbl.select(pan(tbl.video, direction='down')).collect()
    """
    import pixeltable.functions.math as pxtmath

    if crop_pct <= 0.0 or crop_pct >= 1.0:
        raise pxt.Error(f'crop_pct must be between 0.0 and 1.0 (exclusive), got {crop_pct}')

    md = video.get_metadata()
    w = md.streams[0].width
    h = md.streams[0].height
    duration = video.get_duration()

    if direction in ('left', 'right'):
        viewport_w = pxtmath.floor(w * (1 - crop_pct)).to_int()
        pan_range = w - viewport_w
        speed = pan_range / duration
        if direction == 'right':
            return video.scroll(w=viewport_w, x_speed=speed)
        else:
            return video.scroll(w=viewport_w, x_start=pan_range.to_int(), x_speed=-speed)
    elif direction in ('up', 'down'):
        viewport_h = pxtmath.floor(h * (1 - crop_pct)).to_int()
        pan_range = h - viewport_h
        speed = pan_range / duration
        if direction == 'down':
            return video.scroll(h=viewport_h, y_speed=speed)
        else:
            return video.scroll(h=viewport_h, y_start=pan_range.to_int(), y_speed=-speed)
    else:
        raise pxt.Error(f"direction must be one of 'left', 'right', 'up', 'down', got {direction!r}")


@pxt.udf(is_method=True)
def scroll(
    video: pxt.Video,
    *,
    w: int | None = None,
    h: int | None = None,
    x_speed: float = 0,
    y_speed: float = 0,
    x_start: int = 0,
    y_start: int = 0,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a scrolling viewport effect to a video using ffmpeg's crop filter.

    Extracts a viewport of size `w` x `h` from each frame, starting at position (`x_start`, `y_start`) and moving
    at (`x_speed`, `y_speed`) pixels per second. The viewport clamps at the frame edges: once it reaches a boundary,
    it stops moving and the remaining frames show a static crop.

    At least one of `w` or `h` must be smaller than the input dimensions for the effect to be visible.

    The clip duration is unchanged. To pan across the full available range, set
    `x_speed = (input_width - w) / duration`.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        w: Width of the output viewport in pixels. If None, uses the input width.
        h: Height of the output viewport in pixels. If None, uses the input height.
        x_speed: Horizontal scroll speed in pixels per second. Positive values scroll rightward (the viewport moves
            right, revealing content to the right). Negative values scroll leftward.
        y_speed: Vertical scroll speed in pixels per second. Positive values scroll downward. Negative values scroll
            upward.
        x_start: Initial horizontal offset of the viewport in pixels.
        y_start: Initial vertical offset of the viewport in pixels.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the scrolling effect applied. Output dimensions are `w` x `h`.

    Examples:
        Pan rightward across a 1920x1080 video using a 1280-pixel-wide viewport, scrolling at 50 px/s:

        >>> tbl.select(tbl.video.scroll(w=1280, x_speed=50)).collect()

        Pan rightward across the full range of a 1920x1080 video in exactly its duration. The viewport is
        1280 px wide, so the pan range is 1920 - 1280 = 640 px. For a 10-second video, set
        `x_speed = 640 / 10 = 64`:

        >>> tbl.select(tbl.video.scroll(w=1280, x_speed=64)).collect()

        Pan leftward across a 1920x1080 video, starting from the right edge:

        >>> tbl.select(tbl.video.scroll(w=1280, x_start=640, x_speed=-64)).collect()
    """
    Env.get().require_binary('ffmpeg')

    if x_speed == 0 and y_speed == 0:
        raise pxt.Error('at least one of `x_speed` or `y_speed` must be non-zero')
    if w is None and h is None:
        raise pxt.Error('at least one of `w` or `h` must be specified')
    if w is not None and w <= 0:
        raise pxt.Error(f'`w` must be positive, got {w}')
    if h is not None and h <= 0:
        raise pxt.Error(f'`h` must be positive, got {h}')

    # Read input dimensions to fill in defaults and validate
    with av.open(video) as container:
        video_stream = container.streams.video[0]
        in_w = video_stream.width
        in_h = video_stream.height

    out_w = w if w is not None else in_w
    out_h = h if h is not None else in_h

    if out_w > in_w or out_h > in_h:
        raise pxt.Error(f'viewport ({out_w}x{out_h}) must not exceed input dimensions ({in_w}x{in_h})')
    if out_w == in_w and out_h == in_h:
        raise pxt.Error(
            f'viewport ({out_w}x{out_h}) equals input dimensions; at least one must be smaller for scrolling'
        )

    x_max = in_w - out_w
    y_max = in_h - out_h
    if x_start < 0 or x_start > x_max:
        raise pxt.Error(f'x_start must be between 0 and {x_max}, got {x_start}')
    if y_start < 0 or y_start > y_max:
        raise pxt.Error(f'y_start must be between 0 and {y_max}, got {y_start}')

    # Build the crop filter with time-dependent x/y expressions and edge clamping.
    # Example for w=1280 on a 1920-wide input, x_start=0, x_speed=64:
    # crop=1280:1080:min(640\,max(0\,0+64*t)):0
    x_expr = f'min({x_max}\\,max(0\\,{x_start}+{x_speed}*t))'
    y_expr = f'min({y_max}\\,max(0\\,{y_start}+{y_speed}*t))'
    crop_filter = f'crop={out_w}:{out_h}:{x_expr}:{y_expr}'

    cmd = ['-i', str(video), '-vf', crop_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def zoom(
    video: pxt.Video,
    *,
    start_scale: float = 1.0,
    end_scale: float = 1.3,
    center: list[float] | None = None,
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> pxt.Video:
    """
    Apply a smooth zoom effect over the duration of a video using ffmpeg's zoompan filter.

    The zoom factor interpolates linearly from `start_scale` to `end_scale`. The effect works by computing a crop
    region at each frame (centered on `center`) and scaling it back to the original resolution. Output dimensions
    match the input.

    - `start_scale < end_scale`: zoom in (frame progressively tightens)
    - `start_scale > end_scale`: zoom out (frame progressively widens)
    - `start_scale == end_scale`: static zoom (constant crop, no animation)

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video.
        start_scale: Zoom factor at the start of the video. Must be >= 1.0.
        end_scale: Zoom factor at the end of the video. Must be >= 1.0.
        center: Zoom center as `[x, y]` in normalized coordinates (0.0 to 1.0), where `[0.5, 0.5]` is the frame
            center. If None, defaults to `[0.5, 0.5]`.
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with the zoom effect applied. Output resolution matches the input.

    Examples:
        Zoom in (default, 1.0x to 1.3x centered):

        >>> tbl.select(tbl.video.zoom()).collect()

        Zoom out from 2x to 1x:

        >>> tbl.select(tbl.video.zoom(start_scale=2.0, end_scale=1.0)).collect()

        Zoom in toward the upper-left quadrant:

        >>> tbl.select(tbl.video.zoom(end_scale=1.5, center=[0.25, 0.25])).collect()

        Static 1.5x zoom (no animation):

        >>> tbl.select(tbl.video.zoom(start_scale=1.5, end_scale=1.5)).collect()
    """
    Env.get().require_binary('ffmpeg')

    if start_scale < 1.0:
        raise pxt.Error(f'start_scale must be >= 1.0, got {start_scale}')
    if end_scale < 1.0:
        raise pxt.Error(f'end_scale must be >= 1.0, got {end_scale}')
    if center is not None and (len(center) != 2 or not all(0.0 <= c <= 1.0 for c in center)):
        raise pxt.Error(f'center must be [x, y] with values in [0.0, 1.0], got {center}')
    cx, cy = center if center is not None else [0.5, 0.5]

    with av.open(video) as container:
        video_stream = container.streams.video[0]
        in_w = video_stream.width
        in_h = video_stream.height
        fps = float(video_stream.average_rate)

    # zoompan evaluates z/x/y expressions per frame.
    # 'on' is the output frame number (0-based); we use it to interpolate the zoom factor linearly.
    # 'd=1' means each input frame produces exactly 1 output frame

    # Example for start_scale=1.0, end_scale=1.3, center=(0.5, 0.5), fps=25, 10s 1920x1080 video:
    #   zoompan=z='1.0+(1.3-1.0)*on/249':x='iw*0.5*(1-1/zoom)':y='ih*0.5*(1-1/zoom)'
    #           :d=1:s=1920x1080:fps=25
    duration = av_utils.get_video_duration(video)
    if duration is None:
        raise pxt.Error('zoom(): could not determine video duration')
    total_frames = max(1, round(fps * duration))

    # z interpolates linearly from start_scale to end_scale over total_frames
    z_expr = f'{start_scale}+({end_scale}-{start_scale})*on/{max(1, total_frames - 1)}'

    # x/y position the crop so that the normalized center point (cx, cy) stays fixed.
    # For center (cx, cy): x = iw*cx*(1 - 1/zoom)
    x_expr = f'iw*{cx}*(1-1/zoom)'
    y_expr = f'ih*{cy}*(1-1/zoom)'

    zoompan_filter = f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d=1:s={in_w}x{in_h}:fps={fps}"

    cmd = ['-i', str(video), '-vf', zoompan_filter, '-c:a', 'copy']
    output_path = str(TempStore.create_path(extension='.mp4'))
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


@pxt.udf(is_method=True)
def scene_detect_adaptive(
    video: pxt.Video,
    *,
    fps: float | None = None,
    adaptive_threshold: float = 3.0,
    min_scene_len: int = 15,
    window_width: int = 2,
    min_content_val: float = 15.0,
    delta_hue: float = 1.0,
    delta_sat: float = 1.0,
    delta_lum: float = 1.0,
    delta_edges: float = 0.0,
    luma_only: bool = False,
    kernel_size: int | None = None,
) -> list[dict]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [AdaptiveDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.adaptive_detector.AdaptiveDetector).

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None or 0, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        adaptive_threshold: Threshold that the score ratio must exceed to trigger a new scene cut.
            Lower values will detect more scenes (more sensitive), higher values will detect fewer scenes.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.
        window_width: Size of window (number of frames) before and after each frame to average together in order to
            detect deviations from the mean. Must be at least 1.
        min_content_val: Minimum threshold (float) that the content_val must exceed in order to register as a new scene.
            This is calculated the same way that `scene_detect_content()` calculates frame
            score based on weights/luma_only/kernel_size.
        delta_hue: Weight for hue component changes. Higher values make hue changes more important.
        delta_sat: Weight for saturation component changes. Higher values make saturation changes more important.
        delta_lum: Weight for luminance component changes. Higher values make brightness changes more important.
        delta_edges: Weight for edge detection changes. Higher values make edge changes more important.
            Edge detection can help detect cuts in scenes with similar colors but different content.
        luma_only: If True, only analyzes changes in the luminance (brightness) channel of the video,
            ignoring color information. This can be faster and may work better for grayscale content.
        kernel_size: Size of kernel to use for post edge detection filtering. If None, automatically set based on video
            resolution.

    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_adaptive()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(
        ...     tbl.video.scene_detect_adaptive(adaptive_threshold=1.5)
        ... ).collect()

        Use luminance-only detection with a longer minimum scene length:

        >>> tbl.select(
        ...     tbl.video.scene_detect_adaptive(luma_only=True, min_scene_len=30)
        ... ).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(
        ...     scene_cuts=tbl.video.scene_detect_adaptive(adaptive_threshold=2.0)
        ... )

        Analyze at a lower frame rate for faster processing:

        >>> tbl.select(tbl.video.scene_detect_adaptive(fps=2.0)).collect()
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import AdaptiveDetector, ContentDetector

    weights = ContentDetector.Components(
        delta_hue=delta_hue, delta_sat=delta_sat, delta_lum=delta_lum, delta_edges=delta_edges
    )
    try:
        detector = AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len,
            window_width=window_width,
            min_content_val=min_content_val,
            weights=weights,
            luma_only=luma_only,
            kernel_size=kernel_size,
        )
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.Error(f'scene_detect_adaptive(): failed to detect scenes: {e}') from e


@pxt.udf(is_method=True)
def scene_detect_content(
    video: pxt.Video,
    *,
    fps: float | None = None,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    delta_hue: float = 1.0,
    delta_sat: float = 1.0,
    delta_lum: float = 1.0,
    delta_edges: float = 0.0,
    luma_only: bool = False,
    kernel_size: int | None = None,
    filter_mode: Literal['merge', 'suppress'] = 'merge',
) -> list[dict]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [ContentDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.content_detector.ContentDetector).

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        threshold: Threshold that the weighted sum of component changes must exceed to trigger a scene cut.
            Lower values detect more scenes (more sensitive), higher values detect fewer scenes.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.
        delta_hue: Weight for hue component changes. Higher values make hue changes more important.
        delta_sat: Weight for saturation component changes. Higher values make saturation changes more important.
        delta_lum: Weight for luminance component changes. Higher values make brightness changes more important.
        delta_edges: Weight for edge detection changes. Higher values make edge changes more important.
            Edge detection can help detect cuts in scenes with similar colors but different content.
        luma_only: If True, only analyzes changes in the luminance (brightness) channel,
            ignoring color information. This can be faster and may work better for grayscale content.
        kernel_size: Size of kernel for expanding detected edges. Must be odd integer greater than or equal to 3. If
            None, automatically set using video resolution.
        filter_mode: How to handle fast cuts/flashes. 'merge' combines quick cuts, 'suppress' filters them out.

    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_content()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(tbl.video.scene_detect_content(threshold=15.0)).collect()

        Use luminance-only detection:

        >>> tbl.select(tbl.video.scene_detect_content(luma_only=True)).collect()

        Emphasize edge detection for scenes with similar colors:

        >>> tbl.select(
        ...     tbl.video.scene_detect_content(
        ...         delta_edges=1.0, delta_hue=0.5, delta_sat=0.5
        ...     )
        ... ).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(
        ...     scene_cuts=tbl.video.scene_detect_content(threshold=20.0)
        ... )
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import ContentDetector
    from scenedetect.detectors.content_detector import FlashFilter  # type: ignore[import-untyped]

    weights = ContentDetector.Components(
        delta_hue=delta_hue, delta_sat=delta_sat, delta_lum=delta_lum, delta_edges=delta_edges
    )
    filter_mode_enum = FlashFilter.Mode.MERGE if filter_mode == 'merge' else FlashFilter.Mode.SUPPRESS

    try:
        detector = ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
            weights=weights,
            luma_only=luma_only,
            kernel_size=kernel_size,
            filter_mode=filter_mode_enum,
        )
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.Error(f'scene_detect_content(): failed to detect scenes: {e}') from e


@pxt.udf(is_method=True)
def scene_detect_threshold(
    video: pxt.Video,
    *,
    fps: float | None = None,
    threshold: float = 12.0,
    min_scene_len: int = 15,
    fade_bias: float = 0.0,
    add_final_scene: bool = False,
    method: Literal['ceiling', 'floor'] = 'floor',
) -> list[dict]:
    """
    Detect fade-in and fade-out transitions in a video using PySceneDetect's
    [ThresholdDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.threshold_detector.ThresholdDetector).

    ThresholdDetector identifies scenes by detecting when pixel brightness falls below or rises above
    a threshold value, suitable for detecting fade-to-black, fade-to-white, and similar transitions.

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for fade transitions.
        fps: Number of frames to extract per second for analysis. If None or 0, analyzes all frames.
            Lower values process faster but may miss exact transition points.
        threshold: 8-bit intensity value that each pixel value (R, G, and B) must be less than or equal to in order
            to trigger a fade in/out.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.
        fade_bias: Float between -1.0 and +1.0 representing the percentage of timecode skew for the start of a scene
            (-1.0 causing a cut at the fade-to-black, 0.0 in the middle, and +1.0 causing the cut to be right at the
            position where the threshold is passed).
        add_final_scene: Boolean indicating if the video ends on a fade-out to generate an additional scene at this
            timecode.
        method: How to treat threshold when detecting fade events
            - 'ceiling': Fade out happens when frame brightness rises above threshold.
            - 'floor': Fade out happens when frame brightness falls below threshold.


    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect fade-to-black transitions with default parameters:

        >>> tbl.select(tbl.video.scene_detect_threshold()).collect()

        Use a lower threshold to detect darker fades:

        >>> tbl.select(tbl.video.scene_detect_threshold(threshold=8.0)).collect()

        Detect both fade-to-black and fade-to-white using absolute method:

        >>> tbl.select(tbl.video.scene_detect_threshold(method='absolute')).collect()

        Add final scene boundary:

        >>> tbl.select(
        ...     tbl.video.scene_detect_threshold(add_final_scene=True)
        ... ).collect()

        Add fade transitions as a computed column:

        >>> tbl.add_computed_column(
        ...     fade_cuts=tbl.video.scene_detect_threshold(threshold=15.0)
        ... )
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import ThresholdDetector

    method_enum = ThresholdDetector.Method.FLOOR if method == 'floor' else ThresholdDetector.Method.CEILING
    try:
        detector = ThresholdDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
            fade_bias=fade_bias,
            add_final_scene=add_final_scene,
            method=method_enum,
        )
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.Error(f'scene_detect_threshold(): failed to detect scenes: {e}') from e


@pxt.udf(is_method=True)
def scene_detect_histogram(
    video: pxt.Video, *, fps: float | None = None, threshold: float = 0.05, bins: int = 256, min_scene_len: int = 15
) -> list[dict]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [HistogramDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.histogram_detector.HistogramDetector).

    HistogramDetector compares frame histograms on the Y (luminance) channel after YUV conversion.
    It detects scenes based on relative histogram differences and is more robust to gradual lighting
    changes than content-based detection.

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None or 0, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        threshold: Maximum relative difference between 0.0 and 1.0 that the histograms can differ. Histograms are
            calculated on the Y channel after converting the frame to YUV, and normalized based on the number of bins.
            Higher differences imply greater change in content, so larger threshold values are less sensitive to cuts.
            Lower values detect more scenes (more sensitive), higher values detect fewer scenes.
        bins: Number of bins to use for histogram calculation (typically 16-256). More bins provide
            finer granularity but may be more sensitive to noise.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.


    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_histogram()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(tbl.video.scene_detect_histogram(threshold=0.03)).collect()

        Use fewer bins for faster processing:

        >>> tbl.select(tbl.video.scene_detect_histogram(bins=64)).collect()

        Use with a longer minimum scene length:

        >>> tbl.select(tbl.video.scene_detect_histogram(min_scene_len=30)).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(
        ...     scene_cuts=tbl.video.scene_detect_histogram(threshold=0.04)
        ... )
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import HistogramDetector

    try:
        detector = HistogramDetector(threshold=threshold, bins=bins, min_scene_len=min_scene_len)
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.Error(f'scene_detect_histogram(): failed to detect scenes: {e}') from e


@pxt.udf(is_method=True)
def scene_detect_hash(
    video: pxt.Video,
    *,
    fps: float | None = None,
    threshold: float = 0.395,
    size: int = 16,
    lowpass: int = 2,
    min_scene_len: int = 15,
) -> list[dict]:
    """
    Detect scene cuts in a video using PySceneDetect's
    [HashDetector](https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.hash_detector.HashDetector).

    HashDetector uses perceptual hashing for very fast scene detection. It computes a hash of each
    frame at reduced resolution and compares hash distances.

    __Requirements:__

    - `pip install scenedetect`

    Args:
        video: The video to analyze for scene cuts.
        fps: Number of frames to extract per second for analysis. If None, analyzes all frames.
            Lower values process faster but may miss exact scene cuts.
        threshold: Value from 0.0 and 1.0 representing the relative hamming distance between the perceptual hashes of
            adjacent frames. A distance of 0 means the image is the same, and 1 means no correlation. Smaller threshold
            values thus require more correlation, making the detector more sensitive. The Hamming distance is divided
            by size x size before comparing to threshold for normalization.
            Lower values detect more scenes (more sensitive), higher values detect fewer scenes.
        size: Size of square of low frequency data to use for the DCT. Larger values are more precise but slower.
            Common values are 8, 16, or 32.
        lowpass: How much high frequency information to filter from the DCT. A value of 2 means keep lower 1/2 of the
            frequency data, 4 means only keep 1/4, etc. Larger values make the
            detector less sensitive to high-frequency details and noise.
        min_scene_len: Once a cut is detected, this many frames must pass before a new one can be added to the scene
            list.


    Returns:
        A list of dictionaries, one for each detected scene, with the following keys:

        - `start_time` (float): The start time of the scene in seconds.
        - `start_pts` (int): The pts of the start of the scene.
        - `duration` (float): The duration of the scene in seconds.

        The list is ordered chronologically. Returns the full duration of the video if no scenes are detected.

    Examples:
        Detect scene cuts with default parameters:

        >>> tbl.select(tbl.video.scene_detect_hash()).collect()

        Detect more scenes by lowering the threshold:

        >>> tbl.select(tbl.video.scene_detect_hash(threshold=0.3)).collect()

        Use larger hash size for more precision:

        >>> tbl.select(tbl.video.scene_detect_hash(size=32)).collect()

        Use for fast processing with lower frame rate:

        >>> tbl.select(tbl.video.scene_detect_hash(fps=1.0, threshold=0.4)).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(scene_cuts=tbl.video.scene_detect_hash())
    """
    Env.get().require_package('scenedetect')
    from scenedetect.detectors import HashDetector

    try:
        detector = HashDetector(threshold=threshold, size=size, lowpass=lowpass, min_scene_len=min_scene_len)
        return _scene_detect(video, fps, detector)
    except Exception as e:
        raise pxt.Error(f'scene_detect_hash(): failed to detect scenes: {e}') from e


class _SceneDetectFrameInfo(NamedTuple):
    frame_idx: int
    frame_pts: int
    frame_time: float


def _scene_detect(video: str, fps: float, detector: 'SceneDetector') -> list[dict[str, int | float]]:
    from scenedetect import FrameTimecode  # type: ignore[import-untyped]

    with av_utils.VideoFrames(Path(video), fps=fps) as frame_iter:
        video_fps = float(frame_iter.video_framerate)

        scenes: list[dict[str, int | float]] = []
        frame_idx: int | None = None
        start_time: float | None = None  # of current scene
        start_pts: int | None = None  # of current scene

        # in order to determine the cut frame times, we need to record frame times (chronologically) and look them
        # up by index; trying to derive frame times from frame indices isn't possible due to variable frame rates
        frame_info: list[_SceneDetectFrameInfo] = []

        def process_cuts(cuts: list[FrameTimecode]) -> None:
            nonlocal frame_info, start_time, start_pts
            for cut_timecode in cuts:
                cut_frame_idx = cut_timecode.get_frames()
                # we expect cuts to come back in chronological order
                assert cut_frame_idx >= frame_info[0].frame_idx
                info_offset = next((i for i, info in enumerate(frame_info) if info.frame_idx == cut_frame_idx), None)
                assert info_offset is not None  # the cut is at a previously reported frame idx
                info = frame_info[info_offset]
                scenes.append(
                    {'start_time': start_time, 'start_pts': start_pts, 'duration': info.frame_time - start_time}
                )
                start_time = info.frame_time
                start_pts = info.frame_pts
                frame_info = frame_info[info_offset + 1 :]

        for item in frame_iter:
            if start_time is None:
                start_time = item.time
                start_pts = item.pts
            frame_info.append(_SceneDetectFrameInfo(item.frame_idx, item.pts, item.time))
            frame_array = np.array(item.frame.convert('RGB'))
            frame_idx = item.frame_idx
            timecode = FrameTimecode(item.frame_idx, video_fps)
            cuts = detector.process_frame(timecode, frame_array)
            process_cuts(cuts)

        # Post-process to capture any final scene cuts
        if frame_idx is not None:
            final_timecode = FrameTimecode(frame_idx, video_fps)
            final_cuts = detector.post_process(final_timecode)
            process_cuts(final_cuts)

            # if we didn't detect any cuts but the video has content, add the full video as a single scene
            if len(scenes) == 0:
                scenes.append(
                    {
                        'start_time': start_time,
                        'start_pts': start_pts,
                        'duration': frame_info[-1].frame_time - start_time,
                    }
                )

        return scenes


class FrameAttrs(TypedDict):
    index: int
    pts: int
    dts: int | None
    time: float
    is_corrupt: bool
    key_frame: bool
    pict_type: int
    interlaced_frame: bool


class Frame(TypedDict):
    frame: pxt.Image
    frame_attrs: FrameAttrs


@pxt.iterator(unstored_cols=['frame'])
class frame_iterator(pxt.PxtIterator[Frame]):
    """
    Iterator over frames of a video. At most one of `fps`, `num_frames` or `keyframes_only` may be specified. If `fps`
    is specified, then frames will be extracted at the specified rate (frames per second). If `num_frames` is specified,
    then the exact number of frames will be extracted. If neither is specified, then all frames will be extracted.

    __Outputs:__

    One row per extracted frame, with the following columns:

    - `frame` (`pxt.Image`): The extracted video frame
    - `frame_attrs` (`pxt.Json`): A dictionary containing the following attributes (for more information,
        see `pyav`'s documentation on
        [VideoFrame](https://pyav.org/docs/develop/api/video.html#module-av.video.frame) and
        [Frame](https://pyav.org/docs/develop/api/frame.html)):

            * `index` (`int`): The index of the frame in the video stream
            * `pts` (`int | None`): The presentation timestamp of the frame
            * `dts` (`int | None`): The decoding timestamp of the frame
            * `time` (`float | None`): The timestamp of the frame in seconds
            * `is_corrupt` (`bool`): `True` if the frame is corrupt
            * `key_frame` (`bool`): `True` if the frame is a keyframe
            * `pict_type` (`int`): The picture type of the frame
            * `interlaced_frame` (`bool`): `True` if the frame is interlaced

    Args:
        fps: Number of frames to extract per second of video. This may be a fractional value, such as 0.5.
            If omitted, or if greater than the native framerate of the video,
            then the framerate of the video will be used (all frames will be extracted).
        num_frames: Exact number of frames to extract. The frames will be spaced as evenly as possible. If
            `num_frames` is greater than the number of frames in the video, all frames will be extracted.
        keyframes_only: If True, only extract keyframes.

    Examples:
        All these examples assume an existing table `tbl` with a column `video` of type `pxt.Video`.

        Create a view that extracts all frames from all videos:

        >>> pxt.create_view('all_frames', tbl, iterator=frame_iterator(tbl.video))

        Create a view that extracts only keyframes from all videos:

        >>> pxt.create_view(
        ...     'keyframes',
        ...     tbl,
        ...     iterator=frame_iterator(tbl.video, keyframes_only=True),
        ... )

        Create a view that extracts frames from all videos at a rate of 1 frame per second:

        >>> pxt.create_view(
        ...     'one_fps_frames', tbl, iterator=frame_iterator(tbl.video, fps=1.0)
        ... )

        Create a view that extracts exactly 10 frames from each video:

        >>> pxt.create_view(
        ...     'ten_frames', tbl, iterator=frame_iterator(tbl.video, num_frames=10)
        ... )
    """

    # Input parameters
    video_path: Path
    fps: float | None
    num_frames: int | None
    keyframes_only: bool

    # Video info
    container: InputContainer
    video_time_base: Fraction
    video_start_time: float
    video_duration: float | None

    # extraction info
    extraction_step: float | None
    next_extraction_time: float | None

    # state
    pos: int
    video_idx: int
    cur_frame: av.VideoFrame | None

    def __init__(
        self, video: pxt.Video, *, fps: float | None = None, num_frames: int | None = None, keyframes_only: bool = False
    ) -> None:
        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()
        self.video_path = video_path
        self.container = av.open(str(video_path))
        self.fps = fps
        self.num_frames = num_frames
        self.keyframes_only = keyframes_only

        self.video_time_base = self.container.streams.video[0].time_base

        start_time = self.container.streams.video[0].start_time or 0
        self.video_start_time = float(start_time * self.video_time_base)

        duration_pts: int | None = self.container.streams.video[0].duration
        if duration_pts is not None:
            self.video_duration = float(duration_pts * self.video_time_base)
        else:
            # As a backup, try to calculate duration from DURATION metadata field
            metadata = self.container.streams.video[0].metadata
            duration_field = metadata.get('DURATION')  # A string like "00:01:23"
            if duration_field is not None:
                assert isinstance(duration_field, str)
                self.video_duration = pd.to_timedelta(duration_field).total_seconds()
            else:
                # TODO: Anything we can do here? Other methods of determining the duration are expensive and
                #     not so appropriate for an iterator initializer.
                self.video_duration = None

        if self.video_duration is None and self.num_frames is not None:
            raise excs.Error(f'Could not determine duration of video: {video}')

        # If self.fps or self.num_frames is specified, we cannot rely on knowing in advance which frame positions will
        # be needed, since for variable framerate videos we do not know in advance the precise timestamp of each frame.
        # The strategy is: predetermine a list of "extraction times", the idealized timestamps of the frames we want to
        # materialize. As we later iterate through the frames, we will choose the frames that are closest to these
        # idealized timestamps.

        self.pos = 0
        self.video_idx = 0
        if self.num_frames is not None:
            # Divide the video duration into num_frames evenly spaced intervals. The extraction times are the midpoints
            # of those intervals.
            self.extraction_step = (self.video_duration - self.video_start_time) / self.num_frames
            self.next_extraction_time = self.video_start_time + self.extraction_step / 2
        elif self.fps is not None:
            self.extraction_step = 1 / self.fps
            self.next_extraction_time = self.video_start_time
        else:
            self.extraction_step = None
            self.next_extraction_time = None

        _logger.debug(
            f'FrameIterator: path={self.video_path} fps={self.fps} num_frames={self.num_frames} '
            f'keyframes_only={self.keyframes_only}'
        )
        self.cur_frame = self.next_frame()

    def next_frame(self) -> av.VideoFrame | None:
        try:
            return next(self.container.decode(video=0))
        except EOFError:
            return None

    def __next__(self) -> Frame:
        while True:
            if self.cur_frame is None:
                raise StopIteration

            next_frame = self.next_frame()

            if self.keyframes_only and not self.cur_frame.key_frame:
                self.cur_frame = next_frame
                self.video_idx += 1
                continue

            cur_frame_pts = self.cur_frame.pts
            cur_frame_time = float(cur_frame_pts * self.video_time_base)

            if self.extraction_step is not None:
                # We are targeting a specified list of extraction times (because fps or num_frames was specified).
                assert self.next_extraction_time is not None

                if next_frame is None:
                    # cur_frame is the last frame of the video. If it is before the next extraction time, then we
                    # have reached the end of the video.
                    if cur_frame_time < self.next_extraction_time:
                        raise StopIteration
                else:
                    # The extraction time represents the idealized timestamp of the next frame we want to extract.
                    # If next_frame is *closer* to it than cur_frame, then we skip cur_frame.
                    # The following logic handles all three cases:
                    # - next_extraction_time is before cur_frame_time (never skips)
                    # - next_extraction_time is after next_frame_time (always skips)
                    # - next_extraction_time is between cur_frame_time and next_frame_time (depends on which is closer)
                    next_frame_pts = next_frame.pts
                    next_frame_time = float(next_frame_pts * self.video_time_base)
                    if next_frame_time - self.next_extraction_time < self.next_extraction_time - cur_frame_time:
                        self.cur_frame = next_frame
                        self.video_idx += 1
                        continue

            image = self.cur_frame.to_image()
            assert isinstance(image, PIL.Image.Image)
            result: Frame = {
                'frame': image,
                'frame_attrs': {
                    'index': self.video_idx,
                    'pts': cur_frame_pts,
                    'dts': self.cur_frame.dts,
                    'time': float(cur_frame_pts * self.video_time_base),
                    'is_corrupt': self.cur_frame.is_corrupt,
                    'key_frame': self.cur_frame.key_frame,
                    'pict_type': self.cur_frame.pict_type,
                    'interlaced_frame': self.cur_frame.interlaced_frame,
                },
            }

            self.cur_frame = next_frame
            self.video_idx += 1

            self.pos += 1
            if self.extraction_step is not None:
                self.next_extraction_time += self.extraction_step

            return result

    def close(self) -> None:
        self.container.close()

    def seek(self, pos: int, **kwargs: Any) -> None:
        assert next(iter(kwargs.values()), None) is not None

        if self.pos == pos:
            # Nothing to do
            return

        self.pos = pos

        seek_time: float
        if 'pos_msec' in kwargs:
            self.video_idx = kwargs['pos_frame']
            seek_time = kwargs['pos_msec'] / 1000.0
        else:
            assert 'frame_attrs' in kwargs
            self.video_idx = kwargs['frame_attrs']['index']
            seek_time = kwargs['frame_attrs']['time']

        assert isinstance(self.video_idx, int)
        assert isinstance(seek_time, float)

        # Subtlety: The offset passed in to seek() is not the pts, but rather the pts adjusted for the video start time.
        seek_offset = math.floor((seek_time - self.video_start_time) / self.video_time_base)
        self.container.seek(seek_offset, backward=True, stream=self.container.streams.video[0])

        self.cur_frame = self.next_frame()
        while self.cur_frame is not None and float(self.cur_frame.pts * self.video_time_base) < seek_time - 1e-3:
            self.cur_frame = self.next_frame()
        assert self.cur_frame is None or abs(float(self.cur_frame.pts * self.video_time_base) - seek_time) < 1e-3

    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        fps = bound_args.get('fps')
        num_frames = bound_args.get('num_frames')
        keyframes_only = bound_args.get('keyframes_only', False)
        if int(fps is not None) + int(num_frames is not None) + int(keyframes_only) > 1:
            raise excs.Error('At most one of `fps`, `num_frames` or `keyframes_only` may be specified')
        if fps is not None and (not isinstance(fps, (int, float)) or fps <= 0.0):
            raise excs.Error('`fps` must be a positive number')


class LegacyFrame(TypedDict):
    frame: pxt.Image
    frame_idx: int
    pos_msec: float
    pos_frame: int


@pxt.iterator(unstored_cols=['frame'])
class legacy_frame_iterator(pxt.PxtIterator[LegacyFrame]):
    # A retrofitted implementation of the legacy frame iterator interface, with output schema `frame_idx`, `pos_msec`,
    # and `pos_frame` (instead of `frame_attrs`). It wraps the new `frame_iterator` and dervies the legacy outputs from
    # `frame_attrs`.
    underlying: pxt.PxtIterator[Frame]

    def __init__(
        self, video: pxt.Video, *, fps: float | None = None, num_frames: int | None = None, keyframes_only: bool = False
    ) -> None:
        self.underlying = frame_iterator.decorated_callable(  # type: ignore[attr-defined]
            video=video, fps=fps, num_frames=num_frames, keyframes_only=keyframes_only
        )

    def __next__(self) -> LegacyFrame:
        item: Frame = next(self.underlying)
        frame_attrs = item['frame_attrs']
        result: LegacyFrame = {
            'frame': item['frame'],
            # frame_idx == pos, but `pos` has already been incremented in the underlying, since we just called `next()`
            'frame_idx': self.underlying.pos - 1,  # type: ignore[attr-defined]
            'pos_msec': (frame_attrs['time'] - self.underlying.video_start_time) * 1000.0,  # type: ignore[attr-defined]
            'pos_frame': frame_attrs['index'],
        }
        return result

    def close(self) -> None:
        self.underlying.close()

    def seek(self, pos: int, **kwargs: Any) -> None:
        self.underlying.seek(pos, **kwargs)

    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        frame_iterator.decorated_callable.validate(bound_args)  # type: ignore[attr-defined]


class VideoSegment(TypedDict):
    segment_start: float | None
    segment_start_pts: int | None
    segment_end: float | None
    segment_end_pts: int | None
    video_segment: pxt.Video


@pxt.iterator
def video_splitter(
    video: pxt.Video,
    *,
    duration: float | None = None,
    overlap: float | None = None,
    min_segment_duration: float | None = None,
    segment_times: list[float] | None = None,
    mode: Literal['fast', 'accurate'] = 'accurate',
    video_encoder: str | None = None,
    video_encoder_args: dict[str, Any] | None = None,
) -> Iterator[VideoSegment]:
    """
    Iterator over segments of a video file, which is split into segments. The segments are specified either via a
    fixed duration or a list of split points.

    Args:
        duration: Video segment duration in seconds
        overlap: Overlap between consecutive segments in seconds. Only available for `mode='fast'`.
        min_segment_duration: Drop the last segment if it is smaller than min_segment_duration.
        segment_times: List of timestamps (in seconds) in video where segments should be split. Note that these are not
            segment durations. If all segment times are less than the duration of the video, produces exactly
            `len(segment_times) + 1` segments. An argument of `[]` will produce a single segment containing the
            entire video.
        mode: Segmentation mode:

            - `'fast'`: Quick segmentation using stream copy (splits only at keyframes, approximate durations)
            - `'accurate'`: Precise segmentation with re-encoding (exact durations, slower)
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
            Only available for `mode='accurate'`.
        video_encoder_args: Additional arguments to pass to the video encoder. Only available for `mode='accurate'`.

    Examples:
        All these examples assume an existing table `tbl` with a column `video` of type `pxt.Video`.

        Create a view that splits each video into 10-second segments:

        >>> pxt.create_view(
        ...     'ten_second_segments',
        ...     tbl,
        ...     iterator=video_splitter(tbl.video, duration=10.0),
        ... )

        Create a view that splits each video into segments at specified fixed times:

        >>> split_times = [5.0, 15.0, 30.0]
        >>> pxt.create_view(
        ...     'custom_segments',
        ...     tbl,
        ...     iterator=video_splitter(tbl.video, segment_times=split_times),
        ... )

        Create a view that splits each video into segments at times specified by a column `split_times` of type
        `pxt.Json`, containing a list of timestamps in seconds:

        >>> pxt.create_view(
        ...     'custom_segments',
        ...     tbl,
        ...     iterator=video_splitter(tbl.video, segment_times=tbl.split_times),
        ... )
    """
    # Input parameters
    assert (duration is None) != (segment_times is None)
    if duration is not None:
        assert duration > 0.0
        assert min_segment_duration is None or duration >= min_segment_duration
        assert overlap is None or overlap < duration

    video_path = Path(video)
    assert video_path.exists() and video_path.is_file()

    overlap = overlap or 0.0
    min_segment_duration = min_segment_duration or 0.0

    with av.open(video) as container:
        video_time_base = container.streams.video[0].time_base

    if segment_times is not None and len(segment_times) == 0:
        # Return the entire video as a single segment
        with av.open(video) as container:
            video_stream = container.streams.video[0]
            start_ts = (
                float(video_stream.start_time * video_stream.time_base)
                if video_stream.start_time is not None and video_stream.time_base is not None
                else 0.0
            )
            end_pts = (
                video_stream.start_time + video_stream.duration
                if video_stream.start_time is not None and video_stream.duration is not None
                else None
            )
            end_ts = (
                float(end_pts * video_stream.time_base)
                if end_pts is not None and video_stream.time_base is not None
                else 0.0
            )
            yield {
                'segment_start': start_ts,
                'segment_start_pts': video_stream.start_time,
                'segment_end': end_ts,
                'segment_end_pts': end_pts,
                'video_segment': video,
            }

    elif mode == 'fast':
        segment_path: str = ''
        try:
            start_time = 0.0
            start_pts = 0
            segment_idx = 0
            while True:
                target_duration: float | None
                if duration is not None:
                    target_duration = duration
                elif segment_times is not None and segment_idx < len(segment_times):
                    target_duration = segment_times[segment_idx] - start_time
                else:
                    target_duration = None  # the rest of the video

                segment_path = str(TempStore.create_path(extension='.mp4'))
                cmd = av_utils.ffmpeg_clip_cmd(video, segment_path, start_time, target_duration)
                _ = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # use the actual duration
                actual_duration = av_utils.get_video_duration(segment_path)
                if actual_duration - overlap == 0.0 or actual_duration < min_segment_duration:
                    # we're done
                    Path(segment_path).unlink()
                    return

                segment_end = start_time + actual_duration
                segment_end_pts = start_pts + round(actual_duration / video_time_base)
                result: VideoSegment = {
                    'segment_start': start_time,
                    'segment_start_pts': start_pts,
                    'segment_end': segment_end,
                    'segment_end_pts': segment_end_pts,
                    'video_segment': segment_path,
                }
                yield result

                start_time = segment_end - overlap
                start_pts = segment_end_pts - round(overlap / video_time_base)

                segment_idx += 1
                if segment_times is not None and segment_idx > len(segment_times):
                    # We've created all segments including the final segment after the last segment_time
                    break

        except subprocess.CalledProcessError as e:
            if segment_path and Path(segment_path).exists():
                Path(segment_path).unlink()
            error_msg = f'ffmpeg failed with return code {e.returncode}'
            if e.stderr:
                error_msg += f': {e.stderr.strip()}'
            raise pxt.Error(error_msg) from e

    else:  # mode == 'accurate'
        base_path = TempStore.create_path(extension='')
        # Use ffmpeg -f segment for accurate segmentation with re-encoding
        output_pattern = f'{base_path}_segment_%04d.mp4'
        cmd = av_utils.ffmpeg_segment_cmd(
            video,
            output_pattern,
            segment_duration=duration,
            segment_times=segment_times,
            video_encoder=video_encoder,
            video_encoder_args=video_encoder_args,
        )
        try:
            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_paths = sorted(glob.glob(f'{base_path}_segment_*.mp4'))
            # TODO: is this actually an error?
            # if len(output_paths) == 0:
            #     stderr_output = result.stderr.strip() if result.stderr is not None else ''
            #     raise pxt.Error(
            #         f'ffmpeg failed to create output files for commandline: {" ".join(cmd)}\n{stderr_output}'
            #     )
            start_time = 0.0
            start_pts = 0
            for segment_path in output_paths:
                actual_duration = av_utils.get_video_duration(segment_path)
                if actual_duration < min_segment_duration:
                    Path(segment_path).unlink()
                    return

                result = {
                    'segment_start': start_time,
                    'segment_start_pts': start_pts,
                    'segment_end': start_time + actual_duration,
                    'segment_end_pts': start_pts + round(actual_duration / video_time_base),
                    'video_segment': segment_path,
                }
                yield result

                start_time += actual_duration
                start_pts += round(actual_duration / video_time_base)

        except subprocess.CalledProcessError as e:
            error_msg = f'ffmpeg failed with return code {e.returncode}'
            if e.stderr:
                error_msg += f': {e.stderr.strip()}'
            raise pxt.Error(error_msg) from e


@video_splitter.validate
def _(bound_args: dict[str, Any]) -> None:
    Env.get().require_binary('ffmpeg')

    duration = bound_args.get('duration')
    segment_times = bound_args.get('segment_times')
    overlap = bound_args.get('overlap')
    min_segment_duration = bound_args.get('min_segment_duration')
    mode = bound_args.get('mode', 'accurate')
    video_encoder = bound_args.get('video_encoder')
    video_encoder_args = bound_args.get('video_encoder_args')

    if 'duration' in bound_args and 'segment_times' in bound_args and duration is None and segment_times is None:
        # Both 'duration' and 'segment_times' are specified as constants, and they're both `None`
        raise excs.Error('Must specify either duration or segment_times')
    if duration is not None and segment_times is not None:
        raise excs.Error('duration and segment_times cannot both be specified')
    if segment_times is not None and overlap is not None:
        raise excs.Error('overlap cannot be specified with segment_times')
    if duration is not None and isinstance(duration, (int, float)):
        if duration <= 0.0:
            raise excs.Error(f'duration must be a positive number: {duration}')
        if (
            min_segment_duration is not None
            and isinstance(min_segment_duration, (int, float))
            and duration < min_segment_duration
        ):
            raise excs.Error(f'duration must be at least min_segment_duration: {duration} < {min_segment_duration}')
        if overlap is not None and isinstance(overlap, (int, float)) and overlap >= duration:
            raise excs.Error(f'overlap must be less than duration: {overlap} >= {duration}')
    if mode == 'accurate' and overlap is not None:
        raise excs.Error("Cannot specify overlap for mode='accurate'")
    if mode == 'fast':
        if video_encoder is not None:
            raise excs.Error("Cannot specify video_encoder for mode='fast'")
        if video_encoder_args is not None:
            raise excs.Error("Cannot specify video_encoder_args for mode='fast'")


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
