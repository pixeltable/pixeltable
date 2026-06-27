import glob
import logging
import subprocess
from pathlib import Path
from typing import Any, Literal

import av
import av.container
import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable.env import Env
from pixeltable.functions import util
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

from .iterators import VideoSegment

_logger = logging.getLogger(__name__)


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
        if self.container is None or self.stream is None:
            # update() was only ever called with null frames (or not at all), so there is nothing to assemble
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, 'make_video(): no frames to assemble into a video'
            )
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        return str(self.out_file)


@pxt.udf(is_method=True)
def extract_audio(
    video_path: pxt.Video, stream_idx: int = 0, format: str = 'wav', codec: str | None = None
) -> pxt.Audio | None:
    """
    Extract an audio stream from a video.

    Args:
        stream_idx: Index of the audio stream to extract.
        format: The target audio format. (`'wav'`, `'mp3'`, `'flac'`).
        codec: The codec to use for the audio stream. If not provided, a default codec will be used.

    Returns:
        The extracted audio, or None if the video has no audio stream at `stream_idx`.

    Examples:
        Add a computed column to a table `tbl` that extracts audio from an existing column `video_col`:

        >>> tbl.add_computed_column(
        ...     extracted_audio=tbl.video_col.extract_audio(format='flac')
        ... )
    """
    if format not in av_utils.AUDIO_FORMATS:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'extract_audio(): unsupported audio format: {format}')
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
def get_metadata(video: pxt.Video) -> util.ContainerMetadata:
    """
    Gets various metadata associated with a video file and returns it as
    a [`ContainerMetadata`][pixeltable.functions.ContainerMetadata] dictionary.

    Args:
        video: The video for which to get metadata.

    Returns:
        A [`ContainerMetadata`][pixeltable.functions.ContainerMetadata] with typical structure:

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
    return util.get_metadata(video)


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
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'timestamp must be non-negative, got {timestamp}')

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
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'extract_frame(): failed to extract frame: {e}'
        ) from e


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
        mode: Clip mode:

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
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'start_time must be non-negative, got {start_time}')
    if end_time is not None and end_time <= start_time:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'end_time ({end_time}) must be greater than start_time ({start_time})'
        )
    if duration is not None and duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'duration must be positive, got {duration}')
    if end_time is not None and duration is not None:
        raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, 'end_time and duration cannot both be specified')
    if mode == 'fast':
        if video_encoder is not None:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, "video_encoder is not supported for mode='fast'"
            )
        if video_encoder_args is not None:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, "video_encoder_args is not supported for mode='fast'"
            )

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
) -> list[VideoSegment]:
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
        A list of [`VideoSegment`][pixeltable.functions.video.VideoSegment] records, one per generated segment.

    Raises:
        pxt.Error: If the video is missing timing information.

    Examples:
        Split a video at 1 minute intervals using fast mode:

        >>> tbl.select(
        ...     segments=tbl.video.segment_video(duration=60, mode='fast')
        ... ).collect()

        Split video into exact 10-second segments with default accurate mode, using the libx264 encoder with a CRF of 23
        and slow preset (for smaller output files):

        >>> tbl.select(
        ...     segments=tbl.video.segment_video(
        ...         duration=10,
        ...         video_encoder='libx264',
        ...         video_encoder_args={'crf': 23, 'preset': 'slow'},
        ...     )
        ... ).collect()

        Split video into two parts at the midpoint:

        >>> duration = tbl.video.get_duration()
        >>> tbl.select(
        ...     segments=tbl.video.segment_video(segment_times=[duration / 2])
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if duration is not None and segment_times is not None:
        raise pxt.RequestError(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, 'duration and segment_times cannot both be specified'
        )
    if duration is not None and duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'duration must be positive, got {duration}')
    if segment_times is not None and len(segment_times) == 0:
        raise pxt.RequestError(pxt.ErrorCode.MISSING_REQUIRED, 'segment_times cannot be empty')
    if mode == 'fast':
        if video_encoder is not None:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, "video_encoder is not supported for mode='fast'"
            )
        if video_encoder_args is not None:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, "video_encoder_args is not supported for mode='fast'"
            )

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
            return _paths_to_segments(str(video), output_paths)

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
                if segment_duration is None:
                    # a generated segment whose duration we cannot read
                    Path(segment_path).unlink()
                    for p in output_paths:
                        Path(p).unlink()
                    raise pxt.RequestError(
                        pxt.ErrorCode.INVALID_DATA_FORMAT,
                        f'segment_video(): could not determine duration of a generated segment from {video!r}',
                    )
                if segment_duration == 0.0:
                    # we're done
                    Path(segment_path).unlink()
                    return _paths_to_segments(str(video), output_paths)
                output_paths.append(segment_path)
                start_time += segment_duration  # use the actual segment duration here, it won't match duration exactly

                segment_idx += 1
                if segment_times is not None and segment_idx > len(segment_times):
                    break

            return _paths_to_segments(str(video), output_paths)

        except subprocess.CalledProcessError as e:
            # clean up partial results
            for segment_path in output_paths:
                Path(segment_path).unlink()
            av_utils.handle_ffmpeg_error(e)


def _paths_to_segments(video: str, paths: list[str]) -> list[VideoSegment]:
    """Build VideoSegment records (matching video_splitter's output) from segment files cut from `video`."""
    with av.open(video) as container:
        video_time_base = container.streams.video[0].time_base
    segments: list[VideoSegment] = []
    start_time = 0.0
    start_pts = 0
    for path in paths:
        duration = av_utils.get_video_duration(path)
        if duration is None:
            raise pxt.RequestError(
                pxt.ErrorCode.INVALID_DATA_FORMAT,
                f'segment_video(): could not determine duration of generated segment {path!r}',
            )
        segment_end = start_time + duration
        segment_end_pts = start_pts + round(duration / video_time_base)
        segment: VideoSegment = {
            'segment_start': start_time,
            'segment_start_pts': start_pts,
            'segment_end': segment_end,
            'segment_end_pts': segment_end_pts,
            'video_segment': path,
        }
        segments.append(segment)
        start_time = segment_end
        start_pts = segment_end_pts
    return segments


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
        metadata = util.get_metadata(str(video))
        video_stream = next((stream for stream in metadata['streams'] if stream['type'] == 'video'), None)
        if video_stream is None:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION, f'{error_prefix}: file {video!r} has no video stream'
            )
        resolutions.append((video_stream['width'], video_stream['height']))

    # check for divergence
    x0, y0 = resolutions[0]
    for i, (x, y) in enumerate(resolutions[1:], start=1):
        if (x0, y0) != (x, y):
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                f'{error_prefix}: requires that all videos have the same resolution, but:'
                f'\n  video 0 ({videos[0]!r}): {x0}x{y0}'
                f'\n  video {i} ({videos[i]!r}): {x}x{y}.',
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
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'video_start_time must be non-negative, got {video_start_time}'
        )
    if audio_start_time < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'audio_start_time must be non-negative, got {audio_start_time}'
        )
    if video_duration is not None and video_duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'video_duration must be positive, got {video_duration}')
    if audio_duration is not None and audio_duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'audio_duration must be positive, got {audio_duration}')

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
    mix_duration: Literal['first', 'longest', 'shortest'] = 'longest',
    normalize: bool = False,
    dropout_transition: float = 2.0,
    align_to_video: Literal['none', 'trim', 'pad'] = 'trim',
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
        mix_duration: Controls which input determines the length of the mixed audio stream.

            - `"longest"`: the mix runs until the longer of the two audio inputs ends. Use
              this when the added audio (e.g. a music bed) is longer than the video's original audio.
            - `"first"`: the mix ends when the video's original audio track ends. Useful when the
              original audio and video streams are the same length.
            - `"shortest"`: the mix ends when the shorter input ends, truncating whichever track is longer.
        normalize: If `True`, ffmpeg scales the mixed output to prevent clipping by dividing each track's
            contribution by the number of inputs. Defaults to `False` so that `audio_volume` and
            `original_volume` mean what they say; flip on if you are not setting volumes explicitly and
            want automatic clip protection.
        dropout_transition: Duration in seconds over which a track's contribution fades to zero after
            it ends, preventing audible clicks at hard boundaries. Defaults to 2.0 seconds, matching
            ffmpeg's own default. Set to 0.0 to disable. Relevant whenever one input ends before
            the mixed output ends, regardless of which `mix_duration` mode is selected.
        align_to_video: Post-mix adjustment to align the output audio stream with the video stream duration.
            Applied after `amix`, so it is independent of `mix_duration`.

            - `"trim"`: if the mixed audio is longer than the video stream, truncate it to
              match. Pairs naturally with `mix_duration="longest"` for music-bed workflows.
            - `"none"`: no adjustment; output audio duration is whatever `amix` produces.
            - `"pad"`: if the mixed audio is shorter than the video stream, extend it with silence. Also
              trims to the video duration if the mix runs long.
        video_encoder: Video encoder to use. If not specified, uses the default encoder.
        video_encoder_args: Additional arguments to pass to the video encoder.

    Returns:
        A new video with both audio tracks mixed together.

    Examples:
        Add background music at 30% volume. With the defaults, this lays the music as a bed under the
        video: `mix_duration="longest"` keeps the music playing past the end of the original audio,
        `align_to_video="trim"` caps the result at the video stream duration, and `normalize=False`
        means the `audio_volume=0.3` setting is taken at face value rather than halved by `amix`:

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

        Pad a short ambient track with silence so the audio stream matches the full video length:

        >>> tbl.select(
        ...     tbl.video.mix_audio(
        ...         tbl.ambient, audio_volume=0.6, align_to_video='pad'
        ...     )
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')
    if audio_volume < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'audio_volume must be non-negative, got {audio_volume}')
    if original_volume < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'original_volume must be non-negative, got {original_volume}'
        )
    if audio_start_time < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'audio_start_time must be non-negative, got {audio_start_time}'
        )
    if dropout_transition < 0:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'dropout_transition must be non-negative, got {dropout_transition}'
        )
    if mix_duration not in ('first', 'longest', 'shortest'):
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f"mix_duration must be one of 'first', 'longest', 'shortest', got {mix_duration!r}",
        )
    if align_to_video not in ('none', 'trim', 'pad'):
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT,
            f"align_to_video must be one of 'none', 'trim', 'pad', got {align_to_video!r}",
        )
    if not av_utils.has_audio_stream(str(video)):
        raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, 'mix_audio() requires a video with an audio stream')

    align_label = '[aout]'
    if align_to_video != 'none':
        video_duration = av_utils.get_video_duration(str(video))
        if video_duration is None:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                f'align_to_video={align_to_video!r} requires a video with a known duration',
            )
        align_op = 'apad,atrim' if align_to_video == 'pad' else 'atrim'
        align_chain = f';[aout]{align_op}=duration={video_duration},asetpts=N/SR/TB[aligned]'
        align_label = '[aligned]'
    else:
        align_chain = ''

    output_path = str(TempStore.create_path(extension='.mp4'))

    delay_filter = ''
    if audio_start_time > 0:
        delay_ms = int(audio_start_time * 1000)
        delay_filter = f'adelay={delay_ms}|{delay_ms},'
    amix = (
        f'amix=inputs=2:duration={mix_duration}'
        f':dropout_transition={dropout_transition}'
        f':normalize={1 if normalize else 0}'
    )
    filter_complex = (
        f'[0:a]volume={original_volume}[a0];'
        f'[1:a]{delay_filter}volume={audio_volume}[a1];'
        f'[a0][a1]{amix}[aout]'
        f'{align_chain}'
    )
    cmd = ['-i', str(video), '-i', str(audio), '-filter_complex', filter_complex, '-map', '0:v:0', '-map', align_label]
    return av_utils.run_ffmpeg_cmdline(
        cmd, output_path, encode_video=True, video_encoder=video_encoder, video_encoder_args=video_encoder_args
    )


__all__ = local_public_names(__name__)
