"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `VideoType`.
"""

import glob
import logging
import pathlib
import subprocess
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NoReturn

import av
import av.stream
import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

if TYPE_CHECKING:
    from scenedetect.detectors import SceneDetector  # type: ignore[import-untyped]

_logger = logging.getLogger('pixeltable')


@pxt.uda(requires_order_by=True)
class make_video(pxt.Aggregator):
    """
    Aggregator that creates a video from a sequence of images, using the default video encoder and yuv420p pixel format.

    Follows https://pyav.org/docs/develop/cookbook/numpy.html#generating-video

    TODO: provide parameters for video_encoder and pix_fmt

    Args:
        fps: Frames per second for the output video.

    Returns:

    - The created video.

    Examples:
        Create a video from frames extracted using `FrameIterator`:

        >>> import pixeltable as pxt
        >>> from pixeltable.functions.video import make_video
        >>> from pixeltable.iterators import FrameIterator
        >>>
        >>> # Create base table for videos
        >>> videos_table = pxt.create_table('videos', {'video': pxt.Video})
        >>>
        >>> # Create view to extract frames
        >>> frames_view = pxt.create_view(
        ...     'video_frames',
        ...     videos_table,
        ...     iterator=FrameIterator.create(video=videos_table.video, fps=1)
        ... )
        >>>
        >>> # Reconstruct video from frames
        >>> frames_view.group_by(videos_table).select(
        ...     make_video(frames_view.pos, frames_view.frame)
        ... ).show()

        Apply transformations to frames before creating a video:

        >>> # Create video from transformed frames
        >>> frames_view.group_by(videos_table).select(
        ...     make_video(frames_view.pos, frames_view.frame.rotate(30))
        ... ).show()

        Compare multiple processed versions side-by-side:

        >>> frames_view.group_by(videos_table).select(
        ...     make_video(frames_view.pos, frames_view.frame),
        ...     make_video(frames_view.pos, frames_view.frame.rotate(30))
        ... ).show()
    """

    container: av.container.OutputContainer | None
    stream: av.video.stream.VideoStream | None
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


def _handle_ffmpeg_error(e: subprocess.CalledProcessError) -> NoReturn:
    error_msg = f'ffmpeg failed with return code {e.returncode}'
    if e.stderr is not None:
        error_msg += f':\n{e.stderr.strip()}'
    raise pxt.Error(error_msg) from e


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

    output_path = str(TempStore.create_path(extension='.mp4'))

    if end_time is not None:
        duration = end_time - start_time
    cmd = av_utils.ffmpeg_clip_cmd(
        str(video),
        output_path,
        start_time,
        duration,
        fast=(mode == 'fast'),
        video_encoder=video_encoder,
        video_encoder_args=video_encoder_args,
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_file = pathlib.Path(output_path)
        if not output_file.exists() or output_file.stat().st_size == 0:
            stderr_output = result.stderr.strip() if result.stderr is not None else ''
            raise pxt.Error(f'ffmpeg failed to create output file for commandline: {" ".join(cmd)}\n{stderr_output}')
        return output_path
    except subprocess.CalledProcessError as e:
        _handle_ffmpeg_error(e)


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
        ...         video_encoder_args={'crf': 23, 'preset': 'slow'}
        ...     )
        ... ).collect()

        Split video into two parts at the midpoint:

        >>> duration = tbl.video.get_duration()
        >>> tbl.select(
        ...     segment_paths=tbl.video.segment_video(
        ...         segment_times=[duration / 2]
        ...     )
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
            _handle_ffmpeg_error(e)

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
                    pathlib.Path(segment_path).unlink()
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
                pathlib.Path(segment_path).unlink()
            _handle_ffmpeg_error(e)


@pxt.udf(is_method=True)
def concat_videos(videos: list[pxt.Video]) -> pxt.Video:
    """
    Merge multiple videos into a single video.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        videos: List of videos to merge.

    Returns:
        A new video containing the merged videos.
    """
    Env.get().require_binary('ffmpeg')
    if len(videos) == 0:
        raise pxt.Error('concat_videos(): empty argument list')

    # Check that all videos have the same resolution
    resolutions: list[tuple[int, int]] = []
    for video in videos:
        metadata = av_utils.get_metadata(str(video))
        video_stream = next((stream for stream in metadata['streams'] if stream['type'] == 'video'), None)
        if video_stream is None:
            raise pxt.Error(f'concat_videos(): file {video!r} has no video stream')
        resolutions.append((video_stream['width'], video_stream['height']))

    # check for divergence
    x0, y0 = resolutions[0]
    for i, (x, y) in enumerate(resolutions[1:], start=1):
        if (x0, y0) != (x, y):
            raise pxt.Error(
                f'concat_videos(): requires that all videos have the same resolution, but:'
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
        _logger.debug(f'concat_videos(): {" ".join(cmd)}')
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

        video_encoder = Env.get().default_video_encoder
        if video_encoder is not None:
            cmd.extend(['-c:v', video_encoder])
        if all_have_audio:
            cmd.extend(['-c:a', 'aac'])
        cmd.extend(['-pix_fmt', 'yuv420p', str(output_path)])

        _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return str(output_path)

    except subprocess.CalledProcessError as e:
        _handle_ffmpeg_error(e)
    finally:
        filelist_path.unlink()


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
        ...         tbl.music_track,
        ...         video_start_time=5.0,
        ...         audio_start_time=5.0
        ...     )
        ... ).collect()

        Use a 10-second clip from the middle of both files:

        >>> tbl.select(
        ...     tbl.video.with_audio(
        ...         tbl.music_track,
        ...         video_start_time=30.0,
        ...         video_duration=10.0,
        ...         audio_start_time=15.0,
        ...         audio_duration=10.0
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

    cmd = ['ffmpeg']
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
            '-loglevel',
            'error',  # only show errors
            output_path,
        ]
    )

    _logger.debug(f'with_audio(): {" ".join(cmd)}')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_file = pathlib.Path(output_path)
        if not output_file.exists() or output_file.stat().st_size == 0:
            stderr_output = result.stderr.strip() if result.stderr is not None else ''
            raise pxt.Error(f'ffmpeg failed to create output file for commandline: {" ".join(cmd)}\n{stderr_output}')
        return output_path
    except subprocess.CalledProcessError as e:
        _handle_ffmpeg_error(e)


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
        ...         vertical_margin=70
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
        ...         box_border=[20, 10]
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

    cmd = [
        'ffmpeg',
        '-i',
        str(video),
        '-vf',
        'drawtext=' + ':'.join(drawtext_params),
        '-c:a',
        'copy',  # Copy audio stream unchanged
        '-loglevel',
        'error',  # Only show errors
        output_path,
    ]
    _logger.debug(f'overlay_text(): {" ".join(cmd)}')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_file = pathlib.Path(output_path)
        if not output_file.exists() or output_file.stat().st_size == 0:
            stderr_output = result.stderr.strip() if result.stderr is not None else ''
            raise pxt.Error(f'ffmpeg failed to create output file for commandline: {" ".join(cmd)}\n{stderr_output}')
        return output_path
    except subprocess.CalledProcessError as e:
        _handle_ffmpeg_error(e)


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
        if pathlib.Path(font).exists():
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

        >>> tbl.select(tbl.video.scene_detect_adaptive(adaptive_threshold=1.5)).collect()

        Use luminance-only detection with a longer minimum scene length:

        >>> tbl.select(
        ...     tbl.video.scene_detect_adaptive(
        ...         luma_only=True,
        ...         min_scene_len=30
        ...     )
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
        ...         delta_edges=1.0,
        ...         delta_hue=0.5,
        ...         delta_sat=0.5
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
        threshold: 8-bit intensity value that each pixel value (R, G, and B) must be <= to in order to trigger a fade
            in/out.
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
        ...     tbl.video.scene_detect_threshold(
        ...         add_final_scene=True
        ...     )
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

        >>> tbl.select(
        ...     tbl.video.scene_detect_histogram(
        ...         min_scene_len=30
        ...     )
        ... ).collect()

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

        >>> tbl.select(
        ...     tbl.video.scene_detect_hash(
        ...         fps=1.0,
        ...         threshold=0.4
        ...     )
        ... ).collect()

        Add scene cuts as a computed column:

        >>> tbl.add_computed_column(
        ...     scene_cuts=tbl.video.scene_detect_hash()
        ... )
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

    with av_utils.VideoFrames(pathlib.Path(video), fps=fps) as frame_iter:
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


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
