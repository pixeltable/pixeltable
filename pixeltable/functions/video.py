"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `VideoType`.
"""

import logging
import pathlib
import subprocess
from typing import Literal, NoReturn

import av
import av.stream
import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

_logger = logging.getLogger('pixeltable')
_format_defaults: dict[str, tuple[str, str]] = {  # format -> (codec, ext)
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
    if format not in _format_defaults:
        raise ValueError(f'extract_audio(): unsupported audio format: {format}')
    default_codec, ext = _format_defaults[format]

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

        >>> tbl.select(tbl.video.extract_frame(tbl.video.get_metadata().streams[0].duration_seconds - 0.1)).collect()
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
    video: pxt.Video, *, start_time: float, end_time: float | None = None, duration: float | None = None
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

    video_duration = av_utils.get_video_duration(video)
    if video_duration is not None and start_time > video_duration:
        return None

    output_path = str(TempStore.create_path(extension='.mp4'))

    if end_time is not None:
        duration = end_time - start_time
    cmd = av_utils.ffmpeg_clip_cmd(str(video), output_path, start_time, duration)

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
def segment_video(video: pxt.Video, *, duration: float) -> list[str]:
    """
    Split a video into fixed-size segments.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        video: Input video file to segment
        duration: Approximate duration of each segment (in seconds).

    Returns:
        List of file paths for the generated video segments.

    Raises:
        pxt.Error: If the video is missing timing information.

    Examples:
        Split a video at 1 minute intervals

        >>> tbl.select(segment_paths=tbl.video.segment_video(duration=60)).collect()

        Split video into two parts at the midpoint:

        >>> duration = tbl.video.get_duration()
        >>> tbl.select(segment_paths=tbl.video.segment_video(duration=duration / 2 + 1)).collect()
    """
    Env.get().require_binary('ffmpeg')
    if duration <= 0:
        raise pxt.Error(f'duration must be positive, got {duration}')

    base_path = TempStore.create_path(extension='')

    # we extract consecutive clips instead of running ffmpeg -f segment, which is inexplicably much slower
    start_time = 0.0
    result: list[str] = []
    try:
        while True:
            segment_path = f'{base_path}_segment_{len(result)}.mp4'
            cmd = av_utils.ffmpeg_clip_cmd(str(video), segment_path, start_time, duration)

            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
            segment_duration = av_utils.get_video_duration(segment_path)
            if segment_duration == 0.0:
                # we're done
                pathlib.Path(segment_path).unlink()
                return result
            result.append(segment_path)
            start_time += segment_duration  # use the actual segment duration here, it won't match duration exactly

        return result

    except subprocess.CalledProcessError as e:
        # clean up partial results
        for segment_path in result:
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


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
