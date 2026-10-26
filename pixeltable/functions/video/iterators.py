import glob
import logging
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict

import av
import PIL.Image
from av.container import InputContainer

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

_logger = logging.getLogger(__name__)


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
    extraction_step: float | None  # seconds between consecutive extraction times
    start_extraction_time: float | None  # extraction time for pos 0, in seconds

    # state
    pos: int
    video_idx: int
    frames: Iterator[av.VideoFrame]
    current_frame: av.VideoFrame | None  # frame at video_idx, the next one to emit or skip; None after the last frame
    last_frame: av.VideoFrame | None  # latest frame from self.frames; the final frame once current_frame is None
    last_frame_interval: int | None  # pts between the last two frames decoded from self.frames
    at_seek_target: bool  # set by seek(); indicates that current_frame stores the frame of the seek target row

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

        self.video_duration = av_utils.get_stream_duration(self.container.streams.video[0])
        # TODO: determine the duration when the headers don't record it, without scanning the whole file
        if self.video_duration is None and self.num_frames is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Could not determine duration of video: {video}'
            )

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
            self.extraction_step = self.video_duration / self.num_frames
            self.start_extraction_time = self.video_start_time + self.extraction_step / 2
        elif self.fps is not None:
            self.extraction_step = 1 / self.fps
            self.start_extraction_time = self.video_start_time
        else:
            self.extraction_step = None
            self.start_extraction_time = None

        _logger.debug(
            f'FrameIterator: path={self.video_path} fps={self.fps} num_frames={self.num_frames} '
            f'keyframes_only={self.keyframes_only}'
        )
        self._start_decoding()
        self.at_seek_target = False

    def _start_decoding(self) -> None:
        """Start a new decode generator at the container's current position and decode its first frame."""
        self.frames = self.container.decode(video=0)
        self.last_frame = None
        self.last_frame_interval = None
        self._decode_next()

    def _decode_next(self) -> None:
        self.current_frame = next(self.frames, None)
        if self.current_frame is not None:
            if self.last_frame is not None:
                self.last_frame_interval = self.current_frame.pts - self.last_frame.pts
            self.last_frame = self.current_frame

    def video_end_time(self) -> float:
        """Return the end of the video, in seconds. Requires that last_frame is the final frame."""
        assert self.current_frame is None and self.last_frame is not None
        # FFmpeg reports 0 for an unknown frame duration
        if self.last_frame.duration > 0:
            return float((self.last_frame.pts + self.last_frame.duration) * self.video_time_base)
        last_frame_start = float(self.last_frame.pts * self.video_time_base)
        # the header duration can be an estimate that ends before the last frame starts (seen in .mpg files)
        if self.video_duration is not None and self.video_start_time + self.video_duration > last_frame_start:
            return self.video_start_time + self.video_duration
        # nothing records the end: assume the last frame lasts as long as the interval before it
        if self.last_frame_interval is not None:
            return float((self.last_frame.pts + self.last_frame_interval) * self.video_time_base)
        return last_frame_start

    def __next__(self) -> Frame:
        if self.at_seek_target:
            self.at_seek_target = False
            current_frame = self.current_frame
            assert current_frame is not None
            self._decode_next()
            return self._emit_frame(current_frame)
        # the extraction times span the header duration, which can end before the last frame (seen in .mpg files)
        if self.num_frames is not None and self.pos == self.num_frames:
            raise StopIteration
        while True:
            current_frame = self.current_frame
            if current_frame is None:
                raise StopIteration
            self._decode_next()
            next_frame = self.current_frame

            if self.keyframes_only and not current_frame.key_frame:
                self.video_idx += 1
                continue

            if self.extraction_step is not None:
                # We are targeting a specified list of extraction times (because fps or num_frames was specified).
                assert self.start_extraction_time is not None
                # computed from pos, so rounding errors don't accumulate
                extraction_time = self.start_extraction_time + self.pos * self.extraction_step

                if next_frame is None:
                    # current_frame is shown until the end of the video; 1e-6 absorbs rounding in extraction_time;
                    # this only runs when we reach the end of the video, and caching video_end_time isn't worth it
                    if extraction_time >= self.video_end_time() - 1e-6:
                        raise StopIteration
                else:
                    # The extraction time represents the idealized timestamp of the next frame we want to extract.
                    # If next_frame is *closer* to it than current_frame, then we skip current_frame.
                    # The following logic handles all three cases:
                    # - extraction_time is before current_frame_time (never skips)
                    # - extraction_time is after next_frame_time (always skips)
                    # - extraction_time is between current_frame_time and next_frame_time (depends on which is closer)
                    current_frame_time = float(current_frame.pts * self.video_time_base)
                    next_frame_time = float(next_frame.pts * self.video_time_base)
                    if next_frame_time - extraction_time < extraction_time - current_frame_time:
                        self.video_idx += 1
                        continue

            return self._emit_frame(current_frame)

    def _emit_frame(self, frame: av.VideoFrame) -> Frame:
        """Return the output for frame and advance pos and video_idx."""
        image = frame.to_image()
        assert isinstance(image, PIL.Image.Image)
        result: Frame = {
            'frame': image,
            'frame_attrs': {
                'index': self.video_idx,
                'pts': frame.pts,
                'dts': frame.dts,
                'time': float(frame.pts * self.video_time_base),
                'is_corrupt': frame.is_corrupt,
                'key_frame': frame.key_frame,
                'pict_type': frame.pict_type,
                'interlaced_frame': frame.interlaced_frame,
            },
        }
        self.video_idx += 1
        self.pos += 1
        return result

    def close(self) -> None:
        self.container.close()

    def seek(self, pos: int, **kwargs: Any) -> None:
        """Move to the frame of row pos, given the row's stored outputs in kwargs; the next __next__() returns it."""
        assert next(iter(kwargs.values()), None) is not None

        seek_time: float
        if 'pos_msec' in kwargs:
            video_idx = kwargs['pos_frame']
            # TODO: rows written by legacy_frame_iterator measure pos_msec from the video start and need
            #     video_start_time added; rows from the old FrameIterator store roughly absolute milliseconds
            seek_time = kwargs['pos_msec'] / 1000.0
        else:
            assert 'frame_attrs' in kwargs
            video_idx = kwargs['frame_attrs']['index']
            seek_time = kwargs['frame_attrs']['time']

        assert isinstance(video_idx, int)
        assert isinstance(seek_time, float)

        if pos == self.pos:
            # consecutive row: its frame is at or after current_frame
            self._decode_to(seek_time)
        else:
            # Subtlety: The offset passed in to seek() is not the pts, but rather the pts adjusted for the video start
            # time.
            seek_offset = math.floor((seek_time - self.video_start_time) / self.video_time_base)
            self._decode_from(seek_offset, seek_time)
            if self.current_frame is None:
                # container.seek() can end up past the last frame (seen in .flv files)
                self._decode_from(0, seek_time)
        assert (
            self.current_frame is not None
            and abs(float(self.current_frame.pts * self.video_time_base) - seek_time) < 1e-3
        )
        self.pos = pos
        self.video_idx = video_idx
        self.at_seek_target = True

    def _decode_from(self, seek_offset: int, seek_time: float) -> None:
        """Seek the container to seek_offset and decode up to the first frame at or after seek_time."""
        self.container.seek(seek_offset, backward=True, stream=self.container.streams.video[0])
        # the old generator may be exhausted or have frames from before the seek
        self._start_decoding()
        self._decode_to(seek_time)

    def _decode_to(self, seek_time: float) -> None:
        """Decode up to the first frame at or after seek_time."""
        while (
            self.current_frame is not None and float(self.current_frame.pts * self.video_time_base) < seek_time - 1e-3
        ):
            self._decode_next()

    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        fps = bound_args.get('fps')
        num_frames = bound_args.get('num_frames')
        keyframes_only = bound_args.get('keyframes_only', False)
        if int(fps is not None) + int(num_frames is not None) + int(keyframes_only) > 1:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'At most one of `fps`, `num_frames` or `keyframes_only` may be specified',
            )
        if fps is not None and (not isinstance(fps, (int, float)) or fps <= 0.0):
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '`fps` must be a positive number')


class LegacyFrame(TypedDict):
    frame: pxt.Image
    frame_idx: int
    pos_msec: float
    pos_frame: int


@pxt.iterator(unstored_cols=['frame'])
class legacy_frame_iterator(pxt.PxtIterator[LegacyFrame]):
    # A retrofitted implementation of the legacy frame iterator interface, with output schema `frame_idx`, `pos_msec`,
    # and `pos_frame` (instead of `frame_attrs`). It wraps the new `frame_iterator` and derives the legacy outputs from
    # `frame_attrs`.
    underlying: pxt.PxtIterator[Frame]

    def __init__(
        self, video: pxt.Video, *, fps: float | None = None, num_frames: int | None = None, keyframes_only: bool = False
    ) -> None:
        self.underlying = frame_iterator.create_iter(  # type: ignore[attr-defined]
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
    """One segment of a video: the segment's video plus where it falls in the source video."""

    segment_start: float | None
    """Segment start, in seconds from the start of the source video; None if the source stream does not report it."""
    segment_start_pts: int | None
    """Segment start, as a presentation timestamp (in the source stream's time base); None if unavailable."""
    segment_end: float | None
    """Segment end, in seconds from the start of the source video; None if the source stream does not report it."""
    segment_end_pts: int | None
    """Segment end, as a presentation timestamp (in the source stream's time base); None if unavailable."""
    video_segment: pxt.Video
    """The segment's video."""


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
    if not (video_path.exists() and video_path.is_file()):
        raise pxt.RequestError(pxt.ErrorCode.INVALID_DATA_FORMAT, f'video_splitter(): not a valid video file: {video}')

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
                else None
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
                if actual_duration is None:
                    Path(segment_path).unlink()
                    raise pxt.RequestError(
                        pxt.ErrorCode.INVALID_DATA_FORMAT,
                        f'video_splitter(): could not determine duration of a generated segment from {video}',
                    )
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
            raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, error_msg) from e

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
                if actual_duration is None:
                    Path(segment_path).unlink()
                    raise pxt.RequestError(
                        pxt.ErrorCode.INVALID_DATA_FORMAT,
                        f'video_splitter(): could not determine duration of a generated segment from {video}',
                    )
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
            raise pxt.RequestError(pxt.ErrorCode.UNSUPPORTED_OPERATION, error_msg) from e


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
        raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Must specify either duration or segment_times')
    if duration is not None and segment_times is not None:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, 'duration and segment_times cannot both be specified'
        )
    if segment_times is not None and overlap is not None:
        raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'overlap cannot be specified with segment_times')
    if duration is not None and isinstance(duration, (int, float)):
        if duration <= 0.0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'duration must be a positive number: {duration}')
        if (
            min_segment_duration is not None
            and isinstance(min_segment_duration, (int, float))
            and duration < min_segment_duration
        ):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'duration must be at least min_segment_duration: {duration} < {min_segment_duration}',
            )
        if overlap is not None and isinstance(overlap, (int, float)) and overlap >= duration:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'overlap must be less than duration: {overlap} >= {duration}'
            )
    if mode == 'accurate' and overlap is not None:
        raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, "Cannot specify overlap for mode='accurate'")
    if mode == 'fast':
        if video_encoder is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, "Cannot specify video_encoder for mode='fast'"
            )
        if video_encoder_args is not None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, "Cannot specify video_encoder_args for mode='fast'"
            )


_class_names = [
    'FrameAttrs',
    'Frame',
    'frame_iterator',
    'LegacyFrame',
    'legacy_frame_iterator',
    'VideoSegment',
    'video_splitter',
]
__all__ = local_public_names(__name__) + _class_names
