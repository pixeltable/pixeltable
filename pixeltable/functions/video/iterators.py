import glob
import logging
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Literal, NamedTuple, TypedDict

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

# A fresh iterator is on frame 0. Walking farther than this decodes the prefix, so that
# target keyframe-seeks. An open index walks about one second and keyframe-seeks a
# longer gap. An empty relative seek, or one that lands past the target, decodes from
# the start.
_MAX_FORWARD_WALK = 8


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


class _ReloadTarget(NamedTuple):
    """Stored frame for the next() that follows seek().

    check_time gates the extra time comparison in _current_frame_matches(). The time is
    absolute on both seek paths. Legacy pos_msec matches on the frame index.
    """

    time: float
    frame_index: int
    check_time: bool


@pxt.iterator(unstored_cols=['frame'])
class frame_iterator(pxt.PxtIterator[Frame]):
    """
    Iterator over frames of a video. At most one of `fps`, `num_frames` or `keyframes_only` may be specified. If `fps`
    is specified, then frames will be extracted at the specified rate (frames per second). If `num_frames` is specified,
    then frames are extracted at that many evenly spaced times. A frame held mid-video moves later samples onto later
    frames, so the result is shorter than `num_frames` only when samples run past the last frame. If neither is
    specified, then all frames will be extracted.

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
        num_frames: Number of evenly spaced frames to extract. A frame held mid-video moves later samples onto
            later frames, so the iterator returns that many frames unless samples run past the last frame. If
            `num_frames` is greater than the number of frames in the video, all frames are extracted.
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

        Create a view that extracts 10 evenly spaced frames from each video:

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
    # One decode generator for this container; a second decode() drops the frame flushed at the end.
    frame_gen: Iterator[av.VideoFrame]
    # seek() sets this. The following next() returns that stored frame.
    _reload_target: _ReloadTarget | None
    # Formula that matched the frame last landed on. An index this decoder does not walk to resolves again.
    _pos_msec_formula: Literal['current', 'historical'] | None

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

        stream = self.container.streams.video[0]
        self.video_time_base = stream.time_base
        time_range = av_utils.video_time_range(self.container)
        if time_range is None:
            self.video_start_time = float((stream.start_time or 0) * self.video_time_base)
            # num_frames spaces its samples before decoding, so it needs the packet scan now.
            # fps learns the end on the last frame. All frames and keyframes never read it.
            self.video_duration = (
                av_utils.get_video_duration(str(self.video_path)) if self.num_frames is not None else None
            )
        else:
            self.video_start_time, end = time_range
            self.video_duration = end - self.video_start_time

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
        self._reload_target = None
        self._pos_msec_formula = None
        if self.num_frames is not None:
            # Divide the video duration into num_frames evenly spaced intervals. The extraction times are the midpoints
            # of those intervals.
            assert self.video_duration is not None
            self.extraction_step = self.video_duration / self.num_frames
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
        self.frame_gen = self.container.decode(video=0)
        self.cur_frame = self.next_frame()

    def next_frame(self) -> av.VideoFrame | None:
        try:
            return next(self.frame_gen, None)
        except EOFError:
            return None

    def __next__(self) -> Frame:
        # seek() reloads one stored row before sampling. A stale next_extraction_time would
        # decode past that frame, or to EOF, and then seek again. Expansion leaves this unset,
        # so the num_frames cap still applies.
        target = self._reload_target
        self._reload_target = None
        if target is not None:
            if self.cur_frame is not None and self._current_frame_matches(target):
                return self._emit_current(self.next_frame())
            return self._emit_stored_frame(target)

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
                # Stop at the requested count before either branch. A later sample can still
                # land on a real frame: NUT's stored end is the last PTS, and MPEG-PS keeps
                # packets past the duration it reports.
                if self.num_frames is not None and self.pos >= self.num_frames:
                    raise StopIteration

                if next_frame is None:
                    # The last frame stays up until the video ends. A sample after its PTS still belongs
                    # to it. Stop only once the sample is past that end. Yielding once covers later
                    # samples that land on this same frame.
                    if self._sample_past_end(self.next_extraction_time, cur_frame_time):
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

            return self._emit_current(next_frame)

    def _current_frame_matches(self, target: _ReloadTarget) -> bool:
        assert self.cur_frame is not None
        if self.video_idx != target.frame_index:
            return False
        if not target.check_time:
            return True
        frame_time = float(self.cur_frame.pts * self.video_time_base)
        return abs(frame_time - target.time) < 1e-3

    def _emit_current(self, next_frame: av.VideoFrame | None) -> Frame:
        assert self.cur_frame is not None
        cur_frame_pts = self.cur_frame.pts
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
        if self.fps is not None:
            # pos / fps stays on the grid. pos * (1/fps) is 0.9999999999999999 at fps=49.
            self.next_extraction_time = self.video_start_time + self.pos / self.fps
        elif self.extraction_step is not None:
            # (pos + 1/2) * step stays on the grid after seek() jumps pos. A running sum stays behind.
            self.next_extraction_time = self.video_start_time + (self.pos + 0.5) * self.extraction_step
        return result

    def _emit_stored_frame(self, target: _ReloadTarget) -> Frame:
        self._position_at_frame(self.pos, target.time, target.frame_index)
        if self.cur_frame is None:
            raise StopIteration
        return self._emit_current(self.next_frame())

    def _sample_past_end(self, sample_time: float, frame_time: float) -> bool:
        """True when the sample is past the end of the video.

        The end matches VideoFrames.end_time(): the stored range when it runs past this frame,
        otherwise this frame's duration, otherwise one average frame. The sample is then at or
        after that end. With no rate, the end stays at the frame timestamp and the sample is past it.
        """
        assert self.cur_frame is not None
        if self.video_duration is None and self.fps is not None:
            # Header timing did not give an end. Only the last frame needs it.
            self.video_duration = av_utils.get_video_duration(str(self.video_path))
        stored_end = None if self.video_duration is None else self.video_start_time + self.video_duration
        duration_s = av_utils.frame_duration_seconds(self.cur_frame, self.video_time_base)
        rate = self.container.streams.video[0].average_rate
        end = av_utils.display_end(stored_end, frame_time, duration_s, None if rate == 0 else rate)
        # A sample at the only known timestamp still belongs to the frame.
        if end > frame_time:
            if self.fps is not None:
                # pos / fps can land just under an exact end. 33 / 1.1 is 29.999999999999996.
                # The allowance scales with the end index. An absolute bump treats pos 0 as past
                # a tiny positive end, so fps=1e-5 on a 0.04s frame yields nothing.
                end_index = (end - self.video_start_time) * self.fps
                return self.pos >= end_index * (1.0 - 1e-9)
            return sample_time >= end
        return sample_time > frame_time

    def close(self) -> None:
        self.container.close()

    def seek(self, pos: int, **kwargs: Any) -> None:
        assert next(iter(kwargs.values()), None) is not None

        seek_time: float
        candidates: tuple[float, float] | None = None
        if 'pos_msec' in kwargs:
            video_idx = kwargs['pos_frame']
            assert isinstance(video_idx, int)
            candidates = self._pos_msec_candidates(float(kwargs['pos_msec']))
            seek_time = self._pos_msec_seek_time(candidates, video_idx)
            check_time = False
        else:
            assert 'frame_attrs' in kwargs
            video_idx = kwargs['frame_attrs']['index']
            seek_time = kwargs['frame_attrs']['time']
            check_time = True

        assert isinstance(video_idx, int)
        assert isinstance(seek_time, float)
        self._reload_target = _ReloadTarget(seek_time, video_idx, check_time)

        # pos already matches on a sequential read. Formula probes use another container,
        # so this decoder is still on cur_frame. next() seeks when that frame is not the stored one.
        if self.pos == pos and self.cur_frame is not None:
            return

        found = self._position_at_frame(pos, seek_time, video_idx)
        # None means the seek did not start on frame 0. For pos_msec, seek_time was already
        # resolved from that frame, so the picture is the stored one. video_idx means the
        # forward walk or the empty-seek fallback decoded that index.
        if candidates is None or found is None or found == video_idx or self._pos_msec_formula is None:
            return
        # The counted index belongs to the other formula.
        current, historical = candidates
        self._pos_msec_formula = 'historical' if self._pos_msec_formula == 'current' else 'current'
        seek_time = current if self._pos_msec_formula == 'current' else historical
        self._reload_target = _ReloadTarget(seek_time, video_idx, False)
        self._position_at_frame(pos, seek_time, video_idx)

    def _pos_msec_candidates(self, pos_msec: float) -> tuple[float, float]:
        """Decorator time, then the v0.5.6-v0.5.18 time, both in absolute seconds."""
        start = self.video_start_time
        return start + pos_msec / 1000.0, pos_msec / 1000.0 + start / 1000.0

    def _pos_msec_seek_time(self, candidates: tuple[float, float], frame_index: int) -> float:
        """Absolute time for a stored pos_msec.

        Rows since the decorator store (frame_time - start) * 1000 and seek at
        start + pos_msec / 1000. v0.5.6 through v0.5.18 stored frame_time * 1000 - start,
        with start in seconds, and seek at pos_msec / 1000 + start / 1000. The two times
        differ by about start * 0.999 and can both be real frames, so the stored frame index decides.
        The open frame, or a frame walked to from it, supplies the PTS that picks the formula.
        An index behind the decoder, or farther ahead than the forward walk, resolves on its own.
        A cached formula is not applied to a row this decoder does not land on. Probes use another container.
        """
        current, historical = candidates
        if abs(current - historical) < 1e-3:
            return current

        matched = self._candidate_on_current_frame(frame_index, current, historical)
        if matched is not None:
            return matched
        # A sampled row leaves the decoder on the next frame, not on pos_frame. The frame
        # walked to has the PTS; resolving that gap would scan from frame 0.
        if self.video_idx < frame_index and self._walk_forward(frame_index):
            assert self.cur_frame is not None and self.cur_frame.pts is not None
            frame_time = float(self.cur_frame.pts * self.video_time_base)
            if abs(frame_time - current) < 1e-3:
                self._pos_msec_formula = 'current'
                return current
            if abs(frame_time - historical) < 1e-3:
                self._pos_msec_formula = 'historical'
                return historical
            # The stored index is this frame. Keep its PTS so a later check does not seek away.
            self._pos_msec_formula = 'current'
            return frame_time
        return self._resolve_pos_msec_formula(frame_index, current, historical)

    def _candidate_on_current_frame(self, frame_index: int, current: float, historical: float) -> float | None:
        """The candidate within 1ms of cur_frame, when that frame is already frame_index."""
        if self.cur_frame is None or self.cur_frame.pts is None or self.video_idx != frame_index:
            return None
        frame_time = float(self.cur_frame.pts * self.video_time_base)
        if abs(frame_time - current) < 1e-3:
            self._pos_msec_formula = 'current'
            return current
        if abs(frame_time - historical) < 1e-3:
            self._pos_msec_formula = 'historical'
            return historical
        return None

    def _resolve_pos_msec_formula(self, frame_index: int, current: float, historical: float) -> float:
        # New rows store the decorator time. Accept it when the frame it decodes is pos_frame,
        # before scanning from frame 0. Only then try the v0.5.6-v0.5.18 time.
        if self._index_from_first_frame(current) == frame_index:
            self._pos_msec_formula = 'current'
            return current
        if self._index_from_first_frame(historical) == frame_index:
            self._pos_msec_formula = 'historical'
            return historical
        frame_time = self._time_of_frame_index(frame_index)
        if frame_time is not None and abs(current - frame_time) < 1e-3:
            self._pos_msec_formula = 'current'
            return current
        if frame_time is not None and abs(historical - frame_time) < 1e-3:
            self._pos_msec_formula = 'historical'
            return historical
        self._pos_msec_formula = 'current'
        return current

    def _index_from_first_frame(self, seek_time: float) -> int | None:
        """Decoded index of the frame within 1ms of seek_time, if this seek starts on the first frame.

        A seek that lands on a later keyframe does not reveal the absolute index. None also covers
        a relative seek that returns no packet. The caller then scans once from frame 0.
        """
        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            seek_offset = math.floor((seek_time - self.video_start_time) / self.video_time_base)
            container.seek(max(seek_offset, 0), backward=True, stream=stream)
            frame_gen = container.decode(video=0)
            try:
                frame: av.VideoFrame | None = next(frame_gen)
            except (StopIteration, EOFError):
                return None
            if frame is None or frame.pts is None:
                return None
            if abs(float(frame.pts * self.video_time_base) - self.video_start_time) >= 1e-3:
                return None
            index = 0
            while frame is not None and float(frame.pts * self.video_time_base) < seek_time - 1e-3:
                try:
                    frame = next(frame_gen)
                except (StopIteration, EOFError):
                    return None
                index += 1
            if frame is None or frame.pts is None:
                return None
            if abs(float(frame.pts * self.video_time_base) - seek_time) >= 1e-3:
                return None
            return index

    def _time_of_frame_index(self, frame_index: int) -> float | None:
        """PTS, in seconds, of decoded frame frame_index, counted from the start of the stream.

        Uses its own container so the iterator's decoder stays where the caller left it.
        """
        if frame_index < 0:
            return None
        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            # 0 is at or before the first packet. start_time can sit on a later frame.
            container.seek(0, backward=True, stream=stream)
            frame_gen = container.decode(video=0)
            frame: av.VideoFrame | None = None
            index = 0
            while index <= frame_index:
                try:
                    frame = next(frame_gen)
                except (StopIteration, EOFError):
                    return None
                if index == frame_index:
                    break
                index += 1
            if frame is None or frame.pts is None:
                return None
            return float(frame.pts * self.video_time_base)

    def _forward_walk_limit(self) -> int:
        """Frames seek() may decode forward before it keyframe-seeks.

        Frame 0 stops at _MAX_FORWARD_WALK so a late frame does not decode the prefix.
        An open index walks about one second, the gap between consecutive fps=1 samples,
        and keyframe-seeks a longer gap.
        """
        if self.video_idx == 0:
            return _MAX_FORWARD_WALK
        rate = self.container.streams.video[0].average_rate
        if rate is None or rate == 0:
            return _MAX_FORWARD_WALK
        # ceil(rate) is one second of frames. fps=1 leaves the decoder one frame past the
        # previous sample, and the extra frame covers a sample that lands one frame later.
        return max(_MAX_FORWARD_WALK, math.ceil(float(rate)) + 1)

    def _walk_forward(self, video_idx: int) -> bool:
        """Decode the open generator forward to video_idx.

        Returns False without decoding when the target is behind the open index, or farther
        ahead than _forward_walk_limit(). On success cur_frame is that frame. If the stream
        ends first, cur_frame is cleared and the caller seeks.
        """
        if self.cur_frame is None or self.cur_frame.pts is None or self.video_idx > video_idx:
            return False
        if video_idx - self.video_idx > self._forward_walk_limit():
            return False
        frame = self.cur_frame
        index = self.video_idx
        while index < video_idx:
            nxt = self.next_frame()
            if nxt is None or nxt.pts is None:
                self.cur_frame = None
                return False
            frame = nxt
            index += 1
        self.cur_frame = frame
        self.video_idx = index
        return True

    def _advance_to_stored_frame(self, pos: int, seek_time: float, video_idx: int) -> bool:
        """Walk the open generator to video_idx when that frame is ahead.

        A fresh iterator keyframe-seeks past _MAX_FORWARD_WALK. An open index walks about a
        second and keyframe-seeks a longer gap. A backward seek, or a timestamp that is not
        this frame's, leaves the seek path to run.
        """
        if not self._walk_forward(video_idx):
            return False
        assert self.cur_frame is not None and self.cur_frame.pts is not None
        if abs(float(self.cur_frame.pts * self.video_time_base) - seek_time) >= 1e-3:
            return False
        self.pos = pos
        return True

    def _decode_to_index(self, video_idx: int) -> None:
        """Replace frame_gen with one generator from the start and decode through video_idx."""
        stream = self.container.streams.video[0]
        self.container.seek(0, backward=True, stream=stream)
        self.frame_gen = self.container.decode(video=0)
        self.cur_frame = self.next_frame()
        index = 0
        while self.cur_frame is not None and index < video_idx:
            self.cur_frame = self.next_frame()
            index += 1

    def _position_at_frame(self, pos: int, seek_time: float, video_idx: int) -> int | None:
        """Move to seek_time.

        Returns video_idx when the open generator walked to that frame, or when an empty
        relative seek or a seek that landed past the target decoded that index from the
        start. Returns the counted index when the seek itself started on the first frame.
        None means the count is not absolute.
        """
        if self._advance_to_stored_frame(pos, seek_time, video_idx):
            return video_idx

        self.pos = pos
        self.video_idx = video_idx

        # Subtlety: The offset passed in to seek() is not the pts, but rather the pts adjusted for the video start time.
        seek_offset = math.floor((seek_time - self.video_start_time) / self.video_time_base)
        self.container.seek(seek_offset, backward=True, stream=self.container.streams.video[0])
        self.frame_gen = self.container.decode(video=0)

        self.cur_frame = self.next_frame()
        if self.cur_frame is None:
            # A default-GOP MPEG-TS returns no packet once the offset reaches the stream start.
            # seek(0) and this one generator still decode that index, including the last frame.
            self._decode_to_index(video_idx)
            return video_idx if self.cur_frame is not None else None

        first_time = float(self.cur_frame.pts * self.video_time_base)
        started_at_first = abs(first_time - self.video_start_time) < 1e-3
        if not started_at_first and first_time >= seek_time + 1e-3:
            # This keyframe is already past the target. A 20s default-GOP MPEG-TS seeks
            # frame 100 to the keyframe at frame 250. Decoding from the start reaches it.
            # Doing that again cannot help a seek that already read the first frame.
            self._decode_to_index(video_idx)
            assert self.cur_frame is None or abs(float(self.cur_frame.pts * self.video_time_base) - seek_time) < 1e-3
            return video_idx if self.cur_frame is not None else None
        index = 0
        while self.cur_frame is not None and float(self.cur_frame.pts * self.video_time_base) < seek_time - 1e-3:
            self.cur_frame = self.next_frame()
            index += 1
        assert self.cur_frame is None or abs(float(self.cur_frame.pts * self.video_time_base) - seek_time) < 1e-3
        if self.cur_frame is None or not started_at_first:
            return None
        return index

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
