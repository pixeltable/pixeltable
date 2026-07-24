"""
Pixeltable UDFs for `AudioType`.
"""

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any, Generator, NamedTuple, TypedDict

import av
import numpy as np

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.env import Env
from pixeltable.utils import av as av_utils
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

from . import util

_logger = logging.getLogger(__name__)


@pxt.udf(is_method=True)
def get_metadata(audio: pxt.Audio) -> util.ContainerMetadata:
    """
    Gets various metadata associated with an audio file and returns it as
    a [`ContainerMetadata`][pixeltable.functions.ContainerMetadata] dictionary.

    Args:
        audio: The audio to get metadata for.

    Returns:
        A [`ContainerMetadata`][pixeltable.functions.ContainerMetadata] with typical structure:

            ```json
            {
                'size': 2568827,
                'streams': [
                    {
                        'type': 'audio',
                        'frames': 0,
                        'duration': 2646000,
                        'metadata': {},
                        'time_base': 2.2675736961451248e-05,
                        'codec_context': {
                            'name': 'flac',
                            'profile': None,
                            'channels': 1,
                            'codec_tag': '\\x00\\x00\\x00\\x00',
                        },
                        'duration_seconds': 60.0,
                    }
                ],
                'bit_rate': 342510,
                'metadata': {'encoder': 'Lavf61.1.100'},
                'bit_exact': False,
            }
            ```

    Examples:
        Extract metadata for files in the `audio_col` column of the table `tbl`:

        >>> tbl.select(tbl.audio_col.get_metadata()).collect()
    """
    return util.get_metadata(audio)


@pxt.udf()
def encode_audio(
    audio_data: pxt.Array[pxt.Float], *, input_sample_rate: int, format: str, output_sample_rate: int | None = None
) -> pxt.Audio:
    """
    Encodes an audio clip represented as an array into a specified audio format.

    Parameters:
        audio_data: An array of sampled amplitudes. The accepted array shapes are `(N,)` or `(1, N)` for mono audio
            or `(2, N)` for stereo.
        input_sample_rate: The sample rate of the input audio data.
        format: The desired output audio format. The supported formats are 'wav', 'mp3', 'flac', and 'mp4'.
        output_sample_rate: The desired sample rate for the output audio. Defaults to the input sample rate if
            unspecified.

    Examples:
        Add a computed column with encoded FLAC audio files to a table with audio data (as arrays of floats) and sample
        rates:

        >>> t.add_computed_column(
        ...     audio_file=encode_audio(
        ...         t.audio_data, input_sample_rate=t.sample_rate, format='flac'
        ...     )
        ... )
    """
    if format not in av_utils.AUDIO_FORMATS:
        raise pxt.RequestError(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            f'Only the following formats are supported: {av_utils.AUDIO_FORMATS.keys()}',
        )
    if output_sample_rate is None:
        output_sample_rate = input_sample_rate

    codec, ext = av_utils.AUDIO_FORMATS[format]
    output_path = str(TempStore.create_path(extension=f'.{ext}'))

    match audio_data.shape:
        case (_,):
            # Mono audio as 1D array, reshape for pyav
            layout = 'mono'
            audio_data_transformed = audio_data[None, :]
        case (1, _):
            # Mono audio as 2D array, simply reshape and transpose the input for pyav
            layout = 'mono'
            audio_data_transformed = audio_data.reshape(-1, 1).transpose()
        case (2, _):
            # Stereo audio. Input layout: [[L0, L1, L2, ...],[R0, R1, R2, ...]],
            # pyav expects: [L0, R0, L1, R1, L2, R2, ...]
            layout = 'stereo'
            audio_data_transformed = np.empty(audio_data.shape[1] * 2, dtype=audio_data.dtype)
            audio_data_transformed[0::2] = audio_data[0]
            audio_data_transformed[1::2] = audio_data[1]
            audio_data_transformed = audio_data_transformed.reshape(1, -1)
        case _:
            raise pxt.RequestError(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                f'Supported input array shapes are (N,), (1, N) for mono and (2, N) for stereo, got {audio_data.shape}',
            )

    with av.open(output_path, mode='w') as output_container:
        stream = output_container.add_stream(codec, rate=output_sample_rate)
        assert isinstance(stream, av.AudioStream)

        frame = av.AudioFrame.from_ndarray(audio_data_transformed, format='flt', layout=layout)
        frame.sample_rate = input_sample_rate

        for packet in stream.encode(frame):
            output_container.mux(packet)
        for packet in stream.encode():
            output_container.mux(packet)

        return output_path


@pxt.udf(is_method=True, run_in_thread=True)
def multiply_volume(
    audio: pxt.Audio, *, factor: float, start_time: float | None = None, end_time: float | None = None
) -> pxt.Audio:
    """
    Scale the volume of an audio clip by a constant factor using ffmpeg's volume filter.

    If `start_time` and/or `end_time` are given, only the samples within that range are scaled and
    the rest of the clip is passed through unchanged. Omit both to apply the gain to the entire clip.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        audio: Input audio.
        factor: Volume multiplier. `1.0` leaves the audio unchanged, `0.5` halves the amplitude
            (about -6 dB), `2.0` doubles it (about +6 dB), `0.0` mutes the affected range.
        start_time: Start of the range to scale, in seconds. Defaults to the beginning of the clip.
        end_time: End of the range to scale, in seconds. Defaults to the end of the clip.

    Returns:
        A new audio clip with the gain applied.

    Examples:
        Halve the volume of an entire clip:

        >>> tbl.select(tbl.audio.multiply_volume(factor=0.5)).collect()

        Boost only the first five seconds:

        >>> tbl.select(
        ...     tbl.audio.multiply_volume(factor=1.5, start_time=0.0, end_time=5.0)
        ... ).collect()
    """
    Env.get().require_binary('ffmpeg')

    if start_time is not None and start_time < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`start_time` must be non-negative, got {start_time}')
    if end_time is not None and end_time < 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`end_time` must be non-negative, got {end_time}')
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_ARGUMENT, f'`end_time` ({end_time}) must be greater than `start_time` ({start_time})'
        )

    if start_time is None and end_time is None:
        filter_expr = f'volume={factor}'
    else:
        if start_time is None:
            gate = f'lt(t,{end_time})'
        elif end_time is None:
            gate = f'gte(t,{start_time})'
        else:
            gate = f'between(t,{start_time},{end_time})'
        filter_expr = f"volume={factor}:enable='{gate}'"

    input_path = str(audio)
    ext = Path(input_path).suffix or '.wav'
    output_path = str(TempStore.create_path(extension=ext))

    cmd = ['-i', input_path, '-af', filter_expr]
    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


@pxt.udf(is_method=True, run_in_thread=True)
def fade_in(audio: pxt.Audio, *, duration: float) -> pxt.Audio:
    """
    Apply a linear fade-in over the first `duration` seconds of an audio clip using ffmpeg's afade filter.

    The volume ramps from silence at the start of the clip to full volume at time `duration`.
    Samples past `duration` are unchanged.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        audio: Input audio.
        duration: Length of the fade-in, in seconds. Must be positive. If it exceeds the clip's
            duration, the entire clip is faded in.

    Returns:
        A new audio clip with the fade-in applied.

    Examples:
        Fade in over the first two seconds:

        >>> tbl.select(tbl.audio.fade_in(duration=2.0)).collect()
    """
    Env.get().require_binary('ffmpeg')
    if duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`duration` must be positive, got {duration}')

    input_path = str(audio)
    clip_duration = av_utils.get_audio_duration(input_path)
    if clip_duration is None:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'cannot determine duration of audio clip: {input_path}'
        )
    # make sure we reach full volume by the end
    fade_duration = min(duration, clip_duration)

    ext = Path(input_path).suffix or '.wav'
    output_path = str(TempStore.create_path(extension=ext))

    cmd = ['-i', input_path, '-af', f'afade=t=in:st=0:d={fade_duration}']
    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


@pxt.udf(is_method=True, run_in_thread=True)
def fade_out(audio: pxt.Audio, *, duration: float) -> pxt.Audio:
    """
    Apply a linear fade-out over the last `duration` seconds of an audio clip using ffmpeg's afade filter.

    The volume ramps from full volume at time `clip_duration - duration` to silence at the end of
    the clip. Samples before that point are unchanged.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        audio: Input audio.
        duration: Length of the fade-out, in seconds. Must be positive. If it exceeds the clip's
            duration, the entire clip is faded out.

    Returns:
        A new audio clip with the fade-out applied.

    Examples:
        Fade out over the last three seconds:

        >>> tbl.select(tbl.audio.fade_out(duration=3.0)).collect()
    """
    Env.get().require_binary('ffmpeg')
    if duration <= 0:
        raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, f'`duration` must be positive, got {duration}')

    input_path = str(audio)
    clip_duration = av_utils.get_audio_duration(input_path)
    if clip_duration is None:
        raise pxt.RequestError(
            pxt.ErrorCode.INVALID_DATA_FORMAT, f'cannot determine duration of audio clip: {input_path}'
        )
    fade_duration = min(duration, clip_duration)
    start = clip_duration - fade_duration

    ext = Path(input_path).suffix or '.wav'
    output_path = str(TempStore.create_path(extension=ext))

    cmd = ['-i', input_path, '-af', f'afade=t=out:st={start}:d={fade_duration}']
    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


@pxt.udf(is_method=True, run_in_thread=True)
def normalize(audio: pxt.Audio) -> pxt.Audio:
    """
    Peak-normalize an audio clip so its loudest sample reaches full scale (0 dBFS) using ffmpeg's volume filter.

    The whole clip is scaled by a single constant factor chosen so that the maximum absolute
    sample value becomes 1.0. Silent clips are passed through without a gain change.

    __Requirements:__

    - `ffmpeg` needs to be installed and in PATH

    Args:
        audio: Input audio.

    Returns:
        A new audio clip with peak normalization applied.

    Examples:
        Normalize a clip:

        >>> tbl.select(tbl.audio.normalize()).collect()
    """
    Env.get().require_binary('ffmpeg')

    input_path = str(audio)
    # Measure the peak via ffmpeg's volumedetect filter; then apply the inverse gain in dB so
    # the peak sits at 0 dBFS. For silent clips (None) or already-full-scale input (0 dB),
    # `gain_db` is 0 and the second pass is a no-op re-encode.
    max_db = av_utils.get_max_volume_db(input_path)
    gain_db = 0.0 if max_db is None else -max_db

    ext = Path(input_path).suffix or '.wav'
    output_path = str(TempStore.create_path(extension=ext))

    cmd = ['-i', input_path, '-af', f'volume={gain_db}dB']
    return av_utils.run_ffmpeg_cmdline(cmd, output_path)


class _PacketInfo(NamedTuple):
    """
    A demuxed packet plus its original timestamps, retained separately because muxing rescales the packet in place
    and overlap means the same packet is muxed into more than one segment.
    """

    packet: av.Packet
    pts: int
    dts: int | None
    duration: int
    mean_square: float = -1.0  # the packet's normalized energy


class AudioSegment(TypedDict):
    segment_start: float
    segment_end: float
    audio_segment: pxt.Audio | None


class _SilenceScanner:
    """
    Streaming run-length silence detector:
    - packet information is fed in order via observe()
    - tracks the current run of consecutive silent packets to identify silence based on init() parameters
    """

    threshold_meansq: float  # mean-square level at or below which a packet is silent
    min_silence_len_pts: int  # minimum length of a silent run that qualifies as a cut point
    min_segment_pts: int  # minimum distance from the segment start before a cut is allowed

    # silence-related state
    segment_start_pts: int | None  # start of the current segment, set on the first observe()
    run_start_pts: int | None  # start of the current silent run, or None when not in one
    cut_index: int | None  # index of the last packet to keep, or None if no qualifying silence yet

    def __init__(self, threshold_ms: float, min_silence_len_pts: int, min_segment_pts: int):
        self.threshold_meansq = threshold_ms
        self.min_silence_len_pts = min_silence_len_pts
        self.min_segment_pts = min_segment_pts
        self.segment_start_pts = None
        self.run_start_pts = None
        self.cut_index = None

    def observe(self, packet_index: int, start_pts: int, end_pts: int, mean_square: float) -> None:
        """Feed the next packet data and update silence-related state."""
        if self.segment_start_pts is None:
            self.segment_start_pts = start_pts
        if mean_square > self.threshold_meansq:
            self.run_start_pts = None  # ending the current run
            return
        if self.run_start_pts is None:
            self.run_start_pts = start_pts
        # cut as late as possible within a run that is at least min_silence_len long, as long as the resulting
        # segment is at least min_segment_duration
        if (
            end_pts - self.run_start_pts >= self.min_silence_len_pts
            and end_pts - self.segment_start_pts >= self.min_segment_pts
        ):
            self.cut_index = packet_index


@pxt.iterator
class audio_splitter(pxt.PxtIterator[AudioSegment]):
    """
    Iterator over segments of an audio file. The audio file is split into smaller segments, sized either by
    `duration` (seconds) or by `max_size` (bytes). Exactly one of the two must be specified.

    - With `max_size`, every emitted segment is guaranteed not to exceed the given number of bytes.
    - With `duration`, each segment is approximately that long (a segment may run over by up to one packet).

    __Outputs__:

        One row per audio segment, with the following columns:

        - `segment_start` (`pxt.Float`): Start time of the audio segment in seconds
        - `segment_end` (`pxt.Float`): End time of the audio segment in seconds
        - `audio_segment` (`pxt.Audio | None`): The audio content of the segment

    When `min_silence_len` is set, segment boundaries are snapped to silences so that segments do not end in the
    middle of speech: each segment ends at the latest silence at or before its `duration` or `max_size` budget.
    `trim_leading_silence` additionally drops silence from the start of every segment.

    Args:
        duration: Audio segment duration in seconds. Mutually exclusive with `max_size`.
        max_size: Maximum audio segment size in bytes. Mutually exclusive with `duration`.
        overlap: Overlap between consecutive segments in seconds
        min_segment_duration: Drop the last segment if it is smaller than `min_segment_duration` (in seconds).
        min_silence_len: If set, enables silence-aware boundaries; the minimum length in seconds of a quiet stretch
            that counts as a usable cut point.
        silence_thresh: Level in dBFS at or below which audio is considered silent. Used when `min_silence_len` is set
            or `trim_leading_silence` is true.
        trim_leading_silence: If true, drop leading silence from each segment so it starts at audible content.

    Examples:
        This example assumes an existing table `tbl` with a column `audio` of type `pxt.Audio`.

        Create a view that splits all audio files into segments of 30 seconds with 5 seconds overlap:

        >>> pxt.create_view(
        ...     'audio_segments',
        ...     tbl,
        ...     iterator=audio_splitter(tbl.audio, duration=30.0, overlap=5.0),
        ... )

        Create a view that splits all audio files into segments of at most 24 MB, for transcription:

        >>> pxt.create_view(
        ...     'audio_segments',
        ...     tbl,
        ...     iterator=audio_splitter(tbl.audio, max_size=24 * 1024 * 1024),
        ... )
    """

    audio_path: Path
    overlap: float
    min_segment_duration: float
    segment_duration: float  # set only in duration mode
    max_size: int  # set only in max_size mode

    # silence detection
    trim_leading_silence: bool
    with_silence: bool  # whether segment boundaries are placed at silences
    decode_needed: bool  # whether packets must be decoded to measure energy (silence-aware cutting or leading trim)
    silence_threshold_meansq: float  # mean-square form of silence_thresh
    min_silence_len_pts: int  # set only when with_silence

    # audio stream details
    container: av.container.input.InputContainer
    in_stream: av.audio.stream.AudioStream
    audio_time_base: Fraction  # seconds per presentation timestamp unit

    # generator producing the output segments; created in __init__() and driven by __next__()
    _segments: Generator[AudioSegment, None, None]

    def __init__(
        self,
        audio: pxt.Audio,
        *,
        duration: float | None = None,
        max_size: int | None = None,
        overlap: float = 0.0,
        min_segment_duration: float = 0.0,
        min_silence_len: float | None = None,
        silence_thresh: float = -40.0,
        trim_leading_silence: bool = False,
    ):
        assert (duration is None) != (max_size is None)
        assert overlap >= 0.0
        assert min_segment_duration >= 0.0
        audio_path = Path(audio)
        assert audio_path.exists() and audio_path.is_file()
        self.audio_path = audio_path
        self.overlap = overlap
        self.min_segment_duration = min_segment_duration
        self.trim_leading_silence = trim_leading_silence
        self.with_silence = min_silence_len is not None
        self.decode_needed = self.with_silence or trim_leading_silence
        self.silence_threshold_meansq = 10.0 ** (silence_thresh / 10.0)
        self.container = av.open(str(audio_path))
        # AudioType.validate_media() rejects files without an audio stream before they can reach the iterator
        assert len(self.container.streams.audio) > 0
        self.in_stream = self.container.streams.audio[0]
        self.audio_time_base = self.in_stream.time_base
        if self.with_silence:
            self.min_silence_len_pts = round(min_silence_len / self.audio_time_base)

        if max_size is not None:
            self.max_size = max_size
            self._segments = self._iter_segments(size_mode=True)
        else:
            assert duration > 0.0
            assert duration >= min_segment_duration
            assert overlap < duration
            self.segment_duration = duration
            self._segments = self._iter_segments(size_mode=False)

    def __next__(self) -> AudioSegment:
        return next(self._segments)

    def close(self) -> None:
        self._segments.close()
        self.container.close()

    def _iter_segments(self, *, size_mode: bool) -> Generator[AudioSegment, None, None]:
        """
        Streaming segmenter for both modes. It pulls packets until the segment reaches its budget (bytes in size mode,
        elapsed time in duration mode), then optionally moves the end to a silence and trims leading silence.
        """
        overlap_pts = round(self.overlap / self.audio_time_base)
        min_duration_pts = round(self.min_segment_duration / self.audio_time_base)
        duration_pts = 0 if size_mode else round(self.segment_duration / self.audio_time_base)
        packets = self.container.demux(self.in_stream)

        # plain duration (w/o silence) mode anchors segments to the absolute grid k * duration rather than to each
        # segment's own start: the summed target and actual emitted lengths drive the segmentation so per-packet
        # rounding overshoot can't accumulate into boundary drift;
        # silence and trim modes instead bound each segment relative to its own start
        cumulative_target_pts = 0
        cumulative_emitted_pts = 0

        # carryover for the next segment: the overlap tail, packets past the silence cut, any size-trimmed packets,
        # and the one lookahead packet that did not fit the current segment
        pending_packet_info: list[_PacketInfo] = []
        eof = False
        prev_start_pts: int | None = None

        # in size mode, reserve room for the container header and trailer that the packet-size sum doesn't account
        # for; seeded from the overhead observed so far so that after the first segment we usually fit in one remux
        overhead_reserve = 0

        while True:
            segment = list(pending_packet_info)
            segment_bytes = sum(p.packet.size for p in segment)
            segment_start_pts = segment[0].pts if len(segment) > 0 else None
            pending_packet_info = []
            scanner = (
                _SilenceScanner(self.silence_threshold_meansq, self.min_silence_len_pts, min_duration_pts)
                if self.with_silence
                else None
            )
            if scanner is not None:
                for idx, packet_info in enumerate(segment):
                    scanner.observe(
                        idx, packet_info.pts, packet_info.pts + packet_info.duration, packet_info.mean_square
                    )
            lookahead: _PacketInfo | None = None

            while not eof:
                try:
                    packet = next(packets)
                except StopIteration:
                    eof = True
                    break
                if packet.size == 0:
                    # flush sentinel emitted at the end of demux(); nothing to remux
                    continue

                if segment_start_pts is None:
                    segment_start_pts = packet.pts
                is_full: bool
                if size_mode:
                    is_full = len(segment) > 0 and segment_bytes + packet.size > self.max_size - overhead_reserve
                elif self.decode_needed:
                    # silence/trim mode: bound each segment by duration measured from its own start, so the latest
                    # silence within that window becomes the split point and a segment never runs past the budget
                    is_full = len(segment) > 0 and packet.pts - segment_start_pts >= duration_pts
                else:
                    # plain duration mode: anchor split points to the absolute grid (k * duration) by comparing the
                    # total emitted length against the total target, so per-packet rounding overshoot can't accumulate
                    elapsed_pts = cumulative_emitted_pts + (packet.pts - segment_start_pts)
                    is_full = len(segment) > 0 and elapsed_pts >= cumulative_target_pts + duration_pts
                if is_full:
                    # this packet starts the next segment
                    lookahead = self._make_packet_info(packet)
                    break

                packet_info = self._make_packet_info(packet)
                segment.append(packet_info)
                segment_bytes += packet.size
                if scanner is not None:
                    scanner.observe(
                        len(segment) - 1,
                        packet_info.pts,
                        packet_info.pts + packet_info.duration,
                        packet_info.mean_square,
                    )

            if len(segment) == 0:
                return

            prefix, leftover = self._finalize_segment(segment, scanner)
            if len(prefix) == 0:
                # the segment was entirely leading silence; drop it and carry the rest forward
                pending_packet_info = leftover + ([lookahead] if lookahead is not None else [])
                if eof and len(pending_packet_info) == 0:
                    return
                continue

            if size_mode:
                # the reserve is only an estimate, so remux and trim to enforce the exact limit
                segment_file, kept, trimmed = self._remux_measure_trim(prefix)
                segment_size = Path(segment_file).stat().st_size
                overhead_reserve = max(overhead_reserve, segment_size - sum(p.packet.size for p in kept))
            else:
                segment_file = self._remux(prefix)
                kept, trimmed = prefix, []
            segment_start_pts = kept[0].pts
            segment_end_pts = kept[-1].pts + kept[-1].duration

            if prev_start_pts is not None and segment_start_pts <= prev_start_pts:
                budget_name = '`max_size`' if size_mode else '`duration`'
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'`overlap` is too large relative to {budget_name}; segments cannot advance',
                )
            prev_start_pts = segment_start_pts

            more_content = (not eof) or lookahead is not None or len(trimmed) > 0 or len(leftover) > 0
            if not more_content and (segment_end_pts - segment_start_pts) < min_duration_pts:
                # drop the final segment if it is shorter than min_segment_duration
                return
            if more_content:
                # select by packet end, not start: a packet that begins before the overlap window but extends into
                # it still covers part of the overlap, and testing pts alone would drop it (and zero out the overlap
                # entirely when a packet is longer than overlap_pts)
                overlap_tail = [p for p in kept if p.pts + p.duration > segment_end_pts - overlap_pts]
                pending_packet_info = overlap_tail + trimmed + leftover + ([lookahead] if lookahead is not None else [])

            cumulative_target_pts += duration_pts
            cumulative_emitted_pts += segment_end_pts - segment_start_pts

            result: AudioSegment = {
                'segment_start': round(float(segment_start_pts * self.audio_time_base), 4),
                'segment_end': round(float(segment_end_pts * self.audio_time_base), 4),
                'audio_segment': segment_file,
            }
            _logger.debug('audio segment result: %s', result)
            yield result
            if not more_content:
                return

    def _packet_energy(self, packet: av.Packet) -> float:
        """Normalized mean-square amplitude of the packet, used to classify it as silent or audible"""
        sumsq = 0.0
        count = 0
        for frame in packet.decode():
            assert isinstance(frame, av.AudioFrame)
            arr = frame.to_ndarray()
            full_scale = float(np.iinfo(arr.dtype).max) + 1.0 if np.issubdtype(arr.dtype, np.integer) else 1.0
            sumsq += float(np.square(arr.astype(np.float64) / full_scale).sum())
            count += arr.size
        return sumsq / count if count > 0 else 0.0

    def _make_packet_info(self, packet: av.Packet) -> _PacketInfo:
        mean_square = self._packet_energy(packet) if self.decode_needed else -1.0
        return _PacketInfo(packet, packet.pts, packet.dts, packet.duration, mean_square)

    def _finalize_segment(
        self, segment: list[_PacketInfo], scanner: _SilenceScanner | None
    ) -> tuple[list[_PacketInfo], list[_PacketInfo]]:
        # Choose the cut: move to the latest qualifying silence if there is one, otherwise keep the whole accumulated
        # segment. Then optionally drop leading silence. Returns the packets to emit and the packets past the cut that
        # belong to the next segment.
        cut = scanner.cut_index if scanner is not None and scanner.cut_index is not None else len(segment) - 1
        prefix = segment[: cut + 1]
        leftover = segment[cut + 1 :]
        if self.trim_leading_silence:
            lead = 0
            while lead < len(prefix) and prefix[lead].mean_square <= self.silence_threshold_meansq:
                lead += 1
            prefix = prefix[lead:]
        return prefix, leftover

    def _remux_measure_trim(self, segment: list[_PacketInfo]) -> tuple[str, list[_PacketInfo], list[_PacketInfo]]:
        # Remux seg into a segment file, dropping trailing packets until the file fits within max_size. Returns the
        # segment file, the packets it contains, and the trimmed-off packets (in ascending order) for the next segment.
        kept = list(segment)
        trimmed: list[_PacketInfo] = []
        while True:
            segment_file = self._remux(kept)
            if Path(segment_file).stat().st_size <= self.max_size:
                return segment_file, kept, trimmed
            # this remux is too big; discard
            TempStore.delete_media_file(Path(segment_file))
            if len(kept) == 1:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'`max_size` ({self.max_size} bytes) is too small to hold a single packet of audio file '
                    f"'{self.audio_path}'",
                )
            trimmed.insert(0, kept.pop())

    def _remux(self, packet_info: list[_PacketInfo]) -> str:
        segment_file = str(TempStore.create_path(extension=self.audio_path.suffix))
        start_pts = packet_info[0].pts
        with av.open(segment_file, mode='w') as output_container:
            output_stream = output_container.add_stream_from_template(self.in_stream)
            for info in packet_info:
                packet = info.packet
                packet.stream = output_stream
                # rebase timestamps to the segment start, derived from the stored originals since muxing mutates them
                packet.pts = info.pts - start_pts
                packet.dts = None if info.dts is None else info.dts - start_pts
                packet.duration = info.duration
                output_container.mux(packet)
        return segment_file

    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        duration = bound_args.get('duration')
        max_size = bound_args.get('max_size')
        overlap = bound_args.get('overlap')
        min_segment_duration = bound_args.get('min_segment_duration')
        min_silence_len = bound_args.get('min_silence_len')
        silence_thresh = bound_args.get('silence_thresh')

        if (duration is None) == (max_size is None):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, 'Exactly one of `duration` or `max_size` must be specified'
            )
        if overlap is not None and overlap < 0.0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '`overlap` must be non-negative')
        if min_segment_duration is not None and min_segment_duration < 0.0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '`min_segment_duration` must be non-negative')
        if min_silence_len is not None and min_silence_len <= 0.0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '`min_silence_len` must be a positive number')
        if silence_thresh is not None and silence_thresh >= 0.0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '`silence_thresh` must be negative (dBFS)')

        if duration is not None:
            if duration <= 0.0:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '`duration` must be a positive number')
            if min_segment_duration is not None and duration < min_segment_duration:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, '`duration` must be at least `min_segment_duration`'
                )
            if overlap is not None and overlap >= duration:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, '`overlap` must be strictly less than `duration`'
                )
        elif max_size <= 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, '`max_size` must be a positive number of bytes')


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
