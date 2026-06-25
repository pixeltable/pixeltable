import math
import os
from pathlib import Path
from typing import Callable

import av
import numpy as np
import pytest

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.functions import util as functions_util
from pixeltable.functions.audio import audio_splitter, encode_audio
from pixeltable.utils import av as av_utils

from .utils import (
    MediaStore,
    ReloadTester,
    TempStoreView,
    get_audio_file,
    get_audio_files,
    get_video_files,
    pxt_raises,
    rerun,
    skip_test_if_not_installed,
    validate_update_status,
)


class TestAudio:
    def _validate_audio(self, audio_files: list[str]) -> None:
        """Confirm each file is valid audio by inserting into a table with on_error='abort'."""
        t = pxt.create_table('validated_audio', schema={'a': pxt.Audio}, if_exists='ignore')
        validate_update_status(
            t.insert(({'a': a} for a in audio_files), on_error='abort'), expected_rows=len(audio_files)
        )

    def check_audio_params(self, path: str, format: str | None = None, codec: str | None = None) -> None:
        with av.open(path) as container:
            audio_stream = container.streams.audio[0]
            if format == 'mp4':
                # mov, the demuxer for mp4, happens to return a string such as 'mov,mp4,m4a,3gp,3g2,mj2'
                assert 'mp4' in container.format.name
            elif format is not None:
                assert format == container.format.name
            if codec is not None:
                assert codec == audio_stream.codec_context.codec.name

    def test_basic(self, uses_db: None) -> None:
        audio_filepaths = get_audio_files()
        audio_t = pxt.create_table('audio', {'audio_file': pxt.Audio})
        status = audio_t.insert({'audio_file': p} for p in audio_filepaths)
        assert status.num_rows == len(audio_filepaths)
        assert status.num_excs == 0
        paths = audio_t.select(output=audio_t.audio_file.localpath).collect()['output']
        assert set(paths) == set(audio_filepaths)

    def test_extract(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        video_filepaths = get_video_files()
        video_t = pxt.create_table(p('videos'), {'video': pxt.Video})
        video_t.add_computed_column(audio=video_t.video.extract_audio())

        # Directly count the number of videos with audio streams, without relying on the UDF
        videos_with_audio = 0
        for path in video_filepaths:
            md = functions_util.get_metadata(path)
            if sum(1 for stream in md['streams'] if stream['type'] == 'audio') > 0:
                videos_with_audio += 1

        validate_update_status(
            video_t.insert({'video': path} for path in video_filepaths), expected_rows=len(video_filepaths)
        )
        assert MediaStore.count(video_t, default_output_dest=True) == videos_with_audio
        assert video_t.where(video_t.audio != None).count() == videos_with_audio
        tmp_files_before = TempStoreView.count(video_t)

        video_t = pxt.get_table(p('videos'))
        assert video_t.where(video_t.audio != None).count() == videos_with_audio

        # test generating different formats and codecs
        paths = video_t.select(output=video_t.video.extract_audio(format='wav', codec='pcm_s16le')).collect()['output']
        # media files that are created as a part of a query end up in the tmp dir
        assert TempStoreView.count(video_t) == tmp_files_before + video_t.where(video_t.audio != None).count()
        for path in [pth for pth in paths if pth is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s16le')
        # higher resolution
        paths = video_t.select(output=video_t.video.extract_audio(format='wav', codec='pcm_s32le')).collect()['output']
        for path in [pth for pth in paths if pth is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s32le')

        for format in av_utils.AUDIO_FORMATS.keys() - 'wav':
            paths = video_t.select(output=video_t.video.extract_audio(format=format)).collect()['output']
            for path in [pth for pth in paths if pth is not None]:
                self.check_audio_params(path, format=format)

    def test_get_metadata(self, uses_db: None) -> None:
        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        base_t.add_computed_column(metadata=base_t.audio.get_metadata())
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        result = base_t.where(base_t.metadata.size == 2568827).select(base_t.metadata).collect()['metadata'][0]
        assert result == {
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

        # get_metadata() returns correct type information
        expr = base_t.audio.get_metadata().streams[0].duration_seconds
        assert expr.col_type.is_float_type()
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="cannot resolve 'not_an_attr'"):
            _ = base_t.audio.get_metadata().streams[0].not_an_attr

    def __has_audio(self, path: str) -> bool:
        with av.open(path) as container:
            return len(container.streams.audio) > 0

    def __assert_tiling(self, segments: list[dict], *, overlap: float) -> None:
        # segments for a single source, in ascending start order, must tile it: positive spans, strictly advancing
        # starts, and no gap between consecutive segments (overlapping when overlap is requested)
        assert len(segments) > 0
        assert all(seg['segment_end'] > seg['segment_start'] for seg in segments)
        assert all(segments[i + 1]['segment_start'] > segments[i]['segment_start'] for i in range(len(segments) - 1))
        assert all(
            segments[i + 1]['segment_start'] <= segments[i]['segment_end'] + 1e-3 for i in range(len(segments) - 1)
        )
        if overlap > 0.0:
            assert all(segments[i + 1]['segment_start'] < segments[i]['segment_end'] for i in range(len(segments) - 1))

    def test_audio_splitter_on_audio(self, uses_db: None, reload_tester: ReloadTester) -> None:
        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        audio_segment_view = pxt.create_view(
            'audio_segments',
            base_t,
            iterator=audio_splitter(audio=base_t.audio, duration=5.0, overlap=1.25, min_segment_duration=0.5),
        )
        results = reload_tester.run_query(
            audio_segment_view.order_by(audio_segment_view.audio, audio_segment_view.segment_start).select(
                audio_segment_view.audio, audio_segment_view.segment_start, audio_segment_view.segment_end
            )
        )
        segments_by_file: dict[str, list[dict]] = {}
        for result in results:
            segments_by_file.setdefault(result['audio'], []).append(result)
        # every source that has an audio stream is split into a tiling of overlapping segments
        assert set(segments_by_file) == {file for file in audio_filepaths if self.__has_audio(file)}
        for segments in segments_by_file.values():
            self.__assert_tiling(segments, overlap=1.25)
        reload_tester.run_reload_test()

    def test_audio_splitter_on_videos_revert_media_store(
        self, make_catalog_path: Callable[[str], str], reload_tester: ReloadTester
    ) -> None:
        p = make_catalog_path
        video_filepaths = get_video_files()
        video_t = pxt.create_table(p('videos'), {'video': pxt.Video})
        video_t.insert({'video': path} for path in video_filepaths)

        pre_count = MediaStore.count(video_t, default_output_dest=True)
        # extract audio
        video_t.add_computed_column(audio=video_t.video.extract_audio(format='mp3'))
        post_count = MediaStore.count(video_t, default_output_dest=True)
        assert post_count > pre_count  # Some files should have been added

        video_t.revert()
        final_count = MediaStore.count(video_t, default_output_dest=True)
        assert final_count == pre_count  # Reverting should remove the added files

    def test_audio_splitter_single_file(self, uses_db: None, reload_tester: ReloadTester) -> None:
        audio_filepath = get_audio_file('jfk_1961_0109_cityuponahill-excerpt.flac')  # 60s audio file
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))
        audio_segment_view = pxt.create_view(
            'audio_segments',
            base_t,
            iterator=audio_splitter(audio=base_t.audio, duration=5.0, overlap=0.0, min_segment_duration=0.0),
        )
        # a 60s file split into non-overlapping 5s segments yields 12 segments landing exactly on the k * duration
        # grid (no drift) and covering the whole file
        assert audio_segment_view.count() == 12
        results = reload_tester.run_query(audio_segment_view.order_by(audio_segment_view.pos))
        assert all(result['audio'] == audio_filepath for result in results)
        assert results[-1]['segment_end'] == 60
        assert all(math.floor(results[i]['segment_start']) == i * 5.0 for i in range(len(results)))
        assert all(
            round(results[i]['segment_end'] - results[i]['segment_start']) == 5.0 for i in range(len(results) - 1)
        )

        # 60s / 7s leaves a 4s trailing segment; min_segment_duration=5 drops it, so 8 segments end at 56 rather than 9
        drop_view = pxt.create_view(
            'audio_segments_drop',
            base_t,
            iterator=audio_splitter(audio=base_t.audio, duration=7.0, min_segment_duration=5.0),
        )
        assert drop_view.count() == 8
        drop_results = drop_view.order_by(drop_view.pos).select(drop_view.segment_end).collect()
        assert math.floor(drop_results[-1]['segment_end']) == 56
        reload_tester.run_reload_test()

    def test_audio_splitter_overlap_smaller_than_packet(self, uses_db: None) -> None:
        # sample.flac has ~0.096s packets; an overlap smaller than a single packet must still be honored. Selecting
        # overlap packets by start timestamp drops the final packet (its start precedes the overlap window) and would
        # yield contiguous, non-overlapping segments.
        audio_filepath = get_audio_file('sample.flac')
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))
        view = pxt.create_view(
            'audio_segments', base_t, iterator=audio_splitter(audio=base_t.audio, duration=1.0, overlap=0.05)
        )
        results = view.order_by(view.pos).select(view.segment_start, view.segment_end).collect()
        assert len(results) > 1
        # consecutive segments overlap in time despite the requested overlap being smaller than one packet
        assert all(results[i + 1]['segment_start'] < results[i]['segment_end'] for i in range(len(results) - 1))

    def __frame_count(self, path: str) -> int:
        with av.open(path) as container:
            return sum(1 for _ in container.decode(audio=0))

    def test_audio_splitter_max_size(self, uses_db: None, reload_tester: ReloadTester) -> None:
        # exercise the byte-driven packing across every container layout and codec in the fixtures
        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))

        max_size = 128 * 1024
        view = pxt.create_view('audio_segments', base_t, iterator=audio_splitter(audio=base_t.audio, max_size=max_size))
        results = reload_tester.run_query(
            view.order_by(view.audio, view.pos).select(
                view.audio, view.segment_start, view.segment_end, path=view.audio_segment.localpath
            )
        )
        # every emitted segment stays within the byte budget
        assert all(os.path.getsize(r['path']) <= max_size for r in results)
        # every segment spans a positive amount of time
        assert all(r['segment_end'] > r['segment_start'] for r in results)
        # each source produces more than one segment for this budget
        assert all(sum(1 for r in results if r['audio'] == src) > 1 for src in audio_filepaths)
        # with no overlap, the segments together cover each source exactly: no audio is lost or duplicated
        assert all(
            sum(self.__frame_count(r['path']) for r in results if r['audio'] == src) == self.__frame_count(src)
            for src in audio_filepaths
        )
        reload_tester.run_reload_test()

    def test_audio_splitter_max_size_overlap(self, uses_db: None) -> None:
        audio_filepath = get_audio_file('sample.flac')
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))

        max_size = 128 * 1024
        view = pxt.create_view(
            'audio_segments', base_t, iterator=audio_splitter(audio=base_t.audio, max_size=max_size, overlap=0.5)
        )
        results = (
            view.order_by(view.pos)
            .select(view.segment_start, view.segment_end, path=view.audio_segment.localpath)
            .collect()
        )
        assert all(os.path.getsize(r['path']) <= max_size for r in results)
        # consecutive segments overlap in time
        assert all(results[i + 1]['segment_start'] < results[i]['segment_end'] for i in range(len(results) - 1))
        # overlap means the segments together cover more audio than the source
        total_frames = sum(self.__frame_count(r['path']) for r in results)
        assert total_frames > self.__frame_count(audio_filepath)

    def test_audio_splitter_max_size_errors(self, uses_db: None) -> None:
        audio_filepath = get_audio_file('sample.flac')
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))

        # These conditions depend on the packet sizes of the actual audio, so they are detected during view
        # population rather than by the static validate() hook; population surfaces them as an aborted table op.

        # max_size too small to hold even a single packet
        with pxt_raises(pxt.ErrorCode.INTERNAL_ERROR, match=r'too small to hold a single packet'):
            _ = pxt.create_view(
                'audio_segments_tiny', base_t, iterator=audio_splitter(audio=base_t.audio, max_size=1024)
            )

        # overlap so large relative to max_size that segments cannot advance
        with pxt_raises(pxt.ErrorCode.INTERNAL_ERROR, match=r'`overlap` is too large relative to `max_size`'):
            _ = pxt.create_view(
                'audio_segments_overlap',
                base_t,
                iterator=audio_splitter(audio=base_t.audio, max_size=128 * 1024, overlap=100.0),
            )

    def __make_tone_silence_wav(self, path: str, tone_sec: float, silence_sec: float, cycles: int) -> None:
        # Write a mono wav alternating a 440 Hz tone with true silence, so silence boundaries are at known offsets.
        sr = 16000
        tone_samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(tone_sec * sr)) / sr) * 32767).astype(np.int16)
        silence_samples = np.zeros(int(silence_sec * sr), dtype=np.int16)
        audio = np.tile(np.concatenate([tone_samples, silence_samples]), cycles)
        with av.open(path, mode='w') as out:
            stream = out.add_stream('pcm_s16le', rate=sr)
            assert isinstance(stream, av.AudioStream)
            stream.layout = 'mono'
            frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format='s16', layout='mono')
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)

    def __edge_rms(self, path: str, *, at_start: bool) -> float:
        # rms of the first or last packet of a segment, normalized to full scale; a packet is the granularity at
        # which silence detection and leading-trim operate
        with av.open(path) as c:
            packets = [p for p in c.demux(c.streams.audio[0]) if p.size > 0]
            target = packets[0] if at_start else packets[-1]
            chunks: list[np.ndarray] = []
            for fr in target.decode():
                assert isinstance(fr, av.AudioFrame)
                chunks.append(fr.to_ndarray().reshape(-1))
            samples = np.concatenate(chunks).astype(np.float64)
        samples /= 32768.0
        return float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0

    def test_audio_splitter_silence(self, uses_db: None, tmp_path: Path, reload_tester: ReloadTester) -> None:
        # 10 cycles of 0.8s tone + 0.4s silence = 12s, with silence gaps every 1.2s
        audio_filepath = str(tmp_path / 'tone_silence.wav')
        self.__make_tone_silence_wav(audio_filepath, tone_sec=0.8, silence_sec=0.4, cycles=10)
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))

        view = pxt.create_view(
            'audio_segments',
            base_t,
            iterator=audio_splitter(audio=base_t.audio, duration=3.0, min_silence_len=0.2, silence_thresh=-40.0),
        )
        results = reload_tester.run_query(
            view.order_by(view.pos).select(view.segment_start, view.segment_end, path=view.audio_segment.localpath)
        )
        # every segment ends in silence rather than mid-tone, and stays within the duration budget
        assert all(self.__edge_rms(r['path'], at_start=False) < 0.05 for r in results)
        assert all(r['segment_end'] - r['segment_start'] <= 3.0 + 0.1 for r in results)
        # the segments are contiguous and cover the whole clip
        assert results[0]['segment_start'] == 0.0
        assert results[-1]['segment_end'] == 12.0
        assert all(results[i + 1]['segment_start'] == results[i]['segment_end'] for i in range(len(results) - 1))
        reload_tester.run_reload_test()

        # silence-aware cutting also works under a byte budget
        size_view = pxt.create_view(
            'audio_segments_size',
            base_t,
            iterator=audio_splitter(audio=base_t.audio, max_size=128 * 1024, min_silence_len=0.2, silence_thresh=-40.0),
        )
        size_results = size_view.order_by(size_view.pos).select(path=size_view.audio_segment.localpath).collect()
        assert all(os.path.getsize(r['path']) <= 128 * 1024 for r in size_results)
        assert all(self.__edge_rms(r['path'], at_start=False) < 0.05 for r in size_results)

    def test_audio_splitter_trim_leading_silence(self, uses_db: None, tmp_path: Path) -> None:
        # fixed-duration cuts land mid-cycle, so without trimming some segments would start in a silence gap
        audio_filepath = str(tmp_path / 'tone_silence.wav')
        self.__make_tone_silence_wav(audio_filepath, tone_sec=0.8, silence_sec=0.4, cycles=10)
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))

        view = pxt.create_view(
            'audio_segments',
            base_t,
            iterator=audio_splitter(audio=base_t.audio, duration=1.0, silence_thresh=-40.0, trim_leading_silence=True),
        )
        results = view.order_by(view.pos).select(path=view.audio_segment.localpath).collect()
        # every emitted segment begins at audible content
        assert len(results) > 0
        assert all(self.__edge_rms(r['path'], at_start=True) > 0.05 for r in results)

    def test_audio_splitter_trim_leading_silence_all_silent(self, uses_db: None, tmp_path: Path) -> None:
        # 1s tone, then 4s of silence, then 1s tone: fixed-duration segments that fall entirely in the gap are dropped
        audio_filepath = str(tmp_path / 'gap.wav')
        sr = 16000
        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 32767).astype(np.int16)
        audio = np.concatenate([tone, np.zeros(4 * sr, dtype=np.int16), tone])
        with av.open(audio_filepath, mode='w') as out:
            stream = out.add_stream('pcm_s16le', rate=sr)
            assert isinstance(stream, av.AudioStream)
            stream.layout = 'mono'
            frame = av.AudioFrame.from_ndarray(audio.reshape(1, -1), format='s16', layout='mono')
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))

        view = pxt.create_view(
            'audio_segments',
            base_t,
            iterator=audio_splitter(audio=base_t.audio, duration=1.0, silence_thresh=-40.0, trim_leading_silence=True),
        )
        results = view.order_by(view.pos).select(path=view.audio_segment.localpath).collect()
        # no emitted segment is entirely silent: every one begins at audible content
        assert len(results) > 0
        assert all(self.__edge_rms(r['path'], at_start=True) > 0.05 for r in results)

    def test_create_audio_splitter(self, uses_db: None) -> None:
        audio_filepath = get_audio_file('jfk_1961_0109_cityuponahill-excerpt.flac')  # 60s audio file
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`duration` must be a positive number'):
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(audio=base_t.audio, duration=-1, overlap=1, min_segment_duration=1),
            )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`duration` must be at least `min_segment_duration`'):
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(audio=base_t.audio, duration=1, overlap=0, min_segment_duration=2),
            )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`overlap` must be strictly less than `duration`'):
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(audio=base_t.audio, duration=1, overlap=1, min_segment_duration=0),
            )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`overlap` must be non-negative'):
            _ = pxt.create_view(
                'audio_segments', base_t, iterator=audio_splitter(audio=base_t.audio, duration=5.0, overlap=-1.0)
            )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`min_segment_duration` must be non-negative'):
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(audio=base_t.audio, duration=5.0, min_segment_duration=-1.0),
            )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'Exactly one of `duration` or `max_size`'):
            _ = pxt.create_view('audio_segments', base_t, iterator=audio_splitter(audio=base_t.audio))

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'Exactly one of `duration` or `max_size`'):
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(audio=base_t.audio, duration=5.0, max_size=1024 * 1024),
            )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`max_size` must be a positive number of bytes'):
            _ = pxt.create_view('audio_segments', base_t, iterator=audio_splitter(audio=base_t.audio, max_size=0))

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`min_silence_len` must be a positive number'):
            _ = pxt.create_view(
                'audio_segments', base_t, iterator=audio_splitter(audio=base_t.audio, duration=5.0, min_silence_len=0)
            )

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`silence_thresh` must be negative'):
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(audio=base_t.audio, duration=5.0, min_silence_len=0.2, silence_thresh=0.0),
            )

    @pytest.mark.parametrize(
        'format,stereo,downsample,as_1d_array',
        [
            ('wav', False, False, False),  # wav_mono
            ('mp3', True, False, False),  # mp3_stereo
            ('mp3', False, False, False),  # mp3_mono
            ('mp3', True, True, False),  # mp3_downsample_stereo
            ('flac', True, False, False),  # flac_stereo
            ('mp4', True, False, False),  # mp4_stereo
            ('mp4', False, False, True),  # mp4_1d_array_mono
        ],
        ids=[
            'wav_mono',
            'mp3_stereo',
            'mp3_mono',
            'mp3_downsample_stereo',
            'flac_stereo',
            'mp4_stereo',
            'mp4_1d_array_mono',
        ],
    )
    def test_encode_array_to_audio(
        self, format: str, stereo: bool, downsample: bool, as_1d_array: bool, uses_db: None
    ) -> None:
        # Load a sample mp3 file to an array
        sample_path = './tests/data/audio/sample.mp3' if stereo else './docs/resources/10-minute tour of Pixeltable.mp3'
        audio_data, sample_duration_sec, sample_rate = self._decode_audio_file(sample_path)
        assert audio_data.dtype == np.float32
        assert audio_data.ndim == 2
        assert audio_data.shape[0] == 2 if stereo else 1
        if as_1d_array:
            # Validate the scenario in which the input is a mono clip as an (N)-shaped array, as opposed to (1, N)
            assert not stereo
            audio_data = audio_data.flatten()

        # Use encode_audio to encode it to an audio file
        t = pxt.create_table('test_encode_array_to_audio', {'audio_array': pxt.Array[pxt.Float]})  # type: ignore[misc]
        output_sample_rate = sample_rate // 2 if downsample else sample_rate
        t.add_computed_column(
            audio_file=encode_audio(
                t.audio_array, input_sample_rate=sample_rate, format=format, output_sample_rate=output_sample_rate
            )
        )
        validate_update_status(t.insert(audio_array=audio_data), 1)

        row = t.head(1)[0]
        assert set(row.keys()) == {'audio_array', 'audio_file'}
        encoded_path = row['audio_file']
        assert encoded_path.endswith('.m4a' if format == 'mp4' else f'.{format}')
        print(f'Encoded audio file: {row["audio_file"]}')

        # Read back, decode, and validate the encoded file
        _, encoded_duration_sec, encoded_sample_rate = self._decode_audio_file(encoded_path)
        assert abs(encoded_duration_sec - sample_duration_sec) < 1
        assert encoded_sample_rate == output_sample_rate

    def _decode_audio_file(self, file_path: str) -> tuple[np.ndarray, float, int]:
        with av.open(file_path) as container:
            assert len(container.streams.audio) == 1
            audio_stream = container.streams.audio[0]
            duration_seconds = float(audio_stream.duration * audio_stream.time_base)
            sample_rate = audio_stream.rate
            audio_frames = [frame.to_ndarray() for frame in container.decode(audio_stream)]

        audio_data = np.concatenate(audio_frames, axis=1)
        assert len(audio_data) > 0
        return audio_data, duration_seconds, sample_rate

    @pytest.mark.very_expensive  # Downloads a Hugging Face dataset
    @rerun(reruns=3, reruns_delay=15)  # Guard against connection errors downloading datasets
    def test_encode_dataset_audio(self, uses_db: None) -> None:
        """
        The point of this test case is to validate encode_audio UDF on a real-world dataset.
        """
        skip_test_if_not_installed('datasets')
        import datasets  # type: ignore[import-untyped]

        hf_dataset = datasets.load_dataset('Hani89/medical_asr_recording_dataset', split='test')
        t = pxt.create_table('test_encode_dataset_audio', source=hf_dataset)
        row = t.head(1)[0]
        assert set(row.keys()) == {'audio', 'sentence'}
        assert isinstance(row['audio'], dict)
        assert set(row['audio'].keys()) == {'array', 'path', 'sampling_rate'}
        assert isinstance(row['audio']['array'], np.ndarray)
        assert isinstance(row['audio']['sampling_rate'], int)

        update_status = t.add_computed_column(
            audio_file=encode_audio(
                t.audio.array.astype(pxt.Array[pxt.Float]),  # type: ignore[misc]
                input_sample_rate=t.audio.sampling_rate.astype(pxt.Int),
                format='flac',
            )
        )
        validate_update_status(update_status)
        for row in t.head(10):
            assert set(row.keys()) == {'audio', 'sentence', 'audio_file'}
            print(f'Encoded audio file: {row["audio_file"]}')

    @pytest.mark.parametrize(
        'make_columns',
        [
            pytest.param(
                lambda audio: {
                    'louder': audio.multiply_volume(factor=2.0),
                    'quieter': audio.multiply_volume(factor=0.5),
                    'partial': audio.multiply_volume(factor=3.0, start_time=1.0, end_time=3.0),
                    'from_start': audio.multiply_volume(factor=2.0, end_time=3.0),
                    'to_end': audio.multiply_volume(factor=2.0, start_time=1.0),
                },
                id='multiply_volume',
            ),
            pytest.param(lambda audio: {'faded': audio.fade_in(duration=2.0)}, id='fade_in'),
            pytest.param(lambda audio: {'faded': audio.fade_out(duration=2.0)}, id='fade_out'),
            pytest.param(lambda audio: {'normed': audio.normalize()}, id='normalize'),
        ],
    )
    def test_audio_effects(
        self, make_columns: Callable[[exprs.ColumnRef], dict[str, exprs.Expr]], uses_db: None
    ) -> None:
        audio_paths = get_audio_files()
        t = pxt.create_table('test_audio', {'audio': pxt.Audio})
        columns = make_columns(t.audio)
        for name, expr in columns.items():
            t.add_computed_column(**{name: expr})
        validate_update_status(t.insert({'audio': p} for p in audio_paths), expected_rows=len(audio_paths))
        result = t.select(*(getattr(t, name) for name in columns)).collect()
        # the effect produces a valid audio file for every input and every output column
        assert all(row[name] is not None for row in result for name in columns)
        outputs: list[str] = []
        for name in columns:
            outputs += result[name]
        self._validate_audio(outputs)

    def test_multiply_volume_errors(self, uses_db: None) -> None:
        audio_paths = get_audio_files()
        t = pxt.create_table('test_audio', {'audio': pxt.Audio})
        validate_update_status(t.insert({'audio': p} for p in audio_paths), expected_rows=len(audio_paths))

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`start_time` must be non-negative'):
            t.select(t.audio.multiply_volume(factor=1.0, start_time=-1.0)).collect()
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`end_time` must be non-negative'):
            t.select(t.audio.multiply_volume(factor=1.0, end_time=-1.0)).collect()
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`start_time` must be non-negative'):
            t.select(t.audio.multiply_volume(factor=1.0, start_time=-1.0, end_time=3.0)).collect()
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`end_time` must be non-negative'):
            t.select(t.audio.multiply_volume(factor=1.0, start_time=0.0, end_time=-1.0)).collect()
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`end_time` .* must be greater than `start_time`'):
            t.select(t.audio.multiply_volume(factor=1.0, start_time=5.0, end_time=3.0)).collect()
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`end_time` .* must be greater than `start_time`'):
            t.select(t.audio.multiply_volume(factor=1.0, start_time=5.0, end_time=5.0)).collect()

    @pytest.mark.parametrize(
        'make_expr',
        [
            pytest.param(lambda audio, duration: audio.fade_in(duration=duration), id='fade_in'),
            pytest.param(lambda audio, duration: audio.fade_out(duration=duration), id='fade_out'),
        ],
    )
    def test_audio_fade_errors(self, make_expr: Callable[[exprs.ColumnRef, float], exprs.Expr], uses_db: None) -> None:
        audio_paths = get_audio_files()
        t = pxt.create_table('test_audio', {'audio': pxt.Audio})
        validate_update_status(t.insert({'audio': p} for p in audio_paths), expected_rows=len(audio_paths))
        for bad_duration in (0, -1.0):
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r'`duration` must be positive'):
                t.select(make_expr(t.audio, bad_duration)).collect()

    def test_encode_audio_errors(self, uses_db: None) -> None:
        # invalid format
        t = pxt.create_table('test_encode', {'audio_array': pxt.Array[pxt.Float]})  # type: ignore[misc]
        t.insert(audio_array=np.zeros(100, dtype=np.float32))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'Only the following formats are supported'):
            t.select(encode_audio(t.audio_array, input_sample_rate=44100, format='invalid')).collect()

        # invalid array shape: (3, N) is neither mono nor stereo
        t2 = pxt.create_table('test_encode2', {'audio_array': pxt.Array[pxt.Float]})  # type: ignore[misc]
        t2.insert(audio_array=np.zeros((3, 100), dtype=np.float32))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'Supported input array shapes are'):
            t2.select(encode_audio(t2.audio_array, input_sample_rate=44100, format='wav')).collect()
