import math
from typing import Callable, Counter

import av
import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable.functions.audio import audio_splitter, encode_audio
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps

from .utils import (
    IN_CI,
    ReloadTester,
    get_audio_file,
    get_audio_files,
    get_video_files,
    rerun,
    skip_test_if_not_installed,
    validate_update_status,
)


class TestAudio:
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

    def test_extract(self, uses_db: None) -> None:
        video_filepaths = get_video_files()
        video_t = pxt.create_table('videos', {'video': pxt.Video})
        video_t.add_computed_column(audio=video_t.video.extract_audio())

        # Directly count the number of videos with audio streams, without relying on the UDF
        videos_with_audio = 0
        for p in video_filepaths:
            md = av_utils.get_metadata(p)
            if sum(1 for stream in md['streams'] if stream['type'] == 'audio') > 0:
                videos_with_audio += 1

        validate_update_status(
            video_t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths)
        )
        assert ObjectOps.count(video_t._id, default_output_dest=True) == videos_with_audio
        assert video_t.where(video_t.audio != None).count() == videos_with_audio
        tmp_files_before = TempStore.count()

        video_t = pxt.get_table('videos')
        assert video_t.where(video_t.audio != None).count() == videos_with_audio

        # test generating different formats and codecs
        paths = video_t.select(output=video_t.video.extract_audio(format='wav', codec='pcm_s16le')).collect()['output']
        # media files that are created as a part of a query end up in the tmp dir
        assert TempStore.count() == tmp_files_before + video_t.where(video_t.audio != None).count()
        for path in [p for p in paths if p is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s16le')
        # higher resolution
        paths = video_t.select(output=video_t.video.extract_audio(format='wav', codec='pcm_s32le')).collect()['output']
        for path in [p for p in paths if p is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s32le')

        for format in av_utils.AUDIO_FORMATS.keys() - 'wav':
            paths = video_t.select(output=video_t.video.extract_audio(format=format)).collect()['output']
            for path in [p for p in paths if p is not None]:
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

    def __count_segments(
        self,
        start_time_sec: float,
        total_duration_sec: float,
        segment_duration_sec: float,
        overlap_sec: float,
        min_segment_duration_sec: float,
    ) -> int:
        effective_segment_duration_sec = segment_duration_sec - overlap_sec
        segment_count = 0
        start = start_time_sec
        end = start_time_sec + total_duration_sec
        while True:
            if start + segment_duration_sec >= end:
                last_segment_size = end - start
                if last_segment_size > 0 and last_segment_size >= min_segment_duration_sec:
                    segment_count += 1
                break
            start += effective_segment_duration_sec
            segment_count += 1
        return segment_count

    def __get_segment_count(
        self, file: str, target_segment_size_sec: float, overlap_sec: float, min_segment_duration_sec: float
    ) -> int:
        container = av.open(file)
        if len(container.streams.audio) == 0:
            return 0
        total_duration = container.streams.audio[0].duration or 0
        start_time = container.streams.audio[0].start_time or 0
        time_base = container.streams.audio[0].time_base
        return self.__count_segments(
            float(start_time * time_base),
            float(total_duration * time_base),
            target_segment_size_sec,
            overlap_sec,
            min_segment_duration_sec,
        )

    def test_audio_splitter_on_audio(self, uses_db: None, reload_tester: ReloadTester) -> None:
        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        audio_segment_view = pxt.create_view(
            'audio_segments',
            base_t,
            iterator=audio_splitter(
                audio=base_t.audio, segment_duration_sec=5.0, overlap_sec=1.25, min_segment_duration_sec=0.5
            ),
        )
        file_to_segments = {file: self.__get_segment_count(file, 5.0, 1.25, 0.5) for file in audio_filepaths}
        results = reload_tester.run_query(audio_segment_view.order_by(audio_segment_view.pos))
        file_to_segments_from_view: dict[str, int] = dict(Counter(result['audio'] for result in results))
        assert len(results) == sum(file_to_segments.values())
        for file, count in file_to_segments.items():
            assert count == file_to_segments_from_view.get(file, 0)
        reload_tester.run_reload_test()

    def test_audio_splitter_on_videos_revert_media_store(self, uses_db: None, reload_tester: ReloadTester) -> None:
        video_filepaths = get_video_files()
        video_t = pxt.create_table('videos', {'video': pxt.Video})
        video_t.insert({'video': p} for p in video_filepaths)

        pre_count = ObjectOps.count(video_t._id, default_output_dest=True)
        # extract audio
        video_t.add_computed_column(audio=video_t.video.extract_audio(format='mp3'))
        post_count = ObjectOps.count(video_t._id, default_output_dest=True)
        assert post_count > pre_count  # Some files should have been added

        print(video_t.history())
        video_t.revert()
        final_count = ObjectOps.count(video_t._id, default_output_dest=True)
        assert final_count == pre_count  # Reverting should remove the added files

    def test_audio_splitter_on_videos(self, uses_db: None, reload_tester: ReloadTester) -> None:
        video_filepaths = get_video_files()
        video_t = pxt.create_table('videos', {'video': pxt.Video})
        video_t.insert({'video': p} for p in video_filepaths)
        # extract audio
        video_t.add_computed_column(audio=video_t.video.extract_audio(format='mp3'))
        audio_segment_view = pxt.create_view(
            'audio_segments',
            video_t,
            iterator=audio_splitter(
                audio=video_t.audio, segment_duration_sec=2.0, overlap_sec=0.5, min_segment_duration_sec=0.25
            ),
        )
        audio_files = [
            result['audio'] for result in video_t.select(video_t.audio).where(video_t.audio != None).collect()
        ]
        results = reload_tester.run_query(audio_segment_view.order_by(audio_segment_view.pos))
        file_to_segments = {file: self.__get_segment_count(file, 2.0, 0.5, 0.25) for file in audio_files}
        file_to_segments_from_view: dict[str, int] = dict(Counter(result['audio'] for result in results))
        assert len(results) == sum(file_to_segments.values())
        for file, count in file_to_segments.items():
            assert count == file_to_segments_from_view.get(file, 0)
        reload_tester.run_reload_test()

    def test_audio_splitter_build_segments(self) -> None:
        build_segments: Callable = audio_splitter.decorated_callable.build_segments  # type: ignore[attr-defined]
        segments = build_segments(0, 1005, 100, 0, 10)
        assert len(segments) == self.__count_segments(0, 1005, 100, 0, 10)
        assert all((segment[1] - segment[0]) == 100 for segment in segments)
        segments = build_segments(0, 1005, 100, 10, 16)
        assert len(segments) == self.__count_segments(0, 1005, 100, 10, 16)
        assert all((segment[1] - segment[0]) == 100 for segment in segments)
        assert segments[-1][0] == 900
        assert segments[-1][1] == 1000
        segments = build_segments(0, 1005, 100, 10, 0)
        assert len(segments) == self.__count_segments(0, 1005, 100, 10, 0)
        assert all((segment[1] - segment[0]) == 100 for segment in segments[:-1])
        assert segments[-1][0] == 990
        assert segments[-1][1] == 1005
        segments = build_segments(0, 1005, 100, 0, 0)
        assert len(segments) == self.__count_segments(0, 1005, 100, 0, 0)
        assert all((segment[1] - segment[0]) == 100 for segment in segments[:-1])
        assert segments[-1][0] == 1000
        assert segments[-1][1] == 1005
        segments = build_segments(0, 1.25, 0.15, 0, 0.051)
        assert len(segments) == self.__count_segments(0, 1.25, 0.15, 0, 0.051)
        assert all(round((segment[1] - segment[0]), 2) == 0.15 for segment in segments)
        assert round(segments[-1][0], 2) == 1.05
        assert round(segments[-1][1], 2) == 1.2
        segments = build_segments(0.2, 1.25, 0.15, 0, 0.05)
        assert len(segments) == self.__count_segments(0.2, 1.25, 0.15, 0, 0.05)
        assert all(round((segment[1] - segment[0]), 2) == 0.15 for segment in segments[:-1])
        assert round(segments[-1][0], 2) == 1.4
        assert round(segments[-1][1], 2) == 1.45
        segments = build_segments(1000, 5, 100, 0, 10)
        assert len(segments) == 0
        segments = build_segments(1000, 1005, 100, 0, 10)
        assert len(segments) == 10
        segments = build_segments(0, 5, 100, 0, 10)
        assert len(segments) == 0
        segments = build_segments(0, 0, 100, 10, 0)
        assert len(segments) == 0
        segments = build_segments(0, 11.17, 0.5, 0.25, 0)
        assert len(segments) == self.__count_segments(0, 11.17, 0.5, 0.25, 0.0)
        assert round(segments[-1][0], 2) == 10.75
        assert round(segments[-1][1], 2) == 11.17
        segments = build_segments(0, 11.17, 0.5, 0.1, 0)
        assert len(segments) == self.__count_segments(0, 11.17, 0.5, 0.1, 0.0)
        assert round(segments[-1][0], 2) == 10.8
        assert round(segments[-1][1], 2) == 11.17
        segments = build_segments(0, 11.17, 0.5, 0.1, 0.4)
        assert len(segments) == self.__count_segments(0, 11.17, 0.5, 0.1, 0.4)
        assert round(segments[-1][0], 2) == 10.40
        assert round(segments[-1][1], 2) == 10.90
        segments = build_segments(0, 60, 14, 7.5, 10)
        assert len(segments) == self.__count_segments(0, 60, 14, 7.5, 10)
        assert round(segments[-1][0], 2) == 45.5
        assert round(segments[-1][1], 2) == 59.5
        segments = build_segments(10, 60, 14, 7.5, 10)
        assert len(segments) == self.__count_segments(0, 60, 14, 7.5, 10)
        assert round(segments[-1][0], 2) == 55.5
        assert round(segments[-1][1], 2) == 69.5

    def test_audio_splitter_single_file(self, uses_db: None, reload_tester: ReloadTester) -> None:
        audio_filepath = get_audio_file('jfk_1961_0109_cityuponahill-excerpt.flac')  # 60s audio file
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))
        audio_segment_view = pxt.create_view(
            'audio_segments',
            base_t,
            iterator=audio_splitter(
                audio=base_t.audio, segment_duration_sec=5.0, overlap_sec=0.0, min_segment_duration_sec=0.0
            ),
        )
        assert audio_segment_view.count() == self.__get_segment_count(audio_filepath, 5.0, 0.0, 0.0)
        results = reload_tester.run_query(audio_segment_view.order_by(audio_segment_view.pos))
        for result in results:
            assert result['audio'] == audio_filepath
        assert results[-1]['segment_end'] == 60
        for i in range(len(results)):
            assert math.floor(results[i]['segment_start']) == i * 5.0
        for i in range(len(results) - 1):
            assert round(results[i]['segment_end'] - results[i]['segment_start']) == 5.0

        audio_segment_view = pxt.create_view(
            'audio_segments_overlap',
            base_t,
            iterator=audio_splitter(
                audio=base_t.audio, segment_duration_sec=14.0, overlap_sec=2.5, min_segment_duration_sec=0.0
            ),
        )
        assert audio_segment_view.count() == self.__get_segment_count(audio_filepath, 14.0, 2.5, 0.0)
        results = reload_tester.run_query(audio_segment_view.order_by(audio_segment_view.pos))
        for result in results:
            assert result['audio'] == audio_filepath
        assert results[-1]['segment_end'] == 60

        audio_segment_view = pxt.create_view(
            'audio_segments_overlap_with_drop',
            base_t,
            iterator=audio_splitter(
                audio=base_t.audio, segment_duration_sec=14.0, overlap_sec=7.5, min_segment_duration_sec=10
            ),
        )
        assert audio_segment_view.count() == self.__get_segment_count(audio_filepath, 14.0, 7.5, 10.0)
        results = reload_tester.run_query(audio_segment_view.order_by(audio_segment_view.pos))
        for result in results:
            assert result['audio'] == audio_filepath
        reload_tester.run_reload_test()

    def test_create_audio_splitter(self, uses_db: None) -> None:
        audio_filepath = get_audio_file('jfk_1961_0109_cityuponahill-excerpt.flac')  # 60s audio file
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))
        with pytest.raises(pxt.Error) as excinfo:
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(
                    audio=base_t.audio, segment_duration_sec=-1, overlap_sec=1, min_segment_duration_sec=1
                ),
            )
        assert '`segment_duration_sec` must be a positive number' in str(excinfo.value)

        with pytest.raises(pxt.Error) as excinfo:
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(
                    audio=base_t.audio, segment_duration_sec=1, overlap_sec=0, min_segment_duration_sec=2
                ),
            )
        assert '`segment_duration_sec` must be at least `min_segment_duration_sec`' in str(excinfo.value)

        with pytest.raises(pxt.Error) as excinfo:
            _ = pxt.create_view(
                'audio_segments',
                base_t,
                iterator=audio_splitter(
                    audio=base_t.audio, segment_duration_sec=1, overlap_sec=1, min_segment_duration_sec=0
                ),
            )
        assert '`overlap_sec` must be strictly less than `segment_duration_sec`' in str(excinfo.value)

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

    @pytest.mark.skipif(IN_CI, reason='Runs out of disk space on CI')
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
