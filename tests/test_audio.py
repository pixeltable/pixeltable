from typing import Optional

import av
import math

import pixeltable as pxt
import pixeltable.env as env
from pixeltable.iterators.audio import AudioSplitter
from pixeltable.utils.media_store import MediaStore

from .utils import get_audio_files, get_audio_file, get_video_files, validate_update_status


class TestAudio:
    def check_audio_params(self, path: str, format: Optional[str] = None, codec: Optional[str] = None):
        with av.open(path) as container:
            audio_stream = container.streams.audio[0]
            if format is not None:
                assert format == container.format.name
            if codec is not None:
                assert codec == audio_stream.codec_context.codec.name

    def test_basic(self, reset_db) -> None:
        audio_filepaths = get_audio_files()
        audio_t = pxt.create_table('audio', {'audio_file': pxt.Audio})
        status = audio_t.insert({'audio_file': p} for p in audio_filepaths)
        assert status.num_rows == len(audio_filepaths)
        assert status.num_excs == 0
        paths = audio_t.select(output=audio_t.audio_file.localpath).collect()['output']
        assert set(paths) == set(audio_filepaths)

    def test_extract(self, reset_db) -> None:
        video_filepaths = get_video_files()
        video_t = pxt.create_table('videos', {'video': pxt.Video})
        video_t.add_computed_column(audio=video_t.video.extract_audio())

        # one of the 3 videos doesn't have audio
        status = video_t.insert({'video': p} for p in video_filepaths)
        assert status.num_rows == len(video_filepaths)
        assert status.num_excs == 0
        assert MediaStore.count(video_t._id) == len(video_filepaths) - 1
        assert video_t.where(video_t.audio != None).count() == len(video_filepaths) - 1
        assert env.Env.get().num_tmp_files() == 0

        video_t = pxt.get_table('videos')
        assert video_t.where(video_t.audio != None).count() == len(video_filepaths) - 1

        # test generating different formats and codecs
        paths = video_t.select(output=video_t.video.extract_audio(format='wav', codec='pcm_s16le')).collect()['output']
        # media files that are created as a part of a query end up in the tmp dir
        assert env.Env.get().num_tmp_files() == video_t.where(video_t.audio != None).count()
        for path in [p for p in paths if p is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s16le')
        # higher resolution
        paths = video_t.select(output=video_t.video.extract_audio(format='wav', codec='pcm_s32le')).collect()['output']
        for path in [p for p in paths if p is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s32le')

        for format in ['mp3', 'flac']:
            paths = video_t.select(output=video_t.video.extract_audio(format=format)).collect()['output']
            for path in [p for p in paths if p is not None]:
                self.check_audio_params(path, format=format)

    def test_get_metadata(self, reset_db) -> None:
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
                        'codec_tag': '\\x00\\x00\\x00\\x00'
                    },
                    'duration_seconds': 60.0
                }
            ],
            'bit_rate': 342510,
            'metadata': {'encoder': 'Lavf61.1.100'},
            'bit_exact': False
        }

    def __get_chunk_counts(self, files, target_chunk_size_sec) -> dict[str, int]:
        file_to_chunk_count: dict[str, int] = {}
        for file in files:
            container = av.open(file)
            total_duration = container.streams.audio[0].duration or 0
            start_time = container.streams.audio[0].start_time or 0
            time_base = container.streams.audio[0].time_base
            target_chunk_size_pts = int(target_chunk_size_sec / time_base)
            chunks = math.ceil((total_duration - start_time) / target_chunk_size_pts) if total_duration else 0
            if chunks > 0:
                file_to_chunk_count[file] = chunks
        return file_to_chunk_count

    def test_audio_iterator_on_audio(self, reset_db) -> None:
        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        audio_chunk_view = pxt.create_view(
            "audio_chunks",
            base_t,
            iterator = AudioSplitter.create(
                audio=base_t.audio,
                chunk_duration_sec = 5.0,
                overlap_sec = 0.0,
                min_chunk_duration_sec = 0.0,
                drop_incomplete_chunks=False
            ))
        file_to_chunks = self.__get_chunk_counts(audio_filepaths, 5.0)
        results = audio_chunk_view.collect()
        assert len(results) == sum(file_to_chunks.values())
        file_to_chunks_from_view = {}
        for result in results:
            file_to_chunks_from_view[result['audio']] = file_to_chunks_from_view.get(result['audio'], 0) + 1
        assert file_to_chunks_from_view == file_to_chunks

    def test_audio_iterator_on_videos(self, reset_db) -> None:
        video_filepaths = get_video_files()
        video_t = pxt.create_table('videos', {'video': pxt.Video})
        video_t.insert({'video': p} for p in video_filepaths)
        # extract audio
        video_t.add_computed_column(audio=video_t.video.extract_audio(format='mp3'))
        audio_chunk_view = pxt.create_view(
            'audio_chunks',
            video_t,
            iterator=AudioSplitter.create(
                audio=video_t.audio,
                chunk_duration_sec=2.0
            )
        )
        audio_files = [ result['audio'] for result in video_t.select(video_t.audio).where(video_t.audio != None).collect()]
        file_to_chunks = self.__get_chunk_counts(audio_files, 2.0)
        results = audio_chunk_view.collect()
        assert len(results) == sum(file_to_chunks.values())
        file_to_chunks_from_view = {}
        for result in results:
            file_to_chunks_from_view[result['audio']] = file_to_chunks_from_view.get(result['audio'], 0) + 1
        assert file_to_chunks_from_view == file_to_chunks

    def test_audio_iterator_build_chunks(self, reset_db) -> None:
        chunks = AudioSplitter.build_chunks(0, 1005, 100, 0, 10, drop_incomplete_chunks=True)
        assert len(chunks) == 10
        assert all((chunk[1] - chunk[0]) is 100  for chunk in chunks)
        chunks = AudioSplitter.build_chunks(0, 1005, 100, 10, 10, drop_incomplete_chunks=True)
        assert len(chunks) == 10
        assert all((chunk[1] - chunk[0]) is 110 for chunk in chunks[:9])
        assert chunks[-1][0] == 900
        assert chunks[-1][1] == 1005
        chunks = AudioSplitter.build_chunks(0, 1005, 100, 10, 0, drop_incomplete_chunks=False)
        assert len(chunks) == 10
        assert all((chunk[1] - chunk[0]) is 110 for chunk in chunks[:9])
        assert chunks[-1][0] == 900
        assert chunks[-1][1] == 1005
        chunks = AudioSplitter.build_chunks(0, 1005, 100, 0, 0, drop_incomplete_chunks=False)
        assert len(chunks) == 11
        assert all((chunk[1] - chunk[0]) is 100 for chunk in chunks[:10])
        assert chunks[-1][0] == 1000
        assert chunks[-1][1] == 1005
        chunks = AudioSplitter.build_chunks(0, 1055, 100, 10, 60, drop_incomplete_chunks=True)
        assert len(chunks) == 10
        assert all((chunk[1] - chunk[0]) is 110 for chunk in chunks[:10])
        assert chunks[-1][0] == 900
        assert chunks[-1][1] == 1010
        chunks = AudioSplitter.build_chunks(0, 1055, 100, 10, 55, drop_incomplete_chunks=True)
        assert len(chunks) == 11
        assert all((chunk[1] - chunk[0]) is 110 for chunk in chunks[:10])
        assert chunks[-1][0] == 1000
        assert chunks[-1][1] == 1055
        chunks = AudioSplitter.build_chunks(1000, 1005, 100, 0, 10, drop_incomplete_chunks=True)
        assert len(chunks) == 0
        chunks = AudioSplitter.build_chunks(0, 5, 100, 0, 10, drop_incomplete_chunks=True)
        assert len(chunks) == 0
        chunks = AudioSplitter.build_chunks(0, 0, 100, 10, 0, drop_incomplete_chunks=False)
        assert len(chunks) == 0

    def test_audio_iterator_single_file(self, reset_db) -> None:
        audio_filepath = get_audio_file('jfk_1961_0109_cityuponahill-excerpt.flac') # 60s audio file
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert([{'audio': audio_filepath}]))
        audio_chunk_view = pxt.create_view(
            "audio_chunks",
            base_t,
            iterator=AudioSplitter.create(
                audio=base_t.audio,
                chunk_duration_sec=5.0,
                overlap_sec=0.0,
                min_chunk_duration_sec=0.0,
                drop_incomplete_chunks=False
            ))
        assert audio_chunk_view.count() is 12
        result = audio_chunk_view.collect()
        print(result)
        assert all(result['audio'] == audio_filepath for result in result)
        assert result[-1]['end_time_sec'] == 60
        for i in range(len(result)):
            assert math.floor(result[i]['start_time_sec']) == i * 5.0
        for i in range(len(result) - 1):
            assert round(result[i]['duration_sec']) == 5.0

        audio_chunk_view = pxt.create_view(
            "audio_chunks_overlap",
            base_t,
            iterator=AudioSplitter.create(
                audio=base_t.audio,
                chunk_duration_sec=14.0,
                overlap_sec=2.5,
                min_chunk_duration_sec=0.0,
                drop_incomplete_chunks=False
            ))
        assert audio_chunk_view.count() is 5
        result = audio_chunk_view.collect()
        assert all(result['audio'] == audio_filepath for result in result)
        assert result[-1]['end_time_sec'] == 60
        assert round(result[-1]['duration_sec']) == 4
        for i in range(len(result)):
            assert math.floor(result[i]['start_time_sec']) == i * 14.0
        for i in range(len(result) - 1):
            assert round(result[i]['duration_sec']) == 17

        audio_chunk_view = pxt.create_view(
            "audio_chunks_overlap_with_drop",
            base_t,
            iterator=AudioSplitter.create(
                audio=base_t.audio,
                chunk_duration_sec=14.0,
                overlap_sec=2.5,
                min_chunk_duration_sec=4.5,
                drop_incomplete_chunks=True
            ))
        assert audio_chunk_view.count() is 4
        result = audio_chunk_view.collect()
        assert all(result['audio'] == audio_filepath for result in result)
        assert result[-1]['end_time_sec'] < 59
        for i in range(len(result)):
            assert math.floor(result[i]['start_time_sec']) == i * 14.0
        for i in range(len(result)):
            assert round(result[i]['duration_sec']) == 17.0
