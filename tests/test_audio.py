from typing import Optional

import av

import pixeltable as pxt
import pixeltable.env as env
from pixeltable.iterators.audio import AudioIterator
from pixeltable.utils.media_store import MediaStore

from .utils import get_audio_files, get_video_files, validate_update_status


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

    def test_audio_iterator_on_audio(self, reset_db) -> None:
        pxt.drop_dir("audio_chunking", force=True)
        pxt.create_dir("audio_chunking")
        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        audio_chunk_view = pxt.create_view(
            "audio_chunking.audio_chunks",
            base_t,
            iterator = AudioIterator.create(
                audio=base_t.audio,
                chunk_duration = 5.0,
                overlap = 1.0,
                min_chunk_duration = 1.0,
                drop_incomplete_chunks=True
            ))
        print(audio_chunk_view.count())

    def test_audio_iterator_on_videos(self, reset_db) -> None:
        pxt.drop_dir("audio_chunking", force=True)
        pxt.create_dir("audio_chunking")
        video_filepaths = get_video_files()
        video_t = pxt.create_table('videos', {'video': pxt.Video})
        status = video_t.insert({'video': p} for p in video_filepaths)
        print(status)
        # extract audio
        video_t.add_computed_column(audio=video_t.video.extract_audio(format='mp3'))

        audio_chunk_view = pxt.create_view(
            'audio_chunking.audio_chunks',
            video_t,
            iterator=AudioIterator.create(
                audio=video_t.audio,
                chunk_duration=2.0,
                overlap=0.5,
                min_chunk_duration=0.5,
                drop_incomplete_chunks=True
            )
        )
        print(audio_chunk_view.count())
        result = audio_chunk_view.where(audio_chunk_view.chunk_idx >= 2).collect()
        print(result)

