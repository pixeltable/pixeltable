from typing import Optional

import av

import pixeltable as pxt
import pixeltable.env as env
from pixeltable.tests.utils import get_video_files, get_audio_files
from pixeltable.type_system import VideoType, AudioType
from pixeltable.utils.media_store import MediaStore


class TestAudio:
    def check_audio_params(self, path: str, format: Optional[str] = None, codec: Optional[str] = None):
        with av.open(path) as container:
            audio_stream = container.streams.audio[0]
            if format is not None:
                assert format == container.format.name
            if codec is not None:
                assert codec == audio_stream.codec_context.codec.name

    def test_basic(self, test_client: pxt.Client) -> None:
        audio_filepaths = get_audio_files()
        cl = test_client
        audio_t = cl.create_table('audio', {'audio_file': AudioType()})
        status = audio_t.insert({'audio_file': p} for p in audio_filepaths)
        assert status.num_rows == len(audio_filepaths)
        assert status.num_excs == 0
        paths = audio_t.select(output=audio_t.audio_file.localpath).collect()['output']
        assert set(paths) == set(audio_filepaths)

    def test_extract(self, test_client: pxt.Client) -> None:
        video_filepaths = get_video_files()
        cl = test_client
        video_t = cl.create_table('videos', {'video': VideoType()})
        from pixeltable.functions.video import extract_audio
        video_t.add_column(audio=extract_audio(video_t.video))

        # one of the 3 videos doesn't have audio
        status = video_t.insert({'video': p} for p in video_filepaths)
        assert status.num_rows == len(video_filepaths)
        assert status.num_excs == 0
        assert MediaStore.count(video_t.get_id()) == len(video_filepaths) - 1
        assert video_t.where(video_t.audio != None).count() == len(video_filepaths) - 1
        assert env.Env.get().num_tmp_files() == 0

        # make sure everything works with a fresh client
        cl = pxt.Client()
        video_t = cl.get_table('videos')
        assert video_t.where(video_t.audio != None).count() == len(video_filepaths) - 1

        # test generating different formats and codecs
        paths = video_t.select(output=extract_audio(video_t.video, format='wav', codec='pcm_s16le')).collect()['output']
        # media files that are created as a part of a query end up in the tmp dir
        assert env.Env.get().num_tmp_files() == video_t.where(video_t.audio != None).count()
        for path in [p for p in paths if p is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s16le')
        # higher resolution
        paths = video_t.select(output=extract_audio(video_t.video, format='wav', codec='pcm_s32le')).collect()['output']
        for path in [p for p in paths if p is not None]:
            self.check_audio_params(path, format='wav', codec='pcm_s32le')

        for format in ['mp3', 'flac']:
            paths = video_t.select(output=extract_audio(video_t.video, format=format)).collect()['output']
            for path in [p for p in paths if p is not None]:
                self.check_audio_params(path, format=format)
