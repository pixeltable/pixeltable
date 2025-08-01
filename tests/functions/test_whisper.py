import sysconfig

import pytest

import pixeltable as pxt

from ..utils import get_audio_files, rerun, skip_test_if_not_installed, validate_update_status


@rerun(reruns=1, reruns_delay=8)
class TestWhisper:
    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Unreliable on Linux ARM')
    def test_whisper(self, reset_db: None) -> None:
        skip_test_if_not_installed('whisper')
        from pixeltable.functions import whisper

        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t = pxt.create_table('whisper', {'audio': pxt.Audio})
        t.add_computed_column(transcription=whisper.transcribe(t.audio, model='base.en'))
        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        result = t.collect()['transcription'][0]
        assert result['language'] == 'en'
        assert 'city upon a hill' in result['text']
