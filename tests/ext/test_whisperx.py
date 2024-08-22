import platform

import pytest

import pixeltable as pxt
from ..utils import skip_test_if_not_installed, get_audio_files, validate_update_status


class TestWhisperx:
    @pytest.mark.skipif(
        platform.system() == 'Darwin' and platform.machine() != 'arm64',
        reason='Does not run on Intel macOS machines (at least in CI)',
    )
    def test_whisperx(self, reset_db):
        skip_test_if_not_installed('whisperx')
        from pixeltable.ext.functions import whisperx

        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t = pxt.create_table('whisperx', {'audio': pxt.AudioType()})
        t['transcription'] = whisperx.transcribe(t.audio, model='tiny.en')
        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        result = t.collect()['transcription'][0]
        assert result['language'] == 'en'
        assert 'city upon a hill' in result['segments'][1]['text']
