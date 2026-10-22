import sysconfig

import pytest

import pixeltable as pxt

from ..utils import get_audio_files, rerun_on_network_error, skip_test_if_not_installed, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


@rerun_on_network_error()
class TestWhisper:
    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Unreliable on Linux ARM')
    def test_whisper(self, uses_db: None, local_embed: pxt.Function) -> None:
        skip_test_if_not_installed('whisper')
        from pixeltable.functions import whisper

        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t = pxt.create_table('whisper', {'audio': pxt.Audio | None})
        t.add_computed_column(transcription=whisper.transcribe(t.audio, model='base.en'))
        t.add_computed_column(transcription_words=whisper.transcribe(t.audio, model='base.en', word_timestamps=True))
        t.add_computed_column(text=t.transcription.text)
        t.add_computed_column(first_word=t.transcription_words.segments[0].words[0])
        columns = t.get_metadata()['columns']
        assert columns['text']['type_'] == 'String | None'
        assert columns['first_word']['type_'] == (
            "Json[{'word': String, 'start': Float, 'end': Float, 'probability': Float}] | None"
        )
        t.add_embedding_index('text', embedding=local_embed)

        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        row = t.collect()[0]
        assert row['transcription']['language'] == 'en'
        assert 'city upon a hill' in row['text']
        assert row['first_word']['start'] <= row['first_word']['end']

        # insert() checks that both outputs conform to the WhisperTranscription schema
        out = pxt.create_table('whisper_out', {'transcription': whisper.WhisperTranscription})
        validate_update_status(
            out.insert([{'transcription': row['transcription']}, {'transcription': row['transcription_words']}]),
            expected_rows=2,
        )
