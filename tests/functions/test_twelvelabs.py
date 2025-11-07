import pytest

import pixeltable as pxt

from ..utils import get_audio_files, rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestTwelveLabs:
    def test_embed_text(self, reset_db: None) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(embed=embed(model_name='Marengo-retrieval-2.7', text=t.input))
        validate_update_status(t.insert(input='Twelve Labs provides multi-modal embeddings models.'), 1)
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (1024,)

    def test_embed_audio(self, reset_db: None) -> None:
        audio_filepaths = [file for file in get_audio_files() if '.flac' not in file]
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        v = pxt.create_view(
            'audio_chunks',
            base_t,
            iterator=pxt.iterators.AudioSplitter.create(
                audio=base_t.audio, chunk_duration_sec=5.0, min_chunk_duration_sec=0.5
            ),
        )

        _ = v.select(v.audio_chunk.get_metadata()).collect()
        v.add_computed_column(
            embed=pxt.functions.twelvelabs.embed(model_name='Marengo-retrieval-2.7', audio=v.audio_chunk)
        )
        res = v.select(v.audio_chunk.get_metadata(), v.embed).collect()
        for row in res:
            print(row)
