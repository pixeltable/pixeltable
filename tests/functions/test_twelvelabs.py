import pytest

import pixeltable as pxt

from ..utils import get_audio_files, rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status
from ..utils import get_image_files, get_video_files


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestTwelveLabs:
    def test_embed_text(self, reset_db: None) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        t = pxt.create_table('test_tbl', {'input': pxt.String, 'image': pxt.Image})
        t.add_computed_column(embed=embed(model_name='marengo3.0', text=t.input, image=t.image))
        images = get_image_files()
        rows = [
            {'input': 'Twelve Labs provides multimodal embedding models.', 'image': None},
            {'input': 'An optional image can be specified with text embeddings.', 'image': images[0]}
        ]
        validate_update_status(t.insert(rows), 2)
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (512,)

    def test_embed_image(self, reset_db: None) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        image_filepaths = get_image_files()[:2]
        t = pxt.create_table('image_tbl', {'image': pxt.Image})
        t.add_computed_column(embed=embed(model_name='marengo3.0', image=t.image))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (512,)

    def test_embed_audio(self, reset_db: None) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        audio_filepaths = get_audio_files()
        base_t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        validate_update_status(base_t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        v = pxt.create_view(
            'audio_chunks',
            base_t,
            iterator=pxt.iterators.AudioSplitter.create(
                # Twelvelabs models require a minimum audio duration of 4 seconds
                audio=base_t.audio,
                chunk_duration_sec=5.0,
                min_chunk_duration_sec=4.0,
            ),
        )
        v.add_computed_column(embed=embed(model_name='marengo3.0', audio=v.audio_chunk))
        res = v.select(v.embed).collect()
        assert res['embed'][0].shape == (512,)

    def test_embed_video(self, reset_db: None) -> None:
        skip_test_if_not_installed('twelvelabs')
        skip_test_if_no_client('twelvelabs')
        from pixeltable.functions.twelvelabs import embed

        video_filepaths = get_video_files()[:1]  # Just send one of them for testing
        base_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        validate_update_status(base_t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        v = pxt.create_view(
            'video_segments',
            base_t,
            iterator=pxt.iterators.VideoSplitter.create(
                video=base_t.video,
                duration=5.0,
                min_segment_duration=4.0,
            ),
        )
        v.add_computed_column(embed=embed(model_name='marengo3.0', video=v.video_segment))
        res = v.select(v.embed).collect()
        assert res['embed'][0].shape == (512,)