import logging
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts

from ..utils import (
    ensure_s3_pytest_resources_access,
    get_audio_files,
    get_image_files,
    get_test_video_files,
    get_video_files,
    rerun,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)
from .tool_utils import run_tool_invocations_test

_logger = logging.getLogger('pixeltable')


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestGemini:
    @pytest.mark.parametrize('model', ['gemini-2.5-flash', 'gemini-3-pro-preview'])
    def test_generate_content(self, model: str, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateContentConfigDict

        from pixeltable.functions.gemini import generate_content

        t = pxt.create_table('test_tbl', {'contents': pxt.String, 'row_id': pxt.Int})
        t.add_computed_column(output=generate_content(t.contents, model=model))

        if model != 'gemini-3-pro-preview':
            # Some of these options are not supported for gemini-3-pro
            config = GenerateContentConfigDict(
                candidate_count=3,
                stop_sequences=['\n'],
                max_output_tokens=300,
                temperature=1.0,
                top_p=0.95,
                top_k=40,
                response_mime_type='text/plain',
            )
            t.add_computed_column(output2=generate_content(t.contents, model=model, config=config))

        long_text = 'Pixeltable is an amazing tool for multimodal data. ' * 200 + '.mp4'  # 6000+ chars
        validate_update_status(
            t.insert(
                [
                    {'contents': 'Write a sentence about a magic backpack.', 'row_id': 0},
                    {'contents': 'Create a summary of: ' + long_text, 'row_id': 1},
                ]
            ),
            expected_rows=2,
        )
        results = t.order_by(t.row_id).collect()

        text = results['output'][0]['candidates'][0]['content']['parts'][0]['text']
        print(text)
        assert text
        if model != 'gemini-3-pro-preview':
            assert 'backpack' in text  # sanity check (gemini-3-pro is so "creative" that it often omits this word)
            text2 = results['output2'][0]['candidates'][0]['content']['parts'][0]['text']
            print(text2)
            assert 'backpack' in text2

        text = results['output'][1]['candidates'][0]['content']['parts'][0]['text']
        print(text)
        assert text

    def test_generate_content_multimodal(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_content

        images = get_image_files()[:2]

        t = pxt.create_table('test_tbl', {'id': pxt.Int, 'image': pxt.Image})
        t.add_computed_column(
            output=generate_content([t.image, "Describe what's in this image."], model='gemini-2.5-flash')
        )
        validate_update_status(t.insert({'id': n, 'image': image} for n, image in enumerate(images)), expected_rows=2)
        results = t.order_by(t.id).collect()
        assert 'French horn' in results['output'][0]['candidates'][0]['content']['parts'][0]['text']
        assert 'truck' in results['output'][1]['candidates'][0]['content']['parts'][0]['text']

    def test_generate_content_video(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        ensure_s3_pytest_resources_access()

        from google.genai import types

        from pixeltable.functions.gemini import generate_content

        video_files = get_test_video_files()[:2]
        video_files.append('s3://pxt-test/pytest-resources/large_videos/6mb.mp4')
        video_files.append('s3://pxt-test/pytest-resources/large_videos/35mb.mp4')

        t = pxt.create_table('test_tbl', {'id': pxt.Int, 'video': pxt.Video})
        config = types.GenerateContentConfig(
            media_resolution='MEDIA_RESOLUTION_LOW', system_instruction='Analyze the visual content only. Ignore audio.'
        )
        t.add_computed_column(
            output=generate_content(
                [t.video, "understand what's happening in this video and create a short title"],
                model='gemini-2.5-flash',
                config=config,
            )
        )
        with patch('pixeltable.functions.gemini.GEMINI_INLINE_LIMIT_BYTES', 1024**2):
            validate_update_status(
                t.insert({'id': n, 'video': video_file} for n, video_file in enumerate(video_files)), expected_rows=4
            )
            results = t.collect()

        for i in range(4):
            text = results['output'][i]['candidates'][0]['content']['parts'][0]['text'].lower()
            print(f'Video analysis result id={i}: {text}')
            assert text and not any(word in text for word in ['failed', 'unable', 'invalid'])

    def test_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions import gemini

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String}, if_exists='replace')
            t.add_computed_column(response=gemini.generate_content(t.prompt, model='gemini-2.5-flash', tools=tools))
            t.add_computed_column(tool_calls=gemini.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table)

    def test_generate_images(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateImagesConfigDict

        from pixeltable.functions.gemini import generate_images

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=generate_images(t.prompt, model='imagen-4.0-generate-001'))
        config = GenerateImagesConfigDict(aspect_ratio='4:3')
        t.add_computed_column(output2=generate_images(t.prompt, model='imagen-4.0-generate-001', config=config))

        validate_update_status(
            t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1
        )
        results = t.collect()
        assert results['output'][0].size == (1024, 1024)
        assert results['output2'][0].size == (1280, 896)

    @pytest.mark.very_expensive
    @rerun(reruns=3, reruns_delay=30)  # longer delay between reruns
    def test_generate_videos(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_videos

        duration = 4
        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'image': pxt.Image, 'video': pxt.Video})
        t.add_computed_column(
            output=generate_videos(
                t.prompt, t.image, model='veo-3.0-generate-001', config={'duration_seconds': duration}
            )
        )
        prompts = [
            {'prompt': 'A giant pixel floating over the open ocean in a sea of data to the sound of ambient music'},
            {
                'prompt': 'Giraffes are foraging in a lush savannah as the leaves sway in the wind',
                'image': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000025.jpg',
            },
        ]

        t.add_computed_column(metadata=t.output.get_metadata())
        validate_update_status(t.insert(prompts), expected_rows=len(prompts))

        results = t.collect()
        for i in range(len(results)):
            file_path = results['output'][i]
            print(f'Generated video: {file_path}')
            metadata = results['metadata'][i]
            print(f'Generated video metadata: {metadata}')
            assert Path(file_path).exists()

            # Validate metadata
            streams = metadata['streams']
            video_stream = next(s for s in streams if s['type'] == 'video')
            audio_stream = next(s for s in streams if s['type'] == 'audio')
            assert len(streams) == 2, metadata
            assert video_stream['height'] == 720, metadata
            assert video_stream['duration_seconds'] == duration, metadata
            assert audio_stream['duration_seconds'] == duration, metadata

    @pytest.mark.very_expensive
    @rerun(reruns=3, reruns_delay=30)
    def test_generate_videos_reference_images(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_videos

        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'ref1': pxt.Image, 'ref2': pxt.Image})
        t.add_computed_column(
            output=generate_videos(
                t.prompt,
                images=[t.ref1, t.ref2],
                reference_types=['asset', 'asset'],
                model='veo-3.1-generate-preview',
                config={'duration_seconds': 8},
            )
        )
        validate_update_status(
            t.insert(
                [
                    {
                        'prompt': 'A woman wearing the dress walks confidently through a sun-drenched lagoon',
                        'ref1': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000025.jpg',
                        'ref2': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg',
                    }
                ]
            ),
            expected_rows=1,
        )
        results = t.collect()
        file_path = results['output'][0]
        assert Path(file_path).exists()

    @pytest.mark.expensive
    def test_generate_speech(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_speech

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        t.add_computed_column(audio=generate_speech(t.text, model='gemini-2.5-flash-preview-tts', voice='Kore'))
        validate_update_status(t.insert(text='Hello, this is a test of Gemini text to speech.'), expected_rows=1)
        results = t.collect()
        audio_path = results['audio'][0]
        assert Path(audio_path).exists()
        assert audio_path.endswith('.wav')

    @pytest.mark.expensive
    def test_generate_speech_multispeaker(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_speech

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        t.add_computed_column(
            audio=generate_speech(t.text, model='gemini-2.5-flash-preview-tts', voices={'Alice': 'Kore', 'Bob': 'Puck'})
        )
        validate_update_status(
            t.insert(text='Alice: Hello, how are you today? Bob: I am doing great, thanks for asking!'), expected_rows=1
        )
        results = t.collect()
        audio_path = results['audio'][0]
        assert Path(audio_path).exists()
        assert audio_path.endswith('.wav')

    @pytest.mark.expensive
    def test_transcribe(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import transcribe

        audio_files = get_audio_files()
        t = pxt.create_table('test_tbl', {'audio': pxt.Audio})
        t.add_computed_column(
            transcript=transcribe(t.audio, model='gemini-2.5-flash', prompt='Transcribe this audio recording.')
        )
        validate_update_status(t.insert(audio=audio_files[0]), expected_rows=1)
        results = t.collect()
        transcript = results['transcript'][0]
        assert isinstance(transcript, str)
        assert len(transcript) > 0

    def test_embed_content(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import embed_content

        t = pxt.create_table('test', {'rowid': pxt.Int, 'text': pxt.String})

        # Test embeddings as computed columns
        t.add_computed_column(embed0=embed_content(t.text, model='gemini-embedding-001'))
        t.add_computed_column(
            embed1=embed_content(t.text, model='gemini-embedding-001', config={'output_dimensionality': 1536})
        )
        assert t.embed0.col.col_type.matches(ts.ArrayType((3072,), np.dtype('float32'))), t.embed0.col.col_type
        assert t.embed1.col.col_type.matches(ts.ArrayType((1536,), np.dtype('float32'))), t.embed1.col.col_type
        validate_update_status(
            t.insert(
                [
                    {'rowid': 1, 'text': 'Pixeltable is a great tool for AI workload orchestration and storage'},
                    {'rowid': 2, 'text': 'The quick brown fox jumps over the lazy dog.'},
                    {'rowid': 3, 'text': 'The five boxing wizards jump quickly.'},
                ]
            ),
            expected_rows=3,
        )
        for row in t.collect():
            embedding = row['embed0']
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            assert embedding.shape == (3072,)
            embedding = row['embed1']
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            assert embedding.shape == (1536,)

        # Test embeddings as embedding indexes
        t.add_embedding_index(
            t.text, idx_name='embed_idx0', embedding=embed_content.using(model='gemini-embedding-001')
        )
        t.add_embedding_index(
            t.text,
            idx_name='embed_idx1',
            embedding=embed_content.using(
                model='gemini-embedding-001', config={'output_dimensionality': 768}, use_batch_api=False
            ),
        )

        sim = t.text.similarity(string='Coordinating AI tasks can be achieved with Pixeltable.', idx='embed_idx0')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 1

        sim = t.text.similarity(string='The five dueling sorcerers leap rapidly.', idx='embed_idx1')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 3

    @pytest.mark.expensive
    def test_embed_content_batch_api(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import embed_content

        t = pxt.create_table('test', {'rowid': pxt.Int, 'text': pxt.String})
        validate_update_status(
            t.insert(
                [
                    {'rowid': 1, 'text': 'Pixeltable is a great tool for AI workload orchestration and storage'},
                    {'rowid': 2, 'text': 'The quick brown fox jumps over the lazy dog.'},
                    {'rowid': 3, 'text': 'The five boxing wizards jump quickly.'},
                ]
            ),
            expected_rows=3,
        )

        t.add_embedding_index(t.text, embedding=embed_content.using(model='gemini-embedding-001', use_batch_api=True))

        sim = t.text.similarity(string='Coordinating AI tasks can be achieved with Pixeltable.')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 1

        sim = t.text.similarity(string='The five dueling sorcerers leap rapidly.')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 3

    @pytest.mark.expensive
    def test_generate_content_scheduler(self, uses_db: None) -> None:
        """
        Scheduler stress test: 30 rows on gemini-2.5-flash free tier (~10 RPM) triggers 429s
        and verifies that retry logic recovers all rows.

        Meant to be used in conjunction with a free tier gemini api key, to manually verify that
        the retry logic in RateLimitsScheduler is working properly. If the test fails with a non-zero
        number of exceptions, check the logs to see if they were 429 errors and if retries were attempted.
        """
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_content

        with open('tests/data/random_words', encoding='utf-8') as f:
            wordlist = [w.strip() for w in f if w.strip() and not w.startswith('#')]

        num_rows = 30
        model = 'gemini-2.5-flash'

        t = pxt.create_table('scheduler_tbl', {'word1': pxt.String, 'word2': pxt.String})
        t.add_computed_column(prompt='Use "' + t.word1 + '" and "' + t.word2 + '" in one short sentence.')
        t.add_computed_column(response=generate_content(t.prompt, model=model))

        rows = [{'word1': w1, 'word2': w2} for w1, w2 in (random.sample(wordlist, k=2) for _ in range(num_rows))]

        t0 = time.monotonic()
        status = t.insert(rows, on_error='ignore')
        elapsed = time.monotonic() - t0

        succeeded = num_rows - status.num_excs
        _logger.debug(
            f'model={model}, rows={num_rows}, succeeded={succeeded}, errors={status.num_excs}, '
            f'elapsed={elapsed:.2f}s  ({succeeded / max(elapsed, 0.001):.2f} req/s)'
        )

        assert status.num_excs == 0, f'{status.num_excs} rows failed permanently — retries did not recover them'

    def test_embed_content_multimodal(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')

        with patch('pixeltable.functions.gemini.GEMINI_INLINE_LIMIT_BYTES', 2**20):
            self._run_test_embed_content_multimodal()

    def _run_test_embed_content_multimodal(self) -> None:
        from pixeltable.functions.gemini import embed_content

        images = (
            'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000025.jpg',
            'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000030.jpg',
        )
        audio = (
            next(file for file in get_audio_files() if 'jfk' in file),
            next(file for file in get_audio_files() if 'sample.mp3' in file),
        )
        video = (
            next(file for file in get_video_files() if '10-Second Video' in file),
            next(file for file in get_video_files() if 'bangkok' in file),
        )
        # documents = (
        #     next(file for file in get_documents() if '1706.03762.pdf' in file),
        #     next(file for file in get_documents() if 'Vector_database.pdf' in file),
        # )

        t = pxt.create_table('test_tbl_image', {'id': pxt.Int, 'image': pxt.Image})
        t.add_embedding_index(t.image, embedding=embed_content.using(model='gemini-embedding-2-preview'))
        validate_update_status(t.insert({'id': n, 'image': image} for n, image in enumerate(images)), expected_rows=2)

        # Test that the embedding does what it's supposed to
        res = (
            t.select(t.id, sim=t.image.similarity(string='A photograph of a giraffe eating from a tree in a savannah'))
            .order_by(t.id)
            .collect()
        )
        assert res[0]['sim'] - res[1]['sim'] > 0.1  # the giraffe image should have a non-negligibly better match
        res = (
            t.select(t.id, sim=t.image.similarity(string='A photo of a vase full of flowers sitting on a table'))
            .order_by(t.id)
            .collect()
        )
        assert res[1]['sim'] - res[0]['sim'] > 0.1  # as before, in reverse

        t = pxt.create_table('test_tbl_audio', {'id': pxt.Int, 'audio': pxt.Audio})
        t.add_embedding_index(t.audio, embedding=embed_content.using(model='gemini-embedding-2-preview'))
        validate_update_status(
            t.insert({'id': n, 'audio': audio_file} for n, audio_file in enumerate(audio)), expected_rows=2
        )

        res = (
            t.select(
                t.id, sim=t.audio.similarity(string='A political speech by John F. Kennedy mentioning a city on a hill')
            )
            .order_by(t.id)
            .collect()
        )
        assert res[0]['sim'] - res[1]['sim'] > 0.1
        res = t.select(t.id, sim=t.audio.similarity(string='A recording of music playing')).order_by(t.id).collect()
        assert res[1]['sim'] - res[0]['sim'] > 0.1

        t = pxt.create_table('test_tbl_video', {'id': pxt.Int, 'video': pxt.Video})
        t.add_embedding_index(t.video, embedding=embed_content.using(model='gemini-embedding-2-preview'))
        validate_update_status(
            t.insert({'id': n, 'video': video_file} for n, video_file in enumerate(video)), expected_rows=2
        )

        res = (
            t.select(t.id, sim=t.video.similarity(string='A television commercial advertising quesadillas'))
            .order_by(t.id)
            .collect()
        )
        assert res[0]['sim'] - res[1]['sim'] > 0.1
        res = (
            t.select(t.id, sim=t.video.similarity(string='A video of street traffic in a busy city'))
            .order_by(t.id)
            .collect()
        )
        assert res[1]['sim'] - res[0]['sim'] > 0.1
