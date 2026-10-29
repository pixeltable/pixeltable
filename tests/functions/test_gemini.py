import base64
import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.runtime import get_runtime, reset_runtime

from ..utils import (
    ensure_s3_pytest_resources_access,
    get_audio_files,
    get_image_files,
    get_test_video_files,
    get_video_files,
    pxt_raises,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)
from .tool_utils import run_tool_invocations_test

if TYPE_CHECKING:
    from google.genai import types

_logger = logging.getLogger('pixeltable_test')


class TestGeminiLocal:
    """Tests that need no credentials or network, kept out of TestGemini so that its class markers don't deselect
    them.

    They cover what live calls can't produce on demand. One kind is responses that Gemini returns only in edge
    cases: a blocked prompt, a candidate without content or parts, a blob without a MIME type. The other is a
    deterministic check that the runtime closes the client's async resources. The session-teardown leak check sees
    an unclosed client only while something keeps it alive, since the SDK closes a client when it's
    garbage-collected.
    """

    @pytest.mark.parametrize('close_fails', [False, True])
    # google-genai subclasses aiohttp.ClientSession for the session it opens
    @pytest.mark.filterwarnings('ignore:Inheritance class AiohttpClientSession:DeprecationWarning')
    def test_client_closed_on_reset(self, close_fails: bool, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('google.genai')
        monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
        # the test keeps a reference to the client, as the traceback of a failed call does: the SDK closes a client
        # that is garbage-collected while a loop runs, which would hide the leak
        client = get_runtime().get_client('gemini')
        if close_fails:
            # a failure to close the sync client doesn't keep the async one open
            monkeypatch.setattr(client, 'close', Mock(side_effect=RuntimeError('close failed')))
        # no public accessors expose the SDK's async HTTP clients, so this reads the private attributes. The aiohttp
        # session opens on the first request, on the runtime's loop; _get_aiohttp_session() opens it the same way.
        http_client = client._api_client._async_httpx_client
        aiohttp_session = get_runtime().event_loop.run_until_complete(client._api_client._get_aiohttp_session())
        reset_runtime()
        assert http_client.is_closed
        assert aiohttp_session.closed

    @pytest.mark.parametrize('outcome', ['blocked', 'no_content', 'no_parts', 'no_mime_type'])
    def test_response_without_content(self, outcome: str, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blocked prompt has no candidates, a candidate may have no content or content without parts, and a blob
        may have no MIME type. Each is returned as is, without tool calls."""
        skip_test_if_not_installed('google.genai')
        from google.genai import types

        from pixeltable.functions import gemini

        response = self._mock_response(outcome, monkeypatch)
        result = gemini.generate_content.exec(['Say something.'], {'model': 'gemini-test'})
        assert result == response.model_dump()
        assert gemini._gemini_response_to_pxt_tool_calls.exec([result], {}) is None
        if outcome == 'blocked':
            assert result['prompt_feedback']['block_reason'] == types.BlockedReason.SAFETY

    def test_speech_blocked(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('google.genai')
        from pixeltable.functions import gemini

        self._mock_response('blocked', monkeypatch)
        # generate_speech() is polymorphic, so this calls the single-voice overload's function directly
        single_voice = gemini.generate_speech.py_fns[0]
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match='unexpected response structure'):
            get_runtime().run_coro(single_voice('Say something.', model='gemini-test', voice='Kore'))

    def _mock_response(self, outcome: str, monkeypatch: pytest.MonkeyPatch) -> 'types.GenerateContentResponse':
        """Makes the gemini client return a response of the given shape, and returns that response."""
        from google.genai import types

        from pixeltable.functions import gemini

        candidates = {
            'blocked': None,
            'no_content': [types.Candidate(finish_reason=types.FinishReason.SAFETY)],
            'no_parts': [
                types.Candidate(content=types.Content(role='model'), finish_reason=types.FinishReason.MAX_TOKENS)
            ],
            'no_mime_type': [
                types.Candidate(
                    content=types.Content(role='model', parts=[types.Part(inline_data=types.Blob(data=b'x'))])
                )
            ],
        }[outcome]
        prompt_feedback = (
            types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY)
            if outcome == 'blocked'
            else None
        )
        response = types.GenerateContentResponse(candidates=candidates, prompt_feedback=prompt_feedback)
        client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=response)))
        )
        monkeypatch.setattr(gemini, '_genai_client', lambda: client)
        return response

    def _install_video_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        interaction: object | None = None,
        file_states: list[object] | None = None,
        download_bytes: bytes = b'',
        veo_operation: object | None = None,
    ) -> SimpleNamespace:
        """Points Gemini video calls at a fake client. `veo_operation` is what the Veo long-running API returns."""
        from pixeltable.functions import gemini

        if veo_operation is None:
            generate_videos = AsyncMock(side_effect=AssertionError('long-running generate_videos was called'))
        else:
            generate_videos = AsyncMock(return_value=veo_operation)
        client = SimpleNamespace(
            aio=SimpleNamespace(
                interactions=SimpleNamespace(create=AsyncMock(return_value=interaction)),
                files=SimpleNamespace(
                    get=AsyncMock(side_effect=[SimpleNamespace(state=state) for state in file_states or []]),
                    download=AsyncMock(return_value=download_bytes),
                ),
                models=SimpleNamespace(generate_videos=generate_videos),
                operations=SimpleNamespace(get=AsyncMock()),
            )
        )
        monkeypatch.setattr(gemini, '_genai_client', lambda: client)
        return client

    @pytest.mark.parametrize('payload', ['data', 'uri'])
    def test_generate_videos_omni(self, payload: str, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A text prompt returns a video file from inline base64 or from a file URI."""
        skip_test_if_not_installed('google.genai')
        from pixeltable.functions import gemini

        video_bytes = b'omni-video-' + payload.encode()
        if payload == 'data':
            output_video = SimpleNamespace(data=base64.b64encode(video_bytes).decode('ascii'), uri=None)
            file_states: list[object] | None = None
        else:
            output_video = SimpleNamespace(
                data=None, uri='https://generativelanguage.googleapis.com/v1beta/files/abc-123:download?alt=media'
            )
            from google.genai import types

            file_states = [types.FileState.PROCESSING, types.FileState.ACTIVE]
        client = self._install_video_client(
            monkeypatch,
            interaction=SimpleNamespace(output_video=output_video, status='completed'),
            file_states=file_states,
            download_bytes=video_bytes,
        )

        result = get_runtime().run_coro(
            gemini.generate_videos.py_fns[0](
                'a marble rolling on a track',
                None,
                model='gemini-omni-1.1-flash',
                config={'aspect_ratio': '9:16', 'resolution': '1080p', 'duration_seconds': 4},
            )
        )

        assert Path(result).read_bytes() == video_bytes
        create = client.aio.interactions.create
        assert create.await_args.kwargs == {
            'model': 'gemini-omni-1.1-flash',
            'input': 'a marble rolling on a track',
            'response_format': {
                'type': 'video',
                'aspect_ratio': '9:16',
                'resolution': '1080p',
                'duration': '4s',
                'delivery': 'uri',
            },
            'timeout': 300,
        }
        assert client.aio.models.generate_videos.await_count == 0
        if payload == 'data':
            assert client.aio.files.download.await_count == 0
        else:
            assert client.aio.files.get.await_count == 2
            assert client.aio.files.get.await_args.kwargs['name'] == 'files/abc-123'
            assert client.aio.files.download.await_args.kwargs['file'] == 'files/abc-123'

    def test_generate_videos_omni_image(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A prompt plus an image, and an image alone, are sent as Interactions image inputs."""
        skip_test_if_not_installed('google.genai')
        import io

        from pixeltable.functions import gemini

        video_bytes = b'omni-from-image'
        image = PIL.Image.new('RGB', (8, 8), color='red')
        client = self._install_video_client(
            monkeypatch,
            interaction=SimpleNamespace(
                output_video=SimpleNamespace(data=base64.b64encode(video_bytes).decode('ascii'), uri=None),
                status='completed',
            ),
        )
        text_and_image = get_runtime().run_coro(
            gemini.generate_videos.py_fns[0](
                'bring this drawing to life',
                image,
                model='gemini-omni-1.1-flash',
                config={'aspect_ratio': '16:9', 'resolution': '720p', 'duration': '8s', 'delivery': 'inline'},
            )
        )
        create = client.aio.interactions.create
        request = create.await_args.kwargs
        assert Path(text_and_image).read_bytes() == video_bytes
        assert request['response_format'] == {
            'type': 'video',
            'aspect_ratio': '16:9',
            'resolution': '720p',
            'duration': '8s',
            'delivery': 'inline',
        }
        image_part, text_part = request['input']
        assert image_part['type'] == 'image'
        assert image_part['mime_type'] == 'image/webp'
        assert text_part == {'type': 'text', 'text': 'bring this drawing to life'}
        with PIL.Image.open(io.BytesIO(base64.b64decode(image_part['data']))) as decoded:
            assert decoded.size == image.size

        image_only = get_runtime().run_coro(
            gemini.generate_videos.py_fns[0](None, image, model='gemini-omni-1.1-flash')
        )
        assert Path(image_only).read_bytes() == video_bytes
        image_input = create.await_args.kwargs['input']
        assert len(image_input) == 1
        assert image_input[0]['type'] == 'image'

    def test_generate_videos_omni_reference_images(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('google.genai')
        from pixeltable.functions import gemini

        video_bytes = b'omni-from-references'
        images = [PIL.Image.new('RGB', (4, 4), color='blue'), PIL.Image.new('RGB', (4, 4), color='green')]
        client = self._install_video_client(
            monkeypatch,
            interaction=SimpleNamespace(
                output_video=SimpleNamespace(data=base64.b64encode(video_bytes).decode('ascii'), uri=None),
                status='completed',
            ),
        )
        result = get_runtime().run_coro(
            gemini.generate_videos.py_fns[1](
                'the subject from the first image walks past the object in the second',
                images,
                model='gemini-omni-1.1-flash',
                reference_types=['asset', 'asset'],
            )
        )
        assert Path(result).read_bytes() == video_bytes
        request = client.aio.interactions.create.await_args.kwargs
        assert [part['type'] for part in request['input']] == ['image', 'image', 'text']
        assert 'reference_types' not in request

    def test_generate_videos_omni_errors(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('google.genai')
        from google.genai import types

        from pixeltable.functions import gemini

        generate = gemini.generate_videos.py_fns[0]
        self._install_video_client(monkeypatch)
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='person_generation'):
            get_runtime().run_coro(
                generate('a cat', None, model='gemini-omni-1.1-flash', config={'person_generation': 'allow_all'})
            )
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='duration'):
            get_runtime().run_coro(generate('a cat', None, model='gemini-omni-1.1-flash', config={'duration': 2}))

        uri = 'https://generativelanguage.googleapis.com/v1beta/files/missing1'
        self._install_video_client(monkeypatch, interaction=SimpleNamespace(output_video=None, status='failed'))
        with pxt_raises(pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE, match='did not return a video'):
            get_runtime().run_coro(generate('a cat', None, model='gemini-omni-1.1-flash'))

        self._install_video_client(
            monkeypatch,
            interaction=SimpleNamespace(output_video=SimpleNamespace(data=None, uri=uri), status='completed'),
            file_states=[types.FileState.FAILED],
        )
        with pxt_raises(pxt.ErrorCode.PROVIDER_ERROR, match='failed processing'):
            get_runtime().run_coro(generate('a cat', None, model='gemini-omni-1.1-flash'))

    def test_generate_videos_veo_uses_long_running_api(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Veo model ids stay on generate_videos, which is the Vertex path."""
        skip_test_if_not_installed('google.genai')
        from pixeltable.functions import gemini

        video = SimpleNamespace()
        operation = SimpleNamespace(
            done=True, error=None, response=SimpleNamespace(generated_videos=[SimpleNamespace(video=video)])
        )
        client = self._install_video_client(monkeypatch, veo_operation=operation, download_bytes=b'veo-video')
        result = get_runtime().run_coro(
            gemini.generate_videos.py_fns[0](
                'a cat', None, model='veo-3.1-generate-001', config={'aspect_ratio': '16:9', 'duration_seconds': 8}
            )
        )
        assert Path(result).read_bytes() == b'veo-video'
        request = client.aio.models.generate_videos.await_args.kwargs
        assert request['model'] == 'veo-3.1-generate-001'
        assert request['source'].prompt == 'a cat'
        assert request['config'].aspect_ratio == '16:9'
        assert request['config'].duration_seconds == 8
        assert client.aio.interactions.create.await_count == 0
        assert client.aio.files.download.await_args.kwargs['file'] is video


@pytest.mark.remote_api
@pytest.mark.very_expensive
@pytest.mark.db_roots('local', reason='UDF/integration test')
@rerun_on_network_error()
class TestGemini:
    @pytest.mark.parametrize('model', ['gemini-2.5-flash-lite', 'gemini-3.1-pro-preview'])
    def test_generate_content(self, model: str, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateContentConfigDict

        from pixeltable.functions.gemini import generate_content

        t = pxt.create_table('test_tbl', {'contents': pxt.String | None, 'row_id': pxt.Int | None})
        t.add_computed_column(output=generate_content(t.contents, model=model))

        if 'flash' in model:
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
        if 'flash' in model:
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

        t = pxt.create_table('test_tbl', {'id': pxt.Int | None, 'image': pxt.Image | None})
        t.add_computed_column(
            output=generate_content([t.image, "Describe what's in this image."], model='gemini-2.5-flash-lite')
        )
        validate_update_status(t.insert({'id': n, 'image': image} for n, image in enumerate(images)), expected_rows=2)
        results = t.order_by(t.id).collect()
        assert 'french horn' in results['output'][0]['candidates'][0]['content']['parts'][0]['text'].lower()
        assert 'truck' in results['output'][1]['candidates'][0]['content']['parts'][0]['text'].lower()

    def test_generate_content_video(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        ensure_s3_pytest_resources_access()

        from google.genai import types

        from pixeltable.functions.gemini import generate_content

        video_files = get_test_video_files()[:2]
        video_files.append('s3://pxt-test/pytest-resources/large_videos/6mb.mp4')
        video_files.append('s3://pxt-test/pytest-resources/large_videos/35mb.mp4')

        t = pxt.create_table('test_tbl', {'id': pxt.Int | None, 'video': pxt.Video | None})
        config = types.GenerateContentConfig(
            media_resolution='MEDIA_RESOLUTION_LOW', system_instruction='Analyze the visual content only. Ignore audio.'
        )
        t.add_computed_column(
            output=generate_content(
                [t.video, "understand what's happening in this video and create a short title"],
                model='gemini-2.5-flash-lite',
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

    def test_generate_content_nano_banana(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')

        from pixeltable.functions.gemini import generate_content, generate_images

        t = pxt.create_table('test_tbl', {'prompt': pxt.String | None})
        t.add_computed_column(
            response=generate_content(
                t.prompt, model='gemini-3.1-flash-image-preview', config={'response_modalities': ['IMAGE']}
            )
        )
        validate_update_status(
            t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1
        )

        t.add_computed_column(image=t.response.candidates[0].content.parts[0].inline_data.data)

        results = t.collect()
        image = results['image'][0]
        assert isinstance(image, PIL.Image.Image), f'Expected a PIL Image, got {type(image)}'
        assert image.size[0] > 0 and image.size[1] > 0

        # Passing a Gemini image-generation model to `generate_images` (which is for Imagen) must fail.
        t2 = pxt.create_table('test_tbl_fail', {'prompt': pxt.String | None})
        validate_update_status(t2.insert(prompt='anything'), expected_rows=1)
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='Nano Banana'):
            t2.select(output=generate_images(t2.prompt, model='gemini-3.1-flash-image-preview')).collect()

    def test_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions import gemini

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String | None}, if_exists='replace')
            t.add_computed_column(
                response=gemini.generate_content(t.prompt, model='gemini-3.1-flash-lite', tools=tools)
            )
            t.add_computed_column(tool_calls=gemini.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table)

    # TODO: Test in vertexai or deprecate the imagen UDFs
    @pytest.mark.skip('google-genai 2.x reaches Imagen only in Vertex AI mode, not through the Developer API')
    def test_generate_images(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateImagesConfigDict

        from pixeltable.functions.gemini import generate_images

        t = pxt.create_table('test_tbl', {'prompt': pxt.String | None})
        t.add_computed_column(output=generate_images(t.prompt, model='imagen-4.0-generate-001'))
        config = GenerateImagesConfigDict(aspect_ratio='4:3')
        t.add_computed_column(output2=generate_images(t.prompt, model='imagen-4.0-generate-001', config=config))

        validate_update_status(
            t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1
        )
        results = t.collect()
        assert results['output'][0].size == (1024, 1024)
        assert results['output2'][0].size == (1280, 896)

    @rerun_on_network_error(reruns_delay=30)
    def test_generate_videos(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_videos

        duration = 4
        t = pxt.create_table(
            'test_tbl', {'prompt': pxt.String | None, 'image': pxt.Image | None, 'video': pxt.Video | None}
        )
        t.add_computed_column(
            output=generate_videos(
                t.prompt,
                t.image,
                model='gemini-omni-1.1-flash',
                config={'aspect_ratio': '16:9', 'resolution': '720p', 'duration': f'{duration}s'},
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

    @rerun_on_network_error(reruns_delay=30)
    def test_generate_videos_reference_images(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_videos

        t = pxt.create_table(
            'test_tbl', {'prompt': pxt.String | None, 'ref1': pxt.Image | None, 'ref2': pxt.Image | None}
        )
        t.add_computed_column(
            output=generate_videos(
                t.prompt,
                images=[t.ref1, t.ref2],
                reference_types=['asset', 'asset'],
                model='gemini-omni-1.1-flash',
                config={'duration': '4s'},
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

    def test_generate_speech(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_speech

        t = pxt.create_table('test_tbl', {'text': pxt.String | None})
        t.add_computed_column(audio=generate_speech(t.text, model='gemini-3.1-flash-tts-preview', voice='Kore'))
        validate_update_status(t.insert(text='Hello, this is a test of Gemini text to speech.'), expected_rows=1)
        results = t.collect()
        audio_path = results['audio'][0]
        assert Path(audio_path).exists()
        assert audio_path.endswith('.wav')

    def test_generate_speech_multispeaker(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_speech

        t = pxt.create_table('test_tbl', {'text': pxt.String | None})
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

    def test_transcribe(self, uses_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import transcribe

        audio_files = get_audio_files()
        t = pxt.create_table('test_tbl', {'audio': pxt.Audio | None})
        t.add_computed_column(
            transcript=transcribe(t.audio, model='gemini-3-flash-preview', prompt='Transcribe this audio recording.')
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

        t = pxt.create_table('test', {'rowid': pxt.Int | None, 'text': pxt.String | None})

        # Test embeddings as computed columns
        t.add_computed_column(embed0=embed_content(t.text, model='gemini-embedding-2'))
        t.add_computed_column(
            embed1=embed_content(t.text, model='gemini-embedding-2', config={'output_dimensionality': 1536})
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
        t.add_embedding_index(t.text, idx_name='embed_idx0', embedding=embed_content.using(model='gemini-embedding-2'))
        t.add_embedding_index(
            t.text,
            idx_name='embed_idx1',
            embedding=embed_content.using(model='gemini-embedding-2', config={'output_dimensionality': 768}),
        )

        sim = t.text.similarity(string='Coordinating AI tasks can be achieved with Pixeltable.', idx='embed_idx0')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 1

        sim = t.text.similarity(string='The five dueling sorcerers leap rapidly.', idx='embed_idx1')
        res = t.select(t.rowid, t.text, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['rowid'] == 3

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
        model = 'gemini-2.5-flash-lite'

        t = pxt.create_table('scheduler_tbl', {'word1': pxt.String | None, 'word2': pxt.String | None})
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

        t = pxt.create_table('test_tbl_image', {'id': pxt.Int | None, 'image': pxt.Image | None})
        t.add_embedding_index(t.image, embedding=embed_content.using(model='gemini-embedding-2'))
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

        t = pxt.create_table('test_tbl_audio', {'id': pxt.Int | None, 'audio': pxt.Audio | None})
        t.add_embedding_index(t.audio, embedding=embed_content.using(model='gemini-embedding-2'))
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

        t = pxt.create_table('test_tbl_video', {'id': pxt.Int | None, 'video': pxt.Video | None})
        t.add_embedding_index(t.video, embedding=embed_content.using(model='gemini-embedding-2'))
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
