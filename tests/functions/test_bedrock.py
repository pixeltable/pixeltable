from pathlib import Path
from typing import Any, Iterator, cast

import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable import ResultSet
from pixeltable.config import Config

from ..utils import (
    get_audio_files,
    get_image_files,
    get_video_files,
    rerun,
    rerun_on_network_error,
    skip_test_if_no_aws_credentials,
    validate_update_status,
)
from .tool_utils import run_tool_invocations_test

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


@pytest.fixture()
def bedrock_us_east_1(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Configure the Bedrock client and temp_location for us-east-1."""
    monkeypatch.setenv('BEDROCK_REGION_NAME', 'us-east-1')
    monkeypatch.setenv('BEDROCK_TEMP_LOCATION', 's3://pxt-bedrock-test-us-east-1/bedrock_outputs/')
    Config.init(reinit=True)
    yield


@pytest.fixture()
def bedrock_us_west_2(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Configure the Bedrock client and temp_location for us-west-2."""
    monkeypatch.setenv('BEDROCK_REGION_NAME', 'us-west-2')
    monkeypatch.setenv('BEDROCK_TEMP_LOCATION', 's3://pxt-bedrock-test-us-west-2/bedrock_outputs/')
    Config.init(reinit=True)
    yield


def _tbl_name(model_id: str) -> str:
    """Sanitize a model ID into a valid Pixeltable table name."""
    return 'tbl_' + model_id.replace('.', '_').replace(':', '_').replace('-', '_')


def _run_converse_text(model_ids: list[str]) -> None:
    """Run a text converse call for each model_id and assert a text response."""
    from pixeltable.functions.bedrock import converse

    for model_id in model_ids:
        t = pxt.create_table(_tbl_name(model_id), {'input': pxt.String | None})
        messages = [{'role': 'user', 'content': [{'text': t.input}]}]
        t.add_computed_column(output=converse(messages, model_id=model_id))
        validate_update_status(t.insert(input='What is 2+2?'), expected_rows=1)
        results = t.collect()
        assert results[0]['output']['output']['message']['content']


def _assert_twelvelabs_embedding(results: ResultSet) -> None:
    assert 'response' in results[0]
    assert len(results[0]['response']['data'][0]['embedding']) == 512


def _assert_nova_text_response(results: ResultSet) -> None:
    assert results[0]['response']['output']['message']['content'][0]['text']


def _assert_openai_compat_response(results: ResultSet, model_id: str) -> None:
    assert results[0]['response']['choices'][0]['message']['content'], f'No response for {model_id}'


def _assert_text_similarity(t: pxt.Table) -> None:
    sim = t.text.similarity(string='What is machine learning?')
    results = t.order_by(sim, asc=False).limit(2).select(t.text, similarity=sim).collect()
    assert len(results) == 2
    assert 'machine learning' in results['text'][0].lower() or 'intelligence' in results['text'][0].lower()


_LUMA_RAY_MODEL_ID = 'luma.ray-v2:0'
_LUMA_RAY_REGION = 'us-west-2'
_UNAVAILABLE_VIDEO_MODEL_SNIPPETS = (
    'end of its life',
    'end of life',
    "don't have access to the model",
    'do not have access to the model',
    'model access is denied',
    'use case details have not been submitted',
    'not subscribed',
    'aws marketplace',
    'could not find model',
    'could not resolve the foundation model',
    'model identifier is invalid',
)


def _is_throttled_video_request(errortype: str | None, errormsg: str | None) -> bool:
    """Return True when Bedrock rejected the video request because StartAsyncInvoke was throttled.

    A low account quota is shared by concurrent nightly jobs. The rate-limit scheduler retries a
    throttled call up to its own limit. A request that is still throttled after those retries is stored
    as the original botocore error and skipped rather than failed.
    """
    # Share the production classifier's codes.
    from pixeltable.functions import bedrock

    throttle_codes = bedrock._BEDROCK_THROTTLE_CODES
    if errortype in throttle_codes:
        return True
    message = (errormsg or '').lower()
    if 'throttlingexception' in message:
        return True
    return 'too many requests' in message and 'startasyncinvoke' in message


def _is_unavailable_video_model_error(_errortype: str | None, errormsg: str | None) -> bool:
    """Return True when the error message shows the video model is missing, retired, or not enabled.

    Error type alone is not a skip signal. A ResourceNotFoundException from a routing or request bug
    still fails the test unless the message matches a known unavailable-model snippet.
    """
    message = (errormsg or '').lower()
    # Storage failures mention the output bucket; those are configuration errors, not a missing model.
    if 's3:' in message or 's3://' in message:
        return False
    return any(snippet in message for snippet in _UNAVAILABLE_VIDEO_MODEL_SNIPPETS)


def _assert_image_similarity(t: pxt.Table, img_paths: list) -> None:
    with PIL.Image.open(img_paths[0]) as sample_img:
        sim = t.image.similarity(image=sample_img)
        results = t.order_by(sim, asc=False).limit(2).select(t.image, similarity=sim).collect()
    assert len(results) == 2
    assert results['similarity'][0] > results['similarity'][1]


_TEXT_ROWS = [
    {'text': 'Machine learning is a subset of artificial intelligence.'},
    {'text': 'Deep learning uses neural networks with many layers.'},
    {'text': 'Python is a popular programming language.'},
]


def _image_url_body(t: pxt.Table) -> dict:
    """Request body for OpenAI-compatible image_url schema (Mistral, Gemma, NVIDIA, Moonshot, Qwen)."""
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': t.image}},
                    {'type': 'text', 'text': 'Describe this image in one sentence.'},
                ],
            }
        ],
        'max_tokens': 256,
    }


def _converse_image_messages(t: pxt.Table) -> list:
    """Messages for the Converse unified image schema."""
    return [
        {
            'role': 'user',
            'content': [
                {'image': {'format': 'jpeg', 'source': {'bytes': t.image}}},
                {'text': 'What is in this image? Answer in one word.'},
            ],
        }
    ]


def test_throttled_video_request() -> None:
    nightly = (
        'An error occurred (ThrottlingException) when calling the StartAsyncInvoke operation '
        '(reached max retries: 4): Too many requests, please wait before trying again.'
    )
    assert _is_throttled_video_request('ThrottlingException', nightly)
    assert _is_throttled_video_request('ClientError', nightly)
    # Type alone comes from the production code set, so a bland message still skips.
    assert _is_throttled_video_request('TooManyRequestsException', 'please retry')
    assert _is_throttled_video_request('Throttling', None)
    assert not _is_throttled_video_request('ValidationException', 'Invalid input: resolution must be one of 540p, 720p')
    assert not _is_throttled_video_request(
        'ExternalServiceError', 'Async invocation abc failed: model returned an error'
    )
    # "too many requests" without the async operation is not this quota failure.
    assert not _is_throttled_video_request('ClientError', 'Too many requests')
    assert not _is_throttled_video_request('ValidationException', 'too many requests were supplied in the input list')


def _throttle_error(code: str, message: str, *, cls_name: str | None = None) -> Exception:
    from botocore.exceptions import ClientError

    error_cls = ClientError
    if cls_name is not None:
        # typeshed types the three-argument type() as returning type, not the ClientError subclass it builds.
        error_cls = cast(type[ClientError], type(cls_name, (ClientError,), {}))
    return error_cls(
        {
            'Error': {'Code': code, 'Message': message},
            'ResponseMetadata': {
                'RequestId': 'test',
                'HostId': 'test',
                'HTTPStatusCode': 429 if 'throttl' in code.lower() else 400,
                'HTTPHeaders': {},
                'RetryAttempts': 0,
            },
        },
        'StartAsyncInvoke',
    )


def test_bedrock_throttling_classification() -> None:
    from pixeltable.functions.bedrock import _is_bedrock_throttling

    nightly = (
        'An error occurred (ThrottlingException) when calling the StartAsyncInvoke operation '
        '(reached max retries: 4): Too many requests, please wait before trying again.'
    )
    assert _is_bedrock_throttling(Exception(nightly))
    assert _is_bedrock_throttling(_throttle_error('TooManyRequestsException', 'Too many requests'))
    assert not _is_bedrock_throttling(
        _throttle_error('ValidationException', 'too many requests were supplied in the input list')
    )
    assert not _is_bedrock_throttling(
        _throttle_error('ServiceQuotaExceededException', 'The account quota has been exceeded')
    )
    assert not _is_bedrock_throttling(Exception('connection reset'))


def test_bedrock_throttle_retry_delay() -> None:
    """The rate-limit pool waits out a throttle and does not retry a failure after the job was accepted."""
    from pixeltable.functions import bedrock
    from pixeltable.utils.http import is_retryable_error

    info = bedrock.BedrockRateLimitsInfo()
    nightly = _throttle_error(
        'ThrottlingException', 'Too many requests, please wait before trying again.', cls_name='ThrottlingException'
    )
    delay = info.get_retry_delay(nightly, 0)
    assert delay is not None
    assert bedrock._THROTTLE_RETRY_BASE_SECS <= delay < bedrock._THROTTLE_RETRY_BASE_SECS * 2
    capped = info.get_retry_delay(nightly, 5)
    assert capped is not None
    assert (
        bedrock._THROTTLE_RETRY_MAX_SECS
        <= capped
        < bedrock._THROTTLE_RETRY_MAX_SECS + bedrock._THROTTLE_RETRY_BASE_SECS
    )
    assert (
        info.get_retry_delay(
            _throttle_error('ValidationException', 'too many requests were supplied in the input list'), 0
        )
        is None
    )
    assert (
        info.get_retry_delay(_throttle_error('ServiceQuotaExceededException', 'The account quota has been exceeded'), 0)
        is None
    )

    deadline = pxt.ExternalServiceError(
        pxt.ErrorCode.PROVIDER_TIMEOUT, 'Async invocation abc did not finish within 600.0s', provider='bedrock'
    )
    assert not is_retryable_error(deadline)[0]
    accepted = bedrock._terminal_async_error(
        'abc', _throttle_error('ThrottlingException', 'Too many requests, timed out reading the output object')
    )
    assert isinstance(accepted, pxt.ExternalServiceError)
    assert not is_retryable_error(accepted)[0]
    assert info.get_retry_delay(accepted, 0) is None
    # A Bedrock async id is 12 lowercase letters or digits and can contain the retry substring 429.
    id_with_status = bedrock._terminal_async_error(
        'abc429defghi', _throttle_error('ThrottlingException', 'Too many requests, timed out reading the output object')
    )
    assert isinstance(id_with_status, pxt.ExternalServiceError)
    assert not is_retryable_error(id_with_status)[0]
    assert info.get_retry_delay(id_with_status, 0) is None
    assert '429' not in str(id_with_status)
    assert bedrock.invoke_model._resource_pool() == bedrock._BEDROCK_RATE_POOL


def test_terminal_async_error_fallback_has_no_invocation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the classifier still matches, the fallback is the provider error and contains no invocation id."""
    from pixeltable.functions import bedrock

    monkeypatch.setattr(bedrock, 'is_retryable_error', lambda _exc: (True, None))
    monkeypatch.setattr(bedrock, '_is_bedrock_throttling', lambda _exc: False)
    err = bedrock._terminal_async_error('abc429defghi', Exception('model failed'))
    assert isinstance(err, pxt.ExternalServiceError)
    assert str(err) == 'Async invocation failed after it was accepted'
    assert '429' not in str(err)


def _patch_async_clocks(monkeypatch: pytest.MonkeyPatch) -> tuple[list[float], list[float]]:
    """Record scheduler and Bedrock sleeps and return immediately.

    Both modules call asyncio.sleep on the same function, so one wrapper classifies the caller.
    """
    import asyncio
    import inspect

    scheduler_sleeps: list[float] = []
    bedrock_sleeps: list[float] = []

    async def _sleep(delay: float) -> None:  # noqa: RUF029
        caller = inspect.stack()[1].filename
        if caller.endswith('schedulers.py'):
            scheduler_sleeps.append(delay)
        else:
            bedrock_sleeps.append(delay)

    monkeypatch.setattr(asyncio, 'sleep', _sleep)
    return scheduler_sleeps, bedrock_sleeps


class _StartCounter:
    def __init__(self, *, throttles_before_success: int, in_progress: bool, failure_message: str | None = None) -> None:
        self.throttles_before_success = throttles_before_success
        self.in_progress = in_progress
        self.failure_message = failure_message
        self.start_calls = 0

    def start_async_invoke(self, **_kwargs: Any) -> dict[str, str]:
        self.start_calls += 1
        if self.start_calls <= self.throttles_before_success:
            raise _throttle_error(
                'ThrottlingException',
                'Too many requests, please wait before trying again. (reached max retries: 4)',
                cls_name='ThrottlingException',
            )
        return {'invocationArn': 'arn:aws:bedrock:us-west-2:0:async-invoke/abc'}

    def get_async_invoke(self, **_kwargs: Any) -> dict[str, str]:
        if self.failure_message is not None:
            return {'status': 'Failed', 'failureMessage': self.failure_message}
        if self.in_progress:
            return {'status': 'InProgress'}
        return {'status': 'Completed'}


def _install_async_output_mocks(
    monkeypatch: pytest.MonkeyPatch, *, list_error: Exception | None, object_names: list[str] | None = None
) -> tuple[list[str], list[str]]:
    import shutil

    from pixeltable.functions import bedrock
    from pixeltable.utils.object_stores import ObjectPath
    from pixeltable.utils.s3_store import S3Store

    sample = Path(__file__).resolve().parents[1] / 'data' / 'test_videos' / 'test_13_frames.mp4'
    listed: list[str] = []
    deleted: list[str] = []

    class _S3Client:
        def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
            if list_error is not None:
                raise list_error
            prefix = kwargs['Prefix']
            names = ['out.mp4'] if object_names is None else object_names
            keys = [f'{prefix}{name}' for name in names]
            listed.extend(keys)
            if not keys:
                return {}
            return {'Contents': [{'Key': key} for key in keys]}

        def delete_object(self, **kwargs: Any) -> dict[str, Any]:
            deleted.append(kwargs['Key'])
            return {}

    class _Store(S3Store):
        def client(self) -> _S3Client:
            return _S3Client()

    store = _Store(ObjectPath.parse_object_storage_addr('s3://bucket/prefix/', allow_obj_name=False))
    # Capture the bound classmethods before replacing them. Column setup still validates other buckets.
    original_get_store = bedrock.ObjectOps.get_store
    original_copy = bedrock.ObjectOps.copy_object_to_local_file

    def _is_test_bucket(dest: Any) -> bool:
        if isinstance(dest, str):
            return dest.startswith('s3://bucket/')
        return getattr(dest, 'container', None) == 'bucket' and getattr(dest, 'scheme', None) == 's3'

    def _get_store(
        dest: Any, allow_obj_name: bool, col_name: str | None = None, *, scope_credentials: bool = False
    ) -> Any:
        if _is_test_bucket(dest):
            return store
        return original_get_store(dest, allow_obj_name, col_name, scope_credentials=scope_credentials)

    def _copy(src: str, dest: Path) -> None:
        if src.startswith('s3://bucket/'):
            shutil.copyfile(sample, dest)
            return
        original_copy(src, dest)

    monkeypatch.setattr(bedrock, '_get_temp_location', lambda: 's3://bucket/prefix/')
    monkeypatch.setattr(bedrock.ObjectOps, 'get_store', _get_store)
    monkeypatch.setattr(bedrock.ObjectOps, 'copy_object_to_local_file', _copy)
    return listed, deleted


def _insert_luma_ray() -> tuple[Any, str | None, str | None]:
    from pixeltable.functions.bedrock import invoke_model

    t = pxt.create_table('bedrock_sched', {'prompt': pxt.String})
    t.add_computed_column(
        video=invoke_model(
            {'prompt': t.prompt, 'aspect_ratio': '16:9', 'duration': '5s', 'resolution': '540p'},
            model_id=_LUMA_RAY_MODEL_ID,
        )
    )
    status = t.insert(prompt='a dog running on a beach', on_error='ignore')
    row = t.select(errortype=t.video.errortype, errormsg=t.video.errormsg).collect()[0]
    return status, row['errortype'], row['errormsg']


def test_scheduler_retries_throttled_bedrock_invoke(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A throttled StartAsyncInvoke is waited out by the rate-limit scheduler, then retried."""
    from pixeltable.functions import bedrock

    scheduler_sleeps, bedrock_sleeps = _patch_async_clocks(monkeypatch)
    client = _StartCounter(throttles_before_success=1, in_progress=False)
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    _install_async_output_mocks(monkeypatch, list_error=None)

    status, errortype, errormsg = _insert_luma_ray()
    assert status.num_excs == 0, f'{errortype}: {errormsg}'
    assert client.start_calls == 2
    # The first retry has to clear a one-minute quota window, so the scheduler wait is at least 30s.
    assert any(delay >= 30.0 for delay in scheduler_sleeps)
    assert not any(delay >= 30.0 for delay in bedrock_sleeps)


def test_scheduler_stores_throttle_after_max_retries(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eleven throttled submissions are the initial call plus RateLimitsScheduler.MAX_RETRIES.

    The cell then stores the original botocore error, which the live-test skip recognizes.
    """
    from pixeltable.functions import bedrock

    _patch_async_clocks(monkeypatch)
    client = _StartCounter(throttles_before_success=11, in_progress=False)
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    _install_async_output_mocks(monkeypatch, list_error=None)

    status, errortype, errormsg = _insert_luma_ray()
    assert status.num_excs == 1, f'expected the throttle to be stored, got {errortype}: {errormsg}'
    assert client.start_calls == 11
    assert _is_throttled_video_request(errortype, errormsg)


@pytest.mark.parametrize('failure', ['deadline', 'output_read'])
def test_accepted_bedrock_invocation_is_not_retried(
    failure: str, uses_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after StartAsyncInvoke has an invocation ARN does not submit another generation."""
    from pixeltable.functions import bedrock
    from pixeltable.utils.http import is_retryable_error

    _patch_async_clocks(monkeypatch)
    client = _StartCounter(throttles_before_success=0, in_progress=failure == 'deadline')
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    if failure == 'deadline':
        monkeypatch.setattr(bedrock, '_ASYNC_INVOCATION_TIMEOUT_SECS', 0.0)
        list_error: Exception | None = None
    else:
        list_error = _throttle_error(
            'ThrottlingException',
            'Too many requests, timed out reading the output object',
            cls_name='ThrottlingException',
        )
    _install_async_output_mocks(monkeypatch, list_error=list_error)

    status, errortype, errormsg = _insert_luma_ray()
    assert status.num_excs == 1, f'expected a cell error, got {errortype}: {errormsg}'
    assert client.start_calls == 1
    assert errormsg is not None
    assert not is_retryable_error(Exception(errormsg))[0]


@pytest.mark.parametrize(
    ('failure_message', 'object_names', 'message'),
    [
        (None, [], 'No output found'),
        (None, ['manifest.json'], 'only manifest'),
        ('model rejected the prompt', ['manifest.json', 'partial.mp4'], 'model rejected the prompt'),
    ],
)
def test_async_s3_cleanup_deletes_listed_keys(
    failure_message: str | None, object_names: list[str], message: str, uses_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every listed output key is deleted when the job fails or completes without an output file.

    A failed job raises before listing, so there is nothing to delete. Listing and then dropping those
    keys would leave the manifest (and any partial object) in the bucket.
    """
    from pixeltable.functions import bedrock

    _patch_async_clocks(monkeypatch)
    client = _StartCounter(throttles_before_success=0, in_progress=False, failure_message=failure_message)
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    listed, deleted = _install_async_output_mocks(monkeypatch, list_error=None, object_names=object_names)

    status, errortype, errormsg = _insert_luma_ray()
    assert status.num_excs == 1, f'expected a cell error, got {errortype}: {errormsg}'
    assert client.start_calls == 1
    assert errormsg is not None
    assert message in errormsg
    if failure_message is None:
        assert listed == [f'prefix/abc/{name}' for name in object_names]
    assert deleted == listed


def test_validation_exception_with_too_many_requests_is_not_retried(
    uses_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A definitive ValidationException is not retried just because its text says too many requests."""
    from pixeltable.functions import bedrock
    from pixeltable.utils.http import is_retryable_error

    _patch_async_clocks(monkeypatch)

    class _ValidationClient:
        def __init__(self) -> None:
            self.start_calls = 0

        def start_async_invoke(self, **_kwargs: Any) -> dict[str, str]:
            self.start_calls += 1
            raise _throttle_error('ValidationException', 'too many requests were supplied in the input list')

    client = _ValidationClient()
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    _install_async_output_mocks(monkeypatch, list_error=None)

    status, errortype, errormsg = _insert_luma_ray()
    assert status.num_excs == 1, f'expected a cell error, got {errortype}: {errormsg}'
    assert client.start_calls == 1
    assert errormsg is not None
    assert not is_retryable_error(Exception(errormsg))[0]


def test_start_async_read_timeout_is_not_retried(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A StartAsyncInvoke read timeout is not submitted again; Bedrock may already have accepted it."""
    from pixeltable.functions import bedrock
    from pixeltable.utils.http import is_retryable_error

    _patch_async_clocks(monkeypatch)

    class _TimeoutClient:
        def __init__(self) -> None:
            self.start_calls = 0

        def start_async_invoke(self, **_kwargs: Any) -> dict[str, str]:
            self.start_calls += 1
            raise TimeoutError('Read timed out')

    client = _TimeoutClient()
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    _install_async_output_mocks(monkeypatch, list_error=None)

    status, errortype, errormsg = _insert_luma_ray()
    assert status.num_excs == 1, f'expected a cell error, got {errortype}: {errormsg}'
    assert client.start_calls == 1
    assert errormsg is not None
    assert not is_retryable_error(Exception(errormsg))[0]


def test_read_timeout_after_accepted_async_id_is_not_retried(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read timeout after an invocation id is returned does not start another generation.

    The id contains 429, which the generic retry classifier would otherwise treat as a status code.
    """
    from pixeltable.functions import bedrock
    from pixeltable.utils.http import is_retryable_error

    _patch_async_clocks(monkeypatch)

    class _AcceptedTimeout:
        def __init__(self) -> None:
            self.start_calls = 0

        def start_async_invoke(self, **_kwargs: Any) -> dict[str, str]:
            self.start_calls += 1
            return {'invocationArn': 'arn:aws:bedrock:us-west-2:0:async-invoke/abc429defghi'}

        def get_async_invoke(self, **_kwargs: Any) -> dict[str, str]:
            raise TimeoutError('Read timed out')

    client = _AcceptedTimeout()
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    _install_async_output_mocks(monkeypatch, list_error=None)

    status, errortype, errormsg = _insert_luma_ray()
    assert status.num_excs == 1, f'expected a cell error, got {errortype}: {errormsg}'
    assert client.start_calls == 1
    assert errortype == 'ExternalServiceError'
    assert errormsg is not None
    assert '429' not in errormsg
    assert not is_retryable_error(Exception(errormsg))[0]


def test_sync_response_body_timeout_is_not_retried(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A synchronous body read that fails after invoke_model returns does not call Bedrock again."""
    from pixeltable.functions import bedrock
    from pixeltable.functions.bedrock import invoke_model
    from pixeltable.utils.http import is_retryable_error

    _patch_async_clocks(monkeypatch)

    class _TimeoutBody:
        def read(self) -> bytes:
            raise TimeoutError('Read timed out')

    class _SyncClient:
        def __init__(self) -> None:
            self.invoke_calls = 0

        def invoke_model(self, **_kwargs: Any) -> dict[str, Any]:
            self.invoke_calls += 1
            return {'body': _TimeoutBody()}

    client = _SyncClient()
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    t = pxt.create_table('bedrock_sync', {'text': pxt.String})
    t.add_computed_column(response=invoke_model({'inputText': t.text}, model_id='amazon.titan-embed-text-v2:0'))
    status = t.insert(text='hello', on_error='ignore')
    row = t.select(errortype=t.response.errortype, errormsg=t.response.errormsg).collect()[0]
    assert status.num_excs == 1, f'expected a cell error, got {row["errortype"]}: {row["errormsg"]}'
    assert client.invoke_calls == 1
    assert row['errormsg'] is not None
    assert not is_retryable_error(Exception(row['errormsg']))[0]


def test_sync_bedrock_throttle_is_retried(uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit throttle on a synchronous invoke_model is retried by the scheduler."""
    from pixeltable.functions import bedrock
    from pixeltable.functions.bedrock import invoke_model

    scheduler_sleeps, _bedrock_sleeps = _patch_async_clocks(monkeypatch)

    class _JsonBody:
        def read(self) -> bytes:
            return b'{"embedding": [0.25]}'

    class _SyncClient:
        def __init__(self) -> None:
            self.invoke_calls = 0

        def invoke_model(self, **_kwargs: Any) -> dict[str, Any]:
            self.invoke_calls += 1
            if self.invoke_calls == 1:
                raise _throttle_error(
                    'ThrottlingException',
                    'Too many requests, please wait before trying again.',
                    cls_name='ThrottlingException',
                )
            return {'body': _JsonBody()}

    client = _SyncClient()
    monkeypatch.setattr(bedrock, '_bedrock_client', lambda: client)
    t = pxt.create_table('bedrock_sync_throttle', {'text': pxt.String})
    t.add_computed_column(response=invoke_model({'inputText': t.text}, model_id='amazon.titan-embed-text-v2:0'))
    status = t.insert(text='hello', on_error='ignore')
    assert status.num_excs == 0, status
    assert client.invoke_calls == 2
    assert any(delay >= 30.0 for delay in scheduler_sleeps)
    assert t.collect()[0]['response']['embedding'] == [0.25]


def test_unavailable_video_model_error() -> None:
    eol = (
        'An error occurred (ResourceNotFoundException) when calling the StartAsyncInvoke operation: '
        'This model version has reached the end of its life.'
    )
    assert _is_unavailable_video_model_error('ResourceNotFoundException', eol)
    assert _is_unavailable_video_model_error(
        'AccessDeniedException', "You don't have access to the model with the specified model ID."
    )
    assert not _is_unavailable_video_model_error(
        'AccessDeniedException', 'is not authorized to perform: s3:PutObject on resource: s3://bucket/key'
    )
    assert not _is_unavailable_video_model_error(
        'ValidationException', 'Invalid input: resolution must be one of 540p, 720p'
    )
    unresolved = (
        'An error occurred (ResourceNotFoundException) when calling the StartAsyncInvoke operation: '
        'Could not resolve the foundation model from the provided model identifier.'
    )
    assert _is_unavailable_video_model_error('ResourceNotFoundException', unresolved)
    # Type alone must not skip; this message matches none of the unavailable-model snippets.
    assert not _is_unavailable_video_model_error(
        'ResourceNotFoundException',
        'An error occurred (ResourceNotFoundException) when calling the StartAsyncInvoke operation: '
        'The requested resource was not found.',
    )


@pytest.mark.remote_api
@pytest.mark.very_expensive
@pytest.mark.usefixtures('bedrock_us_east_1')
@rerun_on_network_error()
class TestBedrock:
    def test_invoke_model_twelvelabs_marengo_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_computed_column(
            response=invoke_model(
                {'inputType': 'image', 'image': {'mediaSource': {'base64String': t.image}}},
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_marengo_text_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_computed_column(
            response=invoke_model(
                {
                    'inputType': 'text_image',
                    'text_image': {'inputText': 'man walking a dog', 'mediaSource': {'base64String': t.image}},
                },
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_marengo_audio(self, uses_db: None) -> None:
        # Audio auto-routes to StartAsyncInvoke.
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'audio': pxt.Audio | None})
        t.add_computed_column(
            response=invoke_model(
                {'inputType': 'audio', 'audio': {'mediaSource': {'base64String': t.audio}}},
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        audio_filepaths = get_audio_files(extension='.mp3')[:1]
        validate_update_status(t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_marengo_video(self, uses_db: None) -> None:
        # Video auto-routes to StartAsyncInvoke.
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video | None})
        t.add_computed_column(
            response=invoke_model(
                {'inputType': 'video', 'video': {'mediaSource': {'base64String': t.video}}},
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_pegasus_video(self, uses_db: None) -> None:
        """TwelveLabs Pegasus — invoke_model: video (sync)"""
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video | None})
        t.add_computed_column(
            response=invoke_model(
                # Pegasus uses a flat schema: inputPrompt + mediaSource at top level (no inputType nesting)
                {'inputPrompt': 'Describe this video.', 'mediaSource': {'base64String': t.video}},
                model_id='twelvelabs.pegasus-1-2-v1:0',
            )
        )

        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        results = t.select(t.response).collect()
        assert results[0]['response']['message']

    def test_invoke_model_anthropic_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_computed_column(
            response=invoke_model(
                {
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 1024,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'image',
                                    'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': t.image},
                                },
                                {'type': 'text', 'text': 'Describe this image in one sentence.'},
                            ],
                        }
                    ],
                },
                model_id='us.anthropic.claude-haiku-5-5',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        results = t.select(t.response).collect()
        # a response can begin with a thinking block
        assert any(block['type'] == 'text' and block['text'] for block in results[0]['response']['content'])

    def test_converse_anthropic(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('tbl', {'input': pxt.String | None})
        messages = [{'role': 'user', 'content': [{'text': t.input}]}]
        t.add_computed_column(
            output=converse(
                messages,
                model_id='us.anthropic.claude-haiku-5-5',
                system=[{'text': 'You are a helpful assistant. Keep answers short.'}],
                inference_config={'maxTokens': 1024},
                additional_model_request_fields={'output_config': {'effort': 'low'}},
            )
        )
        # a response can begin with a reasoning block, which has no text
        t.add_computed_column(response=t.output.output.message.content['*'].text)
        validate_update_status(t.insert(input='What is the capital of France?'), expected_rows=1)
        assert any('Paris' in text for text in t.collect()[0]['response'] if text is not None)

    @rerun(reruns=3, reruns_delay=8)
    def test_converse_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions import bedrock

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('tbl', {'prompt': pxt.String | None})
            messages = [{'role': 'user', 'content': [{'text': t.prompt}]}]
            t.add_computed_column(
                response=bedrock.converse(messages, model_id='us.anthropic.claude-haiku-5-5', tool_config=tools)
            )
            t.add_computed_column(tool_calls=bedrock.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_multiple_tool_use=False)

    def test_invoke_model_nova_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_computed_column(
            response=invoke_model(
                {
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'image': {'format': 'jpeg', 'source': {'bytes': t.image}}},
                                {'text': 'Describe this image in one sentence.'},
                            ],
                        }
                    ],
                    'inferenceConfig': {'maxTokens': 256},
                },
                model_id='amazon.nova-lite-v1:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_nova_text_response(t.select(t.response).collect())

    def test_invoke_model_nova_video(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video | None})
        t.add_computed_column(
            response=invoke_model(
                {
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'video': {'format': 'mp4', 'source': {'bytes': t.video}}},
                                {'text': 'Describe this video in one sentence.'},
                            ],
                        }
                    ],
                    'inferenceConfig': {'maxTokens': 256},
                },
                model_id='amazon.nova-lite-v1:0',
            )
        )
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        _assert_nova_text_response(t.select(t.response).collect())

    def test_converse_nova_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id='amazon.nova-lite-v1:0'))
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text']

    def test_converse_nova_video(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('tbl', {'video': pxt.Video | None})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'video': {'format': 'mp4', 'source': {'bytes': t.video}}},
                    {'text': 'Describe this video in one sentence.'},
                ],
            }
        ]
        t.add_computed_column(output=converse(messages, model_id='amazon.nova-lite-v1:0'))
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text']

    def test_invoke_model_luma_ray(self, uses_db: None, bedrock_us_west_2: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'prompt': pxt.String | None})
        t.add_computed_column(
            video=invoke_model(
                {'prompt': t.prompt, 'aspect_ratio': '16:9', 'duration': '5s', 'resolution': '540p'},
                model_id=_LUMA_RAY_MODEL_ID,
            )
        )
        status = t.insert(prompt='a dog running on a beach', on_error='ignore')
        if status.num_excs != 0:
            row = t.select(errortype=t.video.errortype, errormsg=t.video.errormsg).collect()[0]
            errortype = row['errortype']
            errormsg = row['errormsg']
            if _is_throttled_video_request(errortype, errormsg):
                # The scheduler retries a continuous throttle up to MAX_RETRIES, then stores the
                # original botocore error. This skip runs after that budget; there is no smaller one.
                pytest.skip(
                    f'Bedrock throttled StartAsyncInvoke for {_LUMA_RAY_MODEL_ID} in {_LUMA_RAY_REGION} '
                    f'after retries: {errormsg}'
                )
            if _is_unavailable_video_model_error(errortype, errormsg):
                pytest.skip(f'Bedrock model {_LUMA_RAY_MODEL_ID} is not available in {_LUMA_RAY_REGION}: {errormsg}')
            raise AssertionError(f'invoke_model({_LUMA_RAY_MODEL_ID}) failed: {errortype}: {errormsg}')
        validate_update_status(status, expected_rows=1)
        result = t.collect()[0]['video']
        assert result is not None and Path(result).exists()

    def test_invoke_model_titan_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'text': pxt.String | None})
        t.add_computed_column(
            response=invoke_model(
                {'inputText': t.text, 'dimensions': 256, 'normalize': True}, model_id='amazon.titan-embed-text-v2:0'
            )
        )
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['response']['embedding']) == 256

    def test_embed_titan_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        model_ids = ['amazon.titan-embed-text-v1', 'amazon.titan-embed-text-v2:0']
        for model_id in model_ids:
            t = pxt.create_table(_tbl_name(model_id), {'text': pxt.String | None})
            t.add_embedding_index('text', string_embed=embed.using(model_id=model_id))
            t.insert(_TEXT_ROWS)
            _assert_text_similarity(t)

    def test_embed_titan_text_custom_dimensions(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String | None})
        t.add_computed_column(embedding=embed(t.text, model_id='amazon.titan-embed-text-v2:0', dimensions=256))
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['embedding']) == 256

    def test_embed_titan_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        img_paths = get_image_files()[:3]
        for model_id in ['amazon.titan-embed-image-v1']:
            t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image | None})
            t.add_embedding_index('image', image_embed=embed.using(model_id=model_id))
            t.insert([{'image': p} for p in img_paths])
            _assert_image_similarity(t, img_paths)

    def test_embed_nova_multimodal_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String | None})
        t.add_embedding_index(
            'text', string_embed=embed.using(model_id='amazon.nova-2-multimodal-embeddings-v1:0', dimensions=1024)
        )
        t.insert(_TEXT_ROWS)
        _assert_text_similarity(t)

    def test_embed_nova_multimodal_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        img_paths = get_image_files()[:3]
        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_embedding_index(
            'image', image_embed=embed.using(model_id='amazon.nova-2-multimodal-embeddings-v1:0', dimensions=1024)
        )
        t.insert([{'image': p} for p in img_paths])
        _assert_image_similarity(t, img_paths)

    def test_invoke_model_nova_multimodal_audio(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'audio': pxt.Audio | None})
        t.add_computed_column(
            response=invoke_model(
                {
                    'taskType': 'SINGLE_EMBEDDING',
                    'singleEmbeddingParams': {
                        'embeddingPurpose': 'GENERIC_INDEX',
                        'embeddingDimension': 1024,
                        'audio': {'format': 'mp3', 'source': {'bytes': t.audio}},
                    },
                },
                model_id='amazon.nova-2-multimodal-embeddings-v1:0',
            )
        )
        audio_filepaths = get_audio_files(extension='.mp3')[:1]
        validate_update_status(t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        assert t.select(t.response).collect()[0]['response']['embeddings']

    def test_invoke_model_nova_multimodal_video(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video | None})
        t.add_computed_column(
            response=invoke_model(
                {
                    'taskType': 'SINGLE_EMBEDDING',
                    'singleEmbeddingParams': {
                        'embeddingPurpose': 'GENERIC_INDEX',
                        'embeddingDimension': 1024,
                        'video': {
                            'format': 'mp4',
                            'embeddingMode': 'AUDIO_VIDEO_COMBINED',
                            'source': {'bytes': t.video},
                        },
                    },
                },
                model_id='amazon.nova-2-multimodal-embeddings-v1:0',
            )
        )
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        assert t.select(t.response).collect()[0]['response']['embeddings']

    def test_converse_ai21(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(['ai21.jamba-1-5-mini-v1:0', 'ai21.jamba-1-5-large-v1:0'])

    def test_embed_cohere_v3_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        for model_id in ['cohere.embed-english-v3', 'cohere.embed-multilingual-v3']:
            t = pxt.create_table(_tbl_name(model_id), {'text': pxt.String | None})
            t.add_embedding_index('text', string_embed=embed.using(model_id=model_id))
            t.insert(_TEXT_ROWS)
            _assert_text_similarity(t)

    def test_embed_cohere_v4_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String | None})
        t.add_computed_column(embedding=embed(t.text, model_id='cohere.embed-v4:0'))
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['embedding']) == 1536

    def test_embed_cohere_v4_text_custom_dimensions(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String | None})
        t.add_computed_column(embedding=embed(t.text, model_id='cohere.embed-v4:0', dimensions=512))
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['embedding']) == 512

    def test_embed_cohere_v4_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        img_paths = get_image_files()[:3]
        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_embedding_index('image', image_embed=embed.using(model_id='cohere.embed-v4:0'))
        t.insert([{'image': p} for p in img_paths])
        _assert_image_similarity(t, img_paths)

    def test_invoke_model_cohere_v4_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image | None})
        t.add_computed_column(
            response=invoke_model(
                {
                    # Cohere embed-v4 uses inputs[] instead of messages[] for embedding requests.
                    'inputs': [{'content': [{'type': 'image_url', 'image_url': {'url': t.image}}]}],
                    'input_type': 'search_document',
                    'embedding_types': ['float'],
                },
                model_id='cohere.embed-v4:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.select(t.response).collect()[0]['response']['embeddings']['float'][0]

    def test_converse_deepseek(self, uses_db: None, bedrock_us_west_2: None) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(
            [
                'us.deepseek.r1-v1:0',  # cross-region inference profile
                'deepseek.v3.2',  # single-region bare ID
            ]
        )

    def test_converse_meta_llama_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'us.meta.llama4-maverick-17b-instruct-v1:0'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image | None})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_mistral_vision_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        model_ids = ['us.mistral.pixtral-large-2502-v1:0', 'mistral.magistral-small-2509']
        image_filepaths = get_image_files()[:1]
        for model_id in model_ids:
            t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image | None})
            t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
            validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
            _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_mistral_vision_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'us.mistral.pixtral-large-2502-v1:0'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image | None})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_converse_openai(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(['openai.gpt-oss-safeguard-20b', 'openai.gpt-oss-safeguard-120b'])

    def test_invoke_model_gemma_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        model_ids = ['google.gemma-3-4b-it', 'google.gemma-3-12b-it', 'google.gemma-3-27b-it']
        image_filepaths = get_image_files()[:1]
        for model_id in model_ids:
            t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image | None})
            t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
            validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
            _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_gemma_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'google.gemma-3-12b-it'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image | None})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_nvidia_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        image_filepaths = get_image_files()[:1]
        model_id = 'nvidia.nemotron-nano-12b-v2'
        t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image | None})
        t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_nvidia_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'nvidia.nemotron-nano-12b-v2'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image | None})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_moonshot_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        image_filepaths = get_image_files()[:1]
        model_id = 'moonshotai.kimi-k2.5'
        t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image | None})
        t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_moonshot_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'moonshotai.kimi-k2.5'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image | None})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_qwen_vl_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        image_filepaths = get_image_files()[:1]
        model_id = 'qwen.qwen3-vl-235b-a22b'
        t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image | None})
        t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_qwen_image_and_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        t = pxt.create_table('tbl_qwen_vl_cv', {'image': pxt.Image | None})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id='qwen.qwen3-vl-235b-a22b'))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text']

        _run_converse_text(['qwen.qwen3-next-80b-a3b'])

    def test_invoke_model_stability_image(self, uses_db: None, bedrock_us_west_2: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'prompt': pxt.String | None})
        t.add_computed_column(
            response=invoke_model(
                {'prompt': t.prompt, 'mode': 'text-to-image', 'aspect_ratio': '1:1', 'output_format': 'jpeg'},
                model_id='stability.sd3-5-large-v1:0',
            )
        )
        validate_update_status(t.insert(prompt='a cat sitting on a mat'), expected_rows=1)
        assert isinstance(t.collect()[0]['response']['images'][0], PIL.Image.Image)

    def test_converse_writer(self, uses_db: None, bedrock_us_west_2: None) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(['us.writer.palmyra-x4-v1:0', 'us.writer.palmyra-x5-v1:0'])
