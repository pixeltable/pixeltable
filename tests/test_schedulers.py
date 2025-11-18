# type: ignore

import types

from pixeltable.exec.expr_eval.schedulers import RequestRateScheduler


class DummyError(Exception):
    pass


class TestSchedulers:
    def test_non_retriable_errors(self) -> None:
        exc = DummyError('dummy')
        assert not RequestRateScheduler._is_retriable_error(exc)[0]

        exc.status = 400
        assert not RequestRateScheduler._is_retriable_error(exc)[0]

        exc = DummyError('dummy')
        exc.status_code = 500
        assert not RequestRateScheduler._is_retriable_error(exc)[0]

        exc = DummyError('dummy')
        exc.status_code = 'IM_A_TEAPOT'
        assert not RequestRateScheduler._is_retriable_error(exc)[0]

    def test_non_retriable_errors_inside_response(self) -> None:
        exc = DummyError('dummy')
        exc.response = types.SimpleNamespace()
        assert not RequestRateScheduler._is_retriable_error(exc)[0]

        exc.response.status = 404
        assert not RequestRateScheduler._is_retriable_error(exc)[0]

        exc = DummyError('dummy')
        exc.response = types.SimpleNamespace()
        exc.response.code = 'BAD_REQUEST'
        assert not RequestRateScheduler._is_retriable_error(exc)[0]

    def test_retriable_errors(self) -> None:
        exc = DummyError('dummy')
        exc.status = 429
        assert RequestRateScheduler._is_retriable_error(exc) == (True, None)
        exc.headers = {'Retry-After': '3'}
        assert RequestRateScheduler._is_retriable_error(exc) == (True, 3)

        exc = DummyError('dummy')
        exc.code = 503
        assert RequestRateScheduler._is_retriable_error(exc) == (True, None)
        exc.headers = {'retryafter': '4', 'Some-Other-Header': 'value'}
        assert RequestRateScheduler._is_retriable_error(exc) == (True, 4)

        exc = DummyError('dummy')
        exc.code = 'too_many_requests'
        assert RequestRateScheduler._is_retriable_error(exc) == (True, None)

    def test_retriable_errors_inside_response(self) -> None:
        exc = DummyError('dummy')
        exc.response = types.SimpleNamespace()
        exc.response.status = 429
        assert RequestRateScheduler._is_retriable_error(exc) == (True, None)
        exc.response.headers = {'Retry-After': '3'}
        assert RequestRateScheduler._is_retriable_error(exc) == (True, 3)

    def test_retriable_errors_based_on_msg(self) -> None:
        exc = DummyError('blah-blah-blah too many requests blah-blah-blah')
        assert RequestRateScheduler._is_retriable_error(exc) == (True, None)

        exc = DummyError('rate exceeded, retry after 123 seconds')
        assert RequestRateScheduler._is_retriable_error(exc) == (True, 123)

        exc = DummyError('（╯ ͡° ل͜ ͡°）╯︵ ┻━┻  request throttled, retry-after:7')  # noqa: RUF001
        assert RequestRateScheduler._is_retriable_error(exc) == (True, 7)

        exc = DummyError('429, try again in 5 seconds')
        assert RequestRateScheduler._is_retriable_error(exc) == (True, 5)

    def test_twelvelabs_exc(self) -> None:
        # Almost the actual error received from TwelveLabs
        exc = DummyError()
        exc.status_code = 429
        exc.args = ()
        exc.body = {
            'code': 'too_many_requests',
            'message': 'You have exceeded the rate limit (100req/1day). '
            'Please try again later after 2025-11-08T01:20:49Z.',
        }
        exc.headers = {
            'date': 'Fri, 07 Nov 2025 22:11:07 GMT',
            'content-type': 'application/json; charset=UTF-8',
            'content-length': '149',
            'connection': 'keep-alive',
            'content-encoding': 'gzip',
            'retry-after': '11382',
            'tl-report': 'backend',
            'vary': 'Accept-Encoding',
            'x-ratelimit-limit': '100',
            'x-ratelimit-remaining': '0',
            'x-ratelimit-reset': '1762564849',
            'x-ratelimit-used': '100',
            'x-trace-id': '122506435189331138',
        }
        assert RequestRateScheduler._is_retriable_error(exc) == (True, 11382)
