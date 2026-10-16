"""Tests for 'pxt feedback', against a stand-in for requests.post: nothing reaches the network."""

import importlib.metadata
import json
import platform
import shlex
from typing import Any

import pytest
import requests

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import feedback

_MESSAGE = 'No way to diff two catalogs.\nTried pxt schema diff on both.'

# every variable that could name an agent, and those a person's own terminal sets too
_AGENT_ENV = (
    *utils._AGENT_MARKERS,
    'AI_AGENT',
    'CURSOR_EXTENSION_HOST_ROLE',
    'CURSOR_TRACE_ID',
    'REPL_ID',
    'COPILOT_AGENT',
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No agent, DO_NOT_TRACK or URL override from the environment the tests run in, and no network."""
    for var in (*_AGENT_ENV, 'DO_NOT_TRACK', 'PIXELTABLE_FEEDBACK_URL'):
        monkeypatch.delenv(var, raising=False)

    def unexpected_post(url: str, **kwargs: Any) -> requests.Response:
        raise AssertionError(f'unexpected POST to {url}')

    monkeypatch.setattr(requests, 'post', unexpected_post)


def _response(status: int, body: Any = None, headers: dict[str, str] | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = b'' if body is None else json.dumps(body).encode()
    resp.headers.update(headers or {})
    return resp


def _serve(monkeypatch: pytest.MonkeyPatch, answer: requests.Response | Exception) -> list[tuple[str, dict]]:
    """Answer every POST with answer; returns (url, keyword arguments) of each POST."""
    posted: list[tuple[str, dict]] = []

    def post(url: str, **kwargs: Any) -> requests.Response:
        posted.append((url, kwargs))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(requests, 'post', post)
    return posted


def _expected_fields(kind: str = 'gap', agent: str | None = None) -> dict[str, str | None]:
    return {
        'kind': kind,
        'message': _MESSAGE,
        'pixeltable_version': importlib.metadata.version('pixeltable'),
        'python_version': platform.python_version(),
        'os': feedback.os_name(),
        'agent': agent,
    }


class TestFeedback:
    def test_preview_sends_nothing(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        """Without --send, every field is shown and no command an agent could run to send it is printed."""
        monkeypatch.setenv('CLAUDECODE', '1')
        feedback.run(['--kind', 'gap', f'  {_MESSAGE}\n'])  # surrounding whitespace is stripped
        fields = _expected_fields(agent='claude')
        assert capsys.readouterr().out == (
            'This would go to the Pixeltable team at www.pixeltable.com:\n'
            '  kind:        gap\n'
            f'  pixeltable:  {fields["pixeltable_version"]}\n'
            f'  python:      {fields["python_version"]}\n'
            f'  os:          {fields["os"]}\n'
            '  agent:       claude\n'
            '  message:\n'
            '    No way to diff two catalogs.\n'
            '    Tried pxt schema diff on both.\n'
            'Nothing was sent. Sending needs --send, once the person you work for has read this.\n'
        )

    def test_preview_without_agent(self, capsys: pytest.CaptureFixture[str]) -> None:
        feedback.run(['--kind', 'docs', _MESSAGE])
        out = capsys.readouterr().out
        assert '  agent:       -\n' in out
        assert not any(line.startswith('next:') or 'pxt feedback' in line for line in out.splitlines())

    def test_preview_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        feedback.run(['--kind', 'gap', _MESSAGE, '--json'])
        assert json.loads(capsys.readouterr().out) == {
            'destination': 'www.pixeltable.com',
            'fields': _expected_fields(),
            'sent': False,
        }

    def test_send_posts_the_six_fields(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('AI_AGENT', 'my-agent')
        posted = _serve(monkeypatch, _response(202, {'id': 'fb_123'}))
        feedback.run(['--kind', 'bug', _MESSAGE, '--send'])
        assert len(posted) == 1
        url, kwargs = posted[0]
        assert url == feedback.DEFAULT_URL
        assert kwargs['headers'] == {'Content-Type': 'application/json'}
        assert json.loads(kwargs['data'].decode('utf-8')) == _expected_fields(kind='bug', agent='my-agent')
        assert kwargs['timeout'] == 15
        # a redirect is not followed to a host the preview did not name
        assert kwargs['allow_redirects'] is False
        assert capsys.readouterr().out == 'Sent to the Pixeltable team (reference fb_123).\n'

    def test_non_ascii_sent_unescaped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The limit counts code points, and the body is UTF-8 without \\u escapes, so it fits the site's 16 KB cap."""
        message = '\U0001f600' * 4000  # 4 bytes each in UTF-8; 12 as an escaped surrogate pair
        posted = _serve(monkeypatch, _response(202, {'id': 'fb_7'}))
        feedback.run(['--kind', 'gap', message, '--send'])
        data = posted[0][1]['data']
        assert json.loads(data.decode('utf-8'))['message'] == message
        assert message in data.decode('utf-8') and b'\\u' not in data
        assert len(data) <= 16 * 1024

    def test_send_json(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        _serve(monkeypatch, _response(202, {'id': 'fb_123'}))
        feedback.run(['--kind', 'gap', _MESSAGE, '--send', '--json'])
        assert json.loads(capsys.readouterr().out) == {'sent': True, 'id': 'fb_123'}

    def test_url_override(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_FEEDBACK_URL', 'http://user:pw@localhost:3000/api/v1/feedback')
        feedback.run(['--kind', 'gap', _MESSAGE])
        # the preview names the host it would go to, without the URL's credentials
        assert 'This would go to the Pixeltable team at localhost:3000:' in capsys.readouterr().out

        posted = _serve(monkeypatch, _response(202, {'id': 'fb_1'}))
        feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert [url for url, _ in posted] == ['http://user:pw@localhost:3000/api/v1/feedback']

    @pytest.mark.parametrize('value', ['1', 'TRUE', 'yes'])
    def test_do_not_track_refuses(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """DO_NOT_TRACK turns --send into a preview that exits 3; the autouse stand-in fails any POST."""
        monkeypatch.setenv('DO_NOT_TRACK', value)
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert exc.value.code == utils.EXIT_REFUSED
        captured = capsys.readouterr()
        assert captured.out.startswith('This would go to the Pixeltable team at www.pixeltable.com:\n')
        assert 'Sending needs --send' not in captured.out
        assert captured.err == 'pxt feedback: DO_NOT_TRACK is set, so nothing was sent.\n'

    def test_do_not_track_off_sends(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('DO_NOT_TRACK', '0')
        posted = _serve(monkeypatch, _response(202, {'id': 'fb_9'}))
        feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert len(posted) == 1

    def test_rate_limited(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        body = {'error_code': 'RATE_LIMITED', 'message': 'slow down', 'retry_after': 42}
        _serve(monkeypatch, _response(429, body, {'Retry-After': '42'}))
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert exc.value.code == utils.EXIT_ERROR
        assert capsys.readouterr().err == 'pxt feedback: not sent (rate limited); try again in 42 s.\n'

    @pytest.mark.parametrize(
        ('answer', 'cause'),
        [
            (_response(503, {'error_code': 'FEEDBACK_UNAVAILABLE', 'message': 'down'}), 'HTTP 503'),
            # the server's message may quote the text, so it is not printed
            (_response(400, {'error_code': 'INVALID_FEEDBACK', 'message': _MESSAGE}), 'HTTP 400'),
            (_response(308, headers={'Location': 'https://elsewhere.example/'}), 'HTTP 308'),
            (requests.ConnectionError(f'refused: {_MESSAGE}'), 'ConnectionError'),
            (requests.ReadTimeout('timed out'), 'ReadTimeout'),
        ],
        ids=['unavailable', 'invalid', 'redirect', 'no connection', 'timeout'],
    )
    def test_send_failure(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        answer: requests.Response | Exception,
        cause: str,
    ) -> None:
        """A failure ends in one line naming its cause, never the text sent."""
        _serve(monkeypatch, answer)
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert exc.value.code == utils.EXIT_ERROR
        captured = capsys.readouterr()
        assert captured.err == f'pxt feedback: could not send ({cause})\n'
        assert captured.out == ''

    @pytest.mark.parametrize('text', ['', ' \n\t', 'x' * 4001], ids=['empty', 'blank', 'too long'])
    def test_text_out_of_bounds(self, capsys: pytest.CaptureFixture[str], text: str) -> None:
        # refused before a send: the autouse stand-in would fail any POST
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', text, '--send'])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert 'TEXT is empty' in err or 'TEXT is 4001 characters; the limit is 4000' in err

    def test_text_at_limit(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The limit applies after stripping, so surrounding whitespace does not count."""
        feedback.run(['--kind', 'gap', f'  {"x" * 4000}\n', '--json'])
        assert json.loads(capsys.readouterr().out)['fields']['message'] == 'x' * 4000

    def test_kind_required(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            feedback.run([_MESSAGE])
        assert exc.value.code == 2
        assert 'the following arguments are required: --kind' in capsys.readouterr().err


class TestAgentName:
    def test_named_as_utils_names_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('CLAUDECODE', '1')
        assert feedback._agent() == 'claude'

    def test_cut_to_what_the_site_accepts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('AI_AGENT', 'a' * 50)
        assert feedback._agent() == 'a' * 40


class TestAfterTraceback:
    def test_an_unexpected_daemon_failure_names_feedback(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The traceback of a daemon failure ends with the command that previews a report, which runs as given."""
        utils.RequestFailedError(
            '500 ValueError: boom', 500, server_tb='Traceback (most recent call last): ...'
        ).report()
        [line] = [line for line in capsys.readouterr().err.splitlines() if line.startswith('next: ')]
        argv = shlex.split(line.removeprefix('next: '), comments=True)
        assert argv[:4] == ['pxt', 'feedback', '--kind', 'bug']

        feedback.run(argv[2:])
        assert 'Nothing was sent.' in capsys.readouterr().out

    def test_an_expected_error_does_not(self, capsys: pytest.CaptureFixture[str]) -> None:
        utils.RequestFailedError('404 no such table', 404, retryable=False).report()
        assert 'feedback' not in capsys.readouterr().err
