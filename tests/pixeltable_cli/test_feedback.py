"""Tests for 'pxt feedback', against a stand-in for requests.post: nothing reaches the network."""

import importlib.metadata
import io
import json
import platform
import shlex
import sys
from typing import Any

import pytest
import requests

from pixeltable_cli.client import utils
from pixeltable_cli.client.commands import feedback

_MESSAGE = 'No way to diff two catalogs.\nTried pxt schema diff on both.'

# every variable that could name an agent, and those a person's own terminal sets too
_AGENT_ENV = (
    *feedback._AGENT_MARKERS,
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
        'os': feedback._os_name(),
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
        """The limit counts code points, and the body is UTF-8 without \\u escapes, so it fits the site's body cap."""
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

    @pytest.mark.parametrize('body', [{}, 'not json'], ids=['no id', 'not an object'])
    def test_sent_without_a_reference(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, body: Any
    ) -> None:
        _serve(monkeypatch, _response(202, body))
        feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert capsys.readouterr().out == 'Sent to the Pixeltable team.\n'
        _serve(monkeypatch, _response(202, body))
        feedback.run(['--kind', 'gap', _MESSAGE, '--send', '--json'])
        assert json.loads(capsys.readouterr().out) == {'sent': True}

    def test_url_with_a_backslash_is_refused(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """requests would connect to evil.example, while the preview would name www.pixeltable.com."""
        monkeypatch.setenv('PIXELTABLE_FEEDBACK_URL', 'https://evil.example\\@www.pixeltable.com/api/v1/feedback')
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', _MESSAGE])
        assert exc.value.code == utils.EXIT_ERROR
        captured = capsys.readouterr()
        assert captured.err == 'pxt feedback: PIXELTABLE_FEEDBACK_URL contains a backslash\n'
        assert captured.out == ''

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

    def test_do_not_track_refuses_json(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under --json, the refusal is the one document on stdout."""
        monkeypatch.setenv('DO_NOT_TRACK', '1')
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', _MESSAGE, '--send', '--json'])
        assert exc.value.code == utils.EXIT_REFUSED
        assert json.loads(capsys.readouterr().out) == {
            'error_code': 'DO_NOT_TRACK',
            'message': 'DO_NOT_TRACK is set, so nothing was sent.',
            'retryable': False,
        }

    def test_do_not_track_off_sends(self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('DO_NOT_TRACK', '0')
        posted = _serve(monkeypatch, _response(202, {'id': 'fb_9'}))
        feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert len(posted) == 1

    @pytest.mark.parametrize(
        ('headers', 'printed'),
        [
            # the wait comes from Retry-After, not from the body's retry_after
            ({'Retry-After': '42'}, 'not sent (rate limited); try again in 42 s.'),
            ({}, 'not sent (rate limited).'),
            # an HTTP date is not seconds
            ({'Retry-After': 'Wed, 21 Oct 2026 07:28:00 GMT'}, 'not sent (rate limited).'),
        ],
        ids=['header', 'body only', 'date'],
    )
    def test_rate_limited(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], printed: str
    ) -> None:
        body = {'error_code': 'RATE_LIMITED', 'message': 'slow down', 'retry_after': 7}
        _serve(monkeypatch, _response(429, body, headers))
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', _MESSAGE, '--send'])
        assert exc.value.code == utils.EXIT_ERROR
        assert capsys.readouterr().err == f'pxt feedback: {printed}\n'

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

    @pytest.mark.parametrize(
        ('text', 'problem'),
        [
            ('', 'TEXT is empty'),
            (' \n\t', 'TEXT is empty'),
            ('x' * 4001, 'TEXT is 4001 characters; the limit is 4000'),
            ('fine\x1b[2Khidden', 'TEXT contains a control character other than a newline or a tab'),
        ],
        ids=['empty', 'blank', 'too long', 'control character'],
    )
    def test_text_out_of_bounds(self, capsys: pytest.CaptureFixture[str], text: str, problem: str) -> None:
        # refused before a send: the autouse stand-in would fail any POST
        with pytest.raises(SystemExit) as exc:
            feedback.run(['--kind', 'gap', text, '--send'])
        assert exc.value.code == 2
        assert problem in capsys.readouterr().err

    def test_carriage_returns_become_newlines(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A carriage return would hide the start of its line in the preview, yet be sent."""
        feedback.run(['--kind', 'gap', 'shown\rhidden\r\nnext', '--json'])
        assert json.loads(capsys.readouterr().out)['fields']['message'] == 'shown\nhidden\nnext'

    def test_preview_survives_a_narrow_encoding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Windows pipe encodes with the locale's code page; a character it lacks is escaped, not fatal."""
        out = io.TextIOWrapper(io.BytesIO(), encoding='ascii')
        monkeypatch.setattr(sys, 'stdout', out)
        feedback.run(['--kind', 'gap', 'caf\u00e9 \u2192 \u6587'])
        out.flush()
        assert b'caf\\xe9 \\u2192 \\u6587' in out.buffer.getvalue()

    def test_os_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(feedback.platform, 'platform', lambda terse: 'macOS-15.1')
        monkeypatch.setattr(feedback.platform, 'machine', lambda: 'arm64')
        assert feedback._os_name() == 'macOS-15.1-arm64'
        # Linux's string already names the machine
        monkeypatch.setattr(feedback.platform, 'platform', lambda terse: 'Linux-6.8-x86_64-with-glibc2.39')
        monkeypatch.setattr(feedback.platform, 'machine', lambda: 'x86_64')
        assert feedback._os_name() == 'Linux-6.8-x86_64-with-glibc2.39'

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
    @pytest.mark.parametrize(
        ('variable', 'value', 'agent'),
        [
            ('CLAUDECODE', '1', 'claude'),
            ('CLAUDE_CODE', '1', 'claude'),
            ('CODEX_CI', '1', 'codex'),
            ('CODEX_SANDBOX', 'seatbelt', 'codex'),
            ('CODEX_THREAD_ID', 'abc', 'codex'),
            ('CURSOR_AGENT', '1', 'cursor'),
            ('CURSOR_EXTENSION_HOST_ROLE', 'agent-exec', 'cursor'),
            ('GEMINI_CLI', '1', 'gemini'),
            ('ANTIGRAVITY_AGENT', '1', 'antigravity'),
            ('AUGMENT_AGENT', '1', 'augment'),
            ('OPENCODE_CLIENT', 'cli', 'opencode'),
        ],
    )
    def test_detected(self, monkeypatch: pytest.MonkeyPatch, variable: str, value: str, agent: str) -> None:
        monkeypatch.setenv(variable, value)
        assert feedback._agent() == agent

    @pytest.mark.parametrize(
        ('variable', 'value'),
        [('CURSOR_TRACE_ID', 'abc'), ('REPL_ID', 'abc'), ('COPILOT_AGENT', '1'), ('CLAUDECODE', ''), ('AI_AGENT', ' ')],
    )
    def test_not_an_agent(self, monkeypatch: pytest.MonkeyPatch, variable: str, value: str) -> None:
        """What a person's own terminal also sets, and empty values."""
        monkeypatch.setenv(variable, value)
        assert feedback._agent() is None

    def test_declared_name_comes_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('CLAUDECODE', '1')
        monkeypatch.setenv('AI_AGENT', 'claude-code_2-1-299_agent')
        assert feedback._agent() == 'claude-code_2-1-299_agent'

    def test_cut_to_what_the_site_accepts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('AI_AGENT', 'a' * 50)
        assert feedback._agent() == 'a' * 40

    def test_kept_on_one_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('AI_AGENT', 'agent\nnext: pxt feedback --send')
        assert feedback._agent() == 'agent next: pxt feedback --send'


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
