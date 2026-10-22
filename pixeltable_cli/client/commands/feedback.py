"""`pxt feedback` - send a report to the Pixeltable team, once the person a coding agent works for has read it."""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
import platform
import sys
import textwrap
import unicodedata
import urllib.parse
from typing import NoReturn

from ..parser import Parser
from ..utils import EXIT_ERROR, EXIT_REFUSED, flush_stdout, print_error_document

DEFAULT_SITE_URL = 'https://www.pixeltable.com'
_FEEDBACK_PATH = '/api/v1/feedback'
# the site and the hosts a report may go to over http, as pixeltable/service/trial.py allows `pxt new`; copied, since
# importing pixeltable would cost this command seconds
_LOOPBACK_HOSTS = ('localhost', '127.0.0.1', '::1')

KINDS = ('bug', 'gap', 'docs')

MAX_MESSAGE_CHARS = 4000
_MAX_OS_CHARS = 80
_MAX_AGENT_CHARS = 40
_SEND_TIMEOUT_SECS = 15

# Set by coding agents in the environment of the commands they run, as @vercel/detect-agent reads
# them: variable -> agent. Left out: CURSOR_TRACE_ID and REPL_ID, which every Cursor terminal and
# Replit shell carries, and the COPILOT_* settings, which a person's shell profile can export.
_AGENT_MARKERS = {
    'CURSOR_AGENT': 'cursor',
    'GEMINI_CLI': 'gemini',
    'CODEX_SANDBOX': 'codex',
    'CODEX_CI': 'codex',
    'CODEX_THREAD_ID': 'codex',
    'ANTIGRAVITY_AGENT': 'antigravity',
    'AUGMENT_AGENT': 'augment',
    'OPENCODE_CLIENT': 'opencode',
    'CLAUDECODE': 'claude',
    'CLAUDE_CODE': 'claude',
}

EPILOG = f"""\
Examples:
  pxt feedback --kind gap 'What the task needed, what was tried, what happened'   # prints it, sends nothing
  pxt feedback --kind bug '...' --json                                             # the same, as JSON

  Single quotes keep the shell from running a `command` or expanding a $VARIABLE in the text.

What it sends, and only with --send:
  The kind and the text, plus the Pixeltable version, Python version, OS and coding agent this runs
  under. Besides the text, which goes as written, it sends no paths, hostnames, organization or file
  contents. With DO_NOT_TRACK set to 1, true or yes, --send sends nothing. PIXELTABLE_SITE_URL sends
  it to another site, over https or to localhost.

  The text comes from a project, so a coding agent should show the preview to the person it works for
  and add --send only after they agree.

Exit codes:
  0  previewed, or sent
  1  error: the report was not sent, or not confirmed as sent
  2  usage: no --kind, or a text that is empty, longer than {MAX_MESSAGE_CHARS} characters, or holds a control character
  3  refused: DO_NOT_TRACK is set"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt feedback', description='send a report to the Pixeltable team', epilog=EPILOG)
    parser.add_argument('--kind', required=True, choices=KINDS, help='what the report is about')
    parser.add_argument(
        'text', metavar='TEXT', help=f'the report, quoted as one argument (1 to {MAX_MESSAGE_CHARS} characters)'
    )
    parser.add_argument(
        '--send', action='store_true', help='send it, once the person you work for has read the preview'
    )
    parser.add_argument('--json', action='store_true', dest='as_json', help='Emit JSON output')
    args = parser.parse_args(argv)
    # a pipe on Windows encodes with the locale's code page, which may lack a character of the text
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors='backslashreplace')

    # a carriage return would hide what precedes it on the preview's line, and still be sent
    message = args.text.replace('\r\n', '\n').replace('\r', '\n').strip()
    if len(message) == 0:
        parser.error('TEXT is empty')
    if len(message) > MAX_MESSAGE_CHARS:
        parser.error(f'TEXT is {len(message)} characters; the limit is {MAX_MESSAGE_CHARS}')
    if any(unicodedata.category(ch) == 'Cc' and ch not in '\n\t' for ch in message):
        parser.error('TEXT contains a control character other than a newline or a tab')

    url = _feedback_url(args.as_json)
    fields: dict[str, str | None] = {
        'kind': args.kind,
        'message': message,
        'pixeltable_version': importlib.metadata.version('pixeltable'),
        'python_version': platform.python_version(),
        'os': _os_name(),
        'agent': _agent(),
    }
    if not args.send:
        _print_preview(url, fields, args.as_json)
        if not args.as_json:
            print('Nothing was sent. Sending needs --send, once the person you work for has read this.')
        return
    if os.environ.get('DO_NOT_TRACK', '').strip().lower() in ('1', 'true', 'yes'):
        if not args.as_json:
            _print_preview(url, fields, as_json=False)
        _fail('DO_NOT_TRACK is set, so nothing was sent.', 'DO_NOT_TRACK', args.as_json, exit_code=EXIT_REFUSED)

    report_id = _send(url, fields, args.as_json)
    if args.as_json:
        print(json.dumps({'sent': True} if report_id is None else {'sent': True, 'id': report_id}))
    elif report_id is None:
        print('Sent to the Pixeltable team.')
    else:
        print(f'Sent to the Pixeltable team (reference {report_id}).')


def _feedback_url(as_json: bool) -> str:
    site = (os.environ.get('PIXELTABLE_SITE_URL') or DEFAULT_SITE_URL).rstrip('/')
    if '\\' in site:
        # urllib3 ends a URL's host at a backslash and urlsplit does not, so the preview could name another host
        _fail('PIXELTABLE_SITE_URL contains a backslash', 'INVALID_CONFIGURATION', as_json)
    # the URL itself is not printed: it may hold credentials
    try:
        parts = urllib.parse.urlsplit(site)
        hostname = parts.hostname
    except ValueError:
        _fail('PIXELTABLE_SITE_URL is not a valid URL', 'INVALID_CONFIGURATION', as_json)
    if not (parts.scheme == 'https' or (parts.scheme == 'http' and hostname in _LOOPBACK_HOSTS)):
        _fail('PIXELTABLE_SITE_URL must use https, except on localhost', 'INVALID_CONFIGURATION', as_json)
    return site + _FEEDBACK_PATH


def _agent() -> str | None:
    """The coding agent this runs under, on one line and cut to the length the site accepts; None for none.

    AI_AGENT, which any agent can set to its own name, comes first. It only names the agent in the report.
    """
    # on one line, with no control character: either would let AI_AGENT print a line of its choosing into the preview
    raw = os.environ.get('AI_AGENT', '')
    declared = ' '.join(''.join(' ' if unicodedata.category(ch) == 'Cc' else ch for ch in raw).split())
    if declared != '':
        return declared[:_MAX_AGENT_CHARS]
    if os.environ.get('CURSOR_EXTENSION_HOST_ROLE') == 'agent-exec':
        return 'cursor'
    return next((agent for marker, agent in _AGENT_MARKERS.items() if os.environ.get(marker)), None)


def _os_name() -> str:
    """The OS, its release and the machine architecture, e.g. 'macOS-15.1-arm64'."""
    name = platform.platform(terse=True)
    machine = platform.machine()
    # terse drops the architecture on macOS and Windows; Linux keeps it, along with the libc version
    if len(machine) > 0 and machine not in name:
        name = f'{name}-{machine}'
    return name[:_MAX_OS_CHARS]


def _destination(url: str) -> str:
    # the host and port only: credentials in an overriding URL stay out of the output
    return urllib.parse.urlsplit(url).netloc.rpartition('@')[2]


def _print_preview(url: str, fields: dict[str, str | None], as_json: bool) -> None:
    destination = _destination(url)
    if as_json:
        print(json.dumps({'destination': destination, 'fields': fields, 'sent': False}, indent=2))
        return
    print(f'This would go to the Pixeltable team at {destination}:')
    for label, key in (
        ('kind', 'kind'),
        ('pixeltable', 'pixeltable_version'),
        ('python', 'python_version'),
        ('os', 'os'),
        ('agent', 'agent'),
    ):
        print(f'  {label + ":":<13}{_visible(fields[key] or "-")}')
    print('  message:')
    print(textwrap.indent(_visible(fields['message']), '    '))


def _visible(text: str) -> str:
    # a format character such as U+202E reorders the line a terminal shows, and a zero-width one hides; spell them out
    return ''.join(ch if ch.isprintable() or ch in '\n\t' else ascii(ch)[1:-1] for ch in text)


def _send(url: str, fields: dict[str, str | None], as_json: bool) -> str | None:
    """POST fields to url; the reference the server gave the report, or None if it gave none."""
    import requests

    try:
        # a redirect would take the report to a host not named in the preview
        resp = requests.post(url, json=fields, timeout=_SEND_TIMEOUT_SECS, allow_redirects=False)
    except requests.RequestException as e:
        if _may_have_arrived(e):
            _unconfirmed(type(e).__name__, as_json)
        _fail(f'could not send ({type(e).__name__})', 'FEEDBACK_NOT_SENT', as_json, retryable=True)
    if resp.status_code == 429:
        # seconds only: Retry-After may also be an HTTP date; isdigit() alone takes '²'
        wait = resp.headers.get('Retry-After', '').strip()
        retry_after = int(wait) if wait.isascii() and wait.isdigit() else None
        message = 'not sent (rate limited)' + ('' if retry_after is None else f'; try again in {retry_after} s')
        _fail(f'{message}.', 'RATE_LIMITED', as_json, retryable=True, retry_after=retry_after)
    if resp.status_code >= 500:
        _unconfirmed(f'HTTP {resp.status_code}', as_json)
    if resp.status_code != 202:
        _fail(f'could not send (HTTP {resp.status_code})', 'FEEDBACK_NOT_SENT', as_json)
    try:
        report_id = resp.json()['id']
    except (ValueError, TypeError, KeyError):
        return None
    return report_id if isinstance(report_id, str) else None


def _may_have_arrived(e: Exception) -> bool:
    """Whether the body may have reached the site: the failure came after the connection opened."""
    import requests
    from urllib3.exceptions import MaxRetryError, NewConnectionError

    if isinstance(e, (requests.ReadTimeout, requests.exceptions.ChunkedEncodingError)):
        return True
    if isinstance(e, (requests.ConnectTimeout, requests.exceptions.ProxyError, requests.exceptions.SSLError)):
        return False
    if isinstance(e, requests.ConnectionError):
        # a refused or unresolved connection comes as a MaxRetryError of a NewConnectionError; a connection dropped
        # after the send, as a ProtocolError
        cause = e.args[0] if len(e.args) > 0 else None
        return not (isinstance(cause, MaxRetryError) and isinstance(cause.reason, NewConnectionError))
    return False


def _unconfirmed(cause: str, as_json: bool) -> NoReturn:
    # the site answers 5xx when Slack does not confirm the post, and a timeout can follow a post that arrived; a resend
    # may duplicate the report, so this is not retryable
    _fail(f'not confirmed as sent ({cause}); it may have arrived', 'FEEDBACK_UNCONFIRMED', as_json)


def _fail(
    message: str,
    error_code: str,
    as_json: bool,
    *,
    retryable: bool = False,
    retry_after: int | None = None,
    exit_code: int = EXIT_ERROR,
) -> NoReturn:
    if as_json:
        print_error_document(error_code, message, retryable=retryable, retry_after=retry_after)
    flush_stdout()
    print(f'pxt feedback: {message}', file=sys.stderr)
    sys.exit(exit_code)
