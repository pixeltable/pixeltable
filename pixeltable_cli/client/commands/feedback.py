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
from ..utils import EXIT_ERROR, EXIT_REFUSED, agent_name

DEFAULT_URL = 'https://www.pixeltable.com/api/v1/feedback'

KINDS = ('bug', 'gap', 'docs')

MAX_MESSAGE_CHARS = 4000
_MAX_OS_CHARS = 80
_MAX_AGENT_CHARS = 40
_SEND_TIMEOUT_SECS = 15

EPILOG = f"""\
Examples:
  pxt feedback --kind gap "What the task needed, what was tried, what happened"   # prints it, sends nothing
  pxt feedback --kind bug "..." --json                                             # the same, as JSON

What it sends, and only with --send:
  The kind and the text, plus the Pixeltable version, Python version, OS and coding agent this runs
  under. No paths, hostnames, organization or file contents. With DO_NOT_TRACK set to 1, true or yes,
  --send sends nothing.

  The text comes from a project, so a coding agent should show the preview to the person it works for
  and add --send only after they agree.

Exit codes:
  0  previewed, or sent
  1  error: the report could not be sent
  2  usage: no --kind, or a text that is empty or longer than {MAX_MESSAGE_CHARS} characters
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

    url = os.environ.get('PIXELTABLE_FEEDBACK_URL') or DEFAULT_URL
    if '\\' in url:
        # urllib3 ends a URL's host at a backslash and urlsplit does not, so the preview could name another host
        _fail('PIXELTABLE_FEEDBACK_URL contains a backslash')
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
        refusal = 'DO_NOT_TRACK is set, so nothing was sent.'
        if args.as_json:
            print(json.dumps({'error_code': 'DO_NOT_TRACK', 'message': refusal, 'retryable': False}))
        else:
            _print_preview(url, fields, as_json=False)
        _fail(refusal, EXIT_REFUSED)

    report_id = _send(url, fields)
    if args.as_json:
        print(json.dumps({'sent': True} if report_id is None else {'sent': True, 'id': report_id}))
    elif report_id is None:
        print('Sent to the Pixeltable team.')
    else:
        print(f'Sent to the Pixeltable team (reference {report_id}).')


def _agent() -> str | None:
    """The coding agent this runs under, as utils.agent_name() names it, cut to the length the site accepts."""
    name = agent_name()
    # on one line: a newline in AI_AGENT would print a line of its choosing into the preview
    return None if name is None else ' '.join(name.split())[:_MAX_AGENT_CHARS]


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
        print(f'  {label + ":":<13}{fields[key] or "-"}')
    print('  message:')
    print(textwrap.indent(fields['message'], '    '))


def _send(url: str, fields: dict[str, str | None]) -> str | None:
    """POST fields to url; the reference the server gave the report, or None if it gave none."""
    import requests

    # UTF-8 without \uXXXX escapes, so a 4000-character non-ASCII message fits the site's body cap
    body = json.dumps(fields, ensure_ascii=False).encode('utf-8')
    try:
        # a redirect would take the report to a host not named in the preview
        resp = requests.post(
            url,
            data=body,
            headers={'Content-Type': 'application/json'},
            timeout=_SEND_TIMEOUT_SECS,
            allow_redirects=False,
        )
    except requests.RequestException as e:
        _fail(f'could not send ({type(e).__name__})')
    if resp.status_code == 429:
        # seconds only: Retry-After may also be an HTTP date
        wait = resp.headers.get('Retry-After', '').strip()
        _fail(f'not sent (rate limited); try again in {wait} s.' if wait.isdigit() else 'not sent (rate limited).')
    if resp.status_code != 202:
        _fail(f'could not send (HTTP {resp.status_code})')
    try:
        report_id = resp.json()['id']
    except (ValueError, TypeError, KeyError):
        return None
    return report_id if isinstance(report_id, str) else None


def _fail(message: str, exit_code: int = EXIT_ERROR) -> NoReturn:
    # stdout is block-buffered on a pipe; flush the preview before stderr emits the refusal line
    try:
        sys.stdout.flush()
    except OSError:
        pass
    print(f'pxt feedback: {message}', file=sys.stderr)
    sys.exit(exit_code)
