"""`pxt login|logout|whoami` - sign in to Pixeltable Cloud without an API key.

`pxt login` shows a code, you approve it in a browser, and the session lands on this machine. It
does not care whether that took a sign-in or a whole sign-up: it waits for the browser to finish and
returns a session you can use. Every later command reuses it, renewing silently, so an API key is
optional rather than a prerequisite.

The session lives with the daemon, which is the process that presents it on every other command.
This module asks the daemon to start the sign-in, opens the browser, and polls until it is done.
"""

from __future__ import annotations

import json
import sys
import time
import webbrowser
from typing import Any, NoReturn

from ..parser import Parser
from ..utils import get_request, post_request

EPILOG = """\
Examples:
  pxt login                     # sign in, or create an account, in a browser
  pxt whoami                    # who this machine is signed in as, and whether that still works
  pxt logout                    # forget this device's cached session

The session is cached in your Pixeltable home directory, readable only by you, and renews itself in
the background for as long as Pixeltable Cloud honors it. An API key, if you have one set, is used
in preference to it.

Which WorkOS environment to sign in to is answered by the control plane itself, so there is nothing
to configure. The browser need not be on this machine, so this works over SSH.


"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt login', description='sign in to Pixeltable Cloud', epilog=EPILOG)
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    start = post_request('/api/login/start', {})
    # stderr, not stdout: this is progress, and pxt login --json promises a parseable document.
    print(f'Your code is {start["user_code"]}', file=sys.stderr)
    print(f'Confirm it at {start["verification_uri"]}', file=sys.stderr)
    if not webbrowser.open(start['verification_uri']):
        print('Could not open a browser; open the link above.', file=sys.stderr)

    granted = _await_approval(start)
    if args.json_output:
        print(json.dumps({'email': granted['email'], 'organization_id': granted['organization_id']}))
        return
    print(f'Signed in as {granted["email"] or "(unknown)"}.')
    print(_org_line(granted['organization_id']))


def _await_approval(start: dict[str, Any]) -> dict[str, Any]:
    """Poll until the browser is done, and exit with what stopped it when it did not finish.

    The interval and the deadline are the sign-in service's own, and `slow_down` asks for five
    seconds more, per RFC 8628.
    """
    interval = float(start['interval'])
    deadline = time.time() + min(float(start['expires_in']), _LOGIN_TIMEOUT_S)
    poll = {'client_id': start['client_id'], 'device_code': start['device_code']}
    while time.time() < deadline:
        time.sleep(min(interval, max(deadline - time.time(), 0.0)))
        answer = post_request('/api/login/poll', poll)
        status = answer['status']
        if status == 'granted':
            return answer
        if status == 'authorization_pending':
            continue
        if status == 'slow_down':
            interval += 5.0
            continue
        if status == 'access_denied':
            _fail('the sign-in was refused in the browser')
        _fail(f'the sign-in failed ({status})')
    _fail('the code expired before it was confirmed')


def _fail(reason: str) -> NoReturn:
    print(f'pxt login: error: {reason}', file=sys.stderr)
    sys.exit(1)


def run_logout(argv: list[str]) -> None:
    parser = Parser(prog='pxt logout', description="forget this device's cached session")
    parser.parse_args(argv)

    answer = post_request('/api/logout', {})
    print('Signed out.' if answer['signed_out'] else 'Not signed in.')
    if answer['warning'] != '':
        print(f'pxt logout: warning: {answer["warning"]}', file=sys.stderr)

    # A browser still signed in confirms the next code without saying which account it is for, so
    # signing out of one and not the other leaves you as someone you did not choose.
    url = answer['browser_logout_url']
    if url != '':
        print('Signing out of the browser.')
        if not webbrowser.open(url):
            print(f'Could not open a browser; open {url} to sign it out.', file=sys.stderr)


def run_whoami(argv: list[str]) -> None:
    parser = Parser(prog='pxt whoami', description='show the credential commands send, and whether it works')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    parser.add_argument(
        '--offline', action='store_true', help='Report the cached session without asking the control plane'
    )
    args = parser.parse_args(argv)

    answer = get_request('/api/whoami', {'offline': args.offline})
    if args.json_output:
        print(json.dumps(answer))
        sys.exit(0 if answer['using'] != 'none' and answer['accepted'] else 1)

    if answer['using'] == 'none':
        print(f'Not signed in to {answer["api_url"]}. Run `pxt login`.', file=sys.stderr)
        sys.exit(1)

    if answer['email'] != '':
        print(f'{answer["email"]} on {answer["api_url"]}')
        print(_org_line(answer['organization_id']))
    if answer['using'] == 'api_key':
        note = ' An API key always takes precedence over a sign-in.' if answer['email'] != '' else ''
        print(f'Commands use the API key from {answer["credential_source"]}.{note}')
    if not answer['accepted']:
        print(f'That credential was not accepted: {answer["rejection"]}', file=sys.stderr)
        sys.exit(1)


def _org_line(organization_id: str) -> str:
    """Which organization the token is scoped to."""
    return f'Organization: {organization_id or "(none)"}'


# A ceiling on polling. The sign-in service sets the real deadline in expires_in, usually shorter.
_LOGIN_TIMEOUT_S = 600.0
