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
from .new import trial_fate

EPILOG = """\
Examples:
  pxt login                     # sign in, or create an account, in a browser
  pxt whoami                    # who this machine is signed in as, and whether Pixeltable Cloud recognizes it
  pxt logout                    # forget this device's cached session, or its `pxt new` trial

The session is cached in your Pixeltable home directory. When its token expires, the next command
that needs a token renews the session, for as long as Pixeltable Cloud honors it. An API key, if
you have one set, is used in preference to it.

The control plane says where to sign in, so there is nothing to configure. The browser need not be
on this machine, so this works over SSH.


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
    replaced = granted['replaced_trial']
    if replaced is not None and not replaced['expired']:
        # the claim link was on this machine only in the trial's record
        print(
            f'pxt login: warning: this machine no longer uses the trial pxt://{replaced["org"]}:{replaced["db"]}. '
            f'{trial_fate(replaced)}',
            file=sys.stderr,
        )
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
    deadline = time.monotonic() + min(float(start['expires_in']), _LOGIN_TIMEOUT_S)
    poll = {'client_id': start['client_id'], 'device_code': start['device_code']}
    while True:
        time.sleep(min(interval, max(deadline - time.monotonic(), 0.0)))
        if time.monotonic() >= deadline:
            _fail(_EXPIRED)
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
        if status == 'expired_token':
            _fail(_EXPIRED)
        detail = f': {answer["detail"]}' if answer['detail'] != '' else ''
        _fail(f'the sign-in failed ({status}{detail})')


_EXPIRED = 'the code expired before it was confirmed'


def _fail(reason: str) -> NoReturn:
    print(f'pxt login: error: {reason}', file=sys.stderr)
    sys.exit(1)


def run_logout(argv: list[str]) -> None:
    parser = Parser(prog='pxt logout', description="forget this device's cached session")
    parser.parse_args(argv)

    answer = post_request('/api/logout', {})
    print('Signed out.' if answer['signed_out'] else 'Not signed in.')
    trial = answer['trial']
    if trial is not None:
        print(f'This machine no longer uses the trial pxt://{trial["org"]}:{trial["db"]}. {trial_fate(trial)}')
    if answer['warning'] != '':
        print(f'pxt logout: warning: {answer["warning"]}', file=sys.stderr)

    # A browser still signed in confirms the next code without naming the account, so signing out of
    # only one leaves you signed in as someone you did not choose.
    url = answer['browser_logout_url']
    if url != '':
        print('Signing out of the browser.')
        if not webbrowser.open(url):
            print(f'Could not open a browser; open {url} to sign it out.', file=sys.stderr)


def run_whoami(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt whoami', description='show the credential commands send, and whether Pixeltable Cloud recognizes it'
    )
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    parser.add_argument(
        '--offline', action='store_true', help='Report the cached session without asking the control plane'
    )
    args = parser.parse_args(argv)

    answer = get_request('/api/whoami', {'offline': args.offline})
    if args.json_output:
        print(json.dumps(answer))
        sys.exit(0 if answer['accepted'] else 1)

    if answer['using'] == 'none':
        print(f'Not signed in to {answer["api_url"]}. Run `pxt login`.', file=sys.stderr)
        sys.exit(1)

    trial = answer['trial']
    if answer['using'] == 'session':
        print(f'{answer["email"] or "(unknown)"} on {answer["api_url"]}')
        print(_org_line(answer['organization_id']))
    elif trial is not None:
        print(f'Trial pxt://{trial["org"]}:{trial["db"]} on {answer["api_url"]}')
        print(trial_fate(trial))
    else:
        print(f'API key on {answer["api_url"]}')
    # a rejection or a note already says which credential was sent
    if not answer['accepted']:
        print(answer['rejection'], file=sys.stderr)
        sys.exit(1)
    if answer['note'] != '':
        print(answer['note'])
    elif answer['using'] == 'api_key':
        print(f'Commands use the API key from {answer["credential_source"]}.')


def _org_line(organization_id: str) -> str:
    """Which organization the token is scoped to."""
    if organization_id == '':
        return 'No organization yet: create one with `pxt org create NAME`'
    return f'Organization: {organization_id}'


# A ceiling on polling. The sign-in service sets the real deadline in expires_in, usually shorter.
_LOGIN_TIMEOUT_S = 600.0
