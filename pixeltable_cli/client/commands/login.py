"""`pxt login|logout|whoami` - sign in to Pixeltable Cloud without an API key.

`pxt login` shows a code, you approve it in a browser, and the session lands on this machine. It
does not care whether that took a sign-in or a whole sign-up: it waits for the browser to finish and
returns a session you can use. Every later command reuses it, renewing silently, so an API key is
optional rather than a prerequisite.
"""

from __future__ import annotations

import argparse
import json
import sys

from pixeltable.service import auth, credentials, management_client
from pixeltable.service.auth import AuthError
from pixeltable.service.management_client import api_url

from ..parser import Parser

EPILOG = """\
Examples:
  pxt login                     # sign in, or create an account, in a browser
  pxt whoami                    # who this machine is signed in as, and for how long
  pxt logout                    # forget this device's cached session

The session is cached in your Pixeltable home directory, readable only by you, and renews itself
for an hour after you sign in; after that, sign in again. An API key, if you have one set, is used
in preference to it, and does not expire.

Which WorkOS environment to sign in to is answered by the control plane itself, so there is nothing
to configure. The browser need not be on this machine, so this works over SSH.


"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt login', description='sign in to Pixeltable Cloud', epilog=EPILOG)
    parser.add_argument('--no-browser', action='store_true', help='Print the link instead of opening a browser')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)
    try:
        _login(args)
    except AuthError as e:
        # A traceback would bury the one line saying what the sign-in provider refused.
        print(f'pxt login: error: {e}', file=sys.stderr)
        sys.exit(1)


def run_logout(argv: list[str]) -> None:
    parser = Parser(prog='pxt logout', description="forget this device's cached session")
    parser.parse_args(argv)

    url = api_url()
    if credentials.clear(url):
        print(f'Signed out of {url}.')
    else:
        print('Not signed in.')


def run_whoami(argv: list[str]) -> None:
    parser = Parser(prog='pxt whoami', description='show the cached session')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    url = api_url()
    session = credentials.load(url)
    # What commands will actually send, which is not always the session.
    kind, where = management_client.credential_source()

    if session is None and kind != 'api_key':
        print(f'Not signed in to {url}. Run `pxt login`.', file=sys.stderr)
        sys.exit(1)

    record = {
        'api_url': url,
        'email': session.email if session else '',
        'expires_in_s': int(session.session_expires_in()) if session else None,
        'using': kind,
        'credential_source': where,
    }
    if args.json_output:
        print(json.dumps(record))
        return

    if session is not None:
        # Time until the browser is needed again. The token's own clock renews on its own.
        left = session.session_expires_in()
        state = 'expired, run `pxt login`' if left <= 0 else f'sign in again in {int(left // 60)}m'
        print(f'{session.email or "(unknown)"} on {url} — {state}')
    if kind == 'api_key':
        note = ' An API key always takes precedence over a sign-in.' if session is not None else ''
        print(f'Commands use the API key from {where}.{note}')


def _login(args: argparse.Namespace) -> None:
    url = api_url()
    session = auth.device_login(url, open_browser=not args.no_browser)

    if args.json_output:
        print(json.dumps({'api_url': url, 'email': session.email, 'organization_id': session.organization_id}))
        return
    print(f'Signed in as {session.email or "(unknown)"} on {url}.')
    if not session.organization_id:
        # Signing in proves who you are; it does not give you anywhere to work. The control plane
        # refuses a token with no organization on every operation but creating one, so say that here
        # rather than let the next command fail as an authorization error.
        dashboard = str(auth.auth_config(url).get('login_url') or 'the dashboard')
        print(f'No organization yet — create one at {dashboard} before running other commands.')
