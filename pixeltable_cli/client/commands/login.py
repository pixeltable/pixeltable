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
import time
import webbrowser
from typing import Optional

from pixeltable.service import auth, credentials, management_client
from pixeltable.service.auth import AuthError
from pixeltable.service.management_client import api_url
from pixeltable.service.management_protocol import ListOrgsRequest, ListOrgsResponse

from ..parser import Parser

# Sign-up runs at human speed -- an email to verify, a name to choose -- so the wait is generous.
_SETUP_TIMEOUT_S = 900.0
_SETUP_POLL_S = 3.0

EPILOG = """\
Examples:
  pxt login                     # sign in, or create an account, in a browser
  pxt whoami                    # who this machine is signed in as, and for how long
  pxt logout                    # forget this environment's session
  pxt logout --all              # forget every environment's

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
    parser = Parser(prog='pxt logout', description='forget a cached session')
    parser.add_argument('--all', action='store_true', help='Forget every environment, not just this one')
    args = parser.parse_args(argv)

    target = None if args.all else api_url()
    if credentials.clear(target):
        print('Signed out.' if args.all else f'Signed out of {target}.')
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
    if not session.organization_id:
        session = _wait_for_organization(url, open_browser=not args.no_browser)

    if args.json_output:
        print(json.dumps({'api_url': url, 'email': session.email, 'organization_id': session.organization_id}))
        return
    print(f'Signed in as {session.email or "(unknown)"} on {url}.')


def _first_ready_org(url: str) -> Optional[str]:
    """The caller's organization once it is usable, or None while it is still being created.

    default_db is the readiness signal: an organization exists here only after its first database
    does, and a token scoped to one without it would fail on the next command.
    """
    response = ListOrgsResponse(**management_client.api_call(ListOrgsRequest()))
    ready = [org for org in response.orgs if org.default_db]
    return ready[0].org_id if ready else None


def _wait_for_organization(url: str, open_browser: bool) -> credentials.Session:
    """Wait out the rest of a sign-up, then scope the session to what it created.

    Approving the device code only proves who you are. A first-time account has no organization
    until the dashboard has made one, and `pxt login` promises a session that works, so it waits
    rather than handing back one that does not.
    """
    dashboard = str(auth.auth_config(url).get('login_url') or '')
    print(f'Finishing setup at {dashboard}')
    if open_browser and dashboard:
        webbrowser.open(dashboard)

    deadline = time.time() + _SETUP_TIMEOUT_S
    while time.time() < deadline:
        org_id = _first_ready_org(url)
        if org_id is not None:
            return auth.authorize_org(url, org_id)
        time.sleep(_SETUP_POLL_S)
    raise AuthError('timed out waiting for your organization to be created')
