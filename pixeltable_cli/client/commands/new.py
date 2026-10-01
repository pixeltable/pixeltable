"""`pxt new` - a free Pixeltable Cloud database, with no account.

The daemon registers an anonymous agent with the site's sign-in service, creates the trial organization
with it, and caches the trial's API key where `pxt login` caches a session, so every later command uses
it. This module prints the result, never the key.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ..parser import Parser
from ..utils import post_request

EPILOG = """\
Examples:
  pxt new                       # a trial organization with one database, and a link to claim it
  pxt new --json                # the same, machine-readable
  pxt whoami                    # the trial and its expiry
  pxt logout                    # remove the trial's key from this machine; `pxt new` then creates another

A trial is an organization with one database (main), at most two services and 50 GB of media storage.
The server sets its expiry, normally 48 hours after creation. Unless claimed, it is deleted at that
time. Whoever opens the claim link signs in or signs up, and becomes the organization's admin.
`pxt new` prints the link only when it creates the trial, and `pxt logout` prints it once more.

The trial's API key is cached in your Pixeltable home directory for the control plane that commands
reach (PIXELTABLE_API_URL), and later commands use it. While the trial lasts, running `pxt new` again
prints the same trial and creates nothing. An API key or a `pxt login` session outranks a trial, so
with either one configured, `pxt new` exits 1 and creates nothing. `pxt logout` does not revoke the
key: it works until the organization is claimed or expires.

PIXELTABLE_SITE_URL sets the Pixeltable site that `pxt new` asks for a trial; the default is
https://www.pixeltable.com. It must use https, except on localhost."""


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt new', description='get a free Pixeltable Cloud database, no account needed', epilog=EPILOG
    )
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    answer = post_request('/api/trial', {})
    trial = answer['trial']
    # stderr, so that --json leaves one document on stdout
    for warning in answer['warnings']:
        print(f'pxt new: warning: {warning}', file=sys.stderr)
    if args.json_output:
        fields = ('org', 'org_id', 'db', 'expires_at')
        print(
            json.dumps(
                {
                    'created': answer['created'],
                    'api_url': answer['api_url'],
                    **{f: trial[f] for f in fields},
                    'claim_url': answer['claim_url'],  # null unless this run created the trial
                    'warnings': answer['warnings'],
                }
            )
        )
        return

    uri = f'pxt://{trial["org"]}:{trial["db"]}'
    what = f'organization {trial["org"]}, with database {trial["db"]}'
    if answer['created']:
        print(f'Created a free Pixeltable Cloud trial: {what}.')
    else:
        print(f'This machine already has a Pixeltable Cloud trial: {what}.')
    print(f'\n  {uri}\n')
    print(trial_fate(trial, answer['claim_url']))
    print(f"Later pxt commands on this machine send the trial's API key to {answer['api_url']}.")
    print('\nPoint a project at it in pixeltable.toml (`pxt init` writes the file):\n')
    print(f"  [[pixeltable.database]]\n  name = '{uri}'\n")
    print('Then run:\n')
    print(f'  pxt db update {uri}')
    print(f'  pxt schema update app.py {uri}')
    print(f'  pxt service update app.py {uri}')
    if not answer['created']:
        print('\nTo start over with a new trial, run `pxt logout`, then `pxt new`.')


def trial_fate(trial: dict[str, Any], claim_url: str | None = None) -> str:
    """What happens to the trial unless it is claimed, and the claim link, printed only where it is given."""
    if trial['expired']:
        return f'It expired at {trial["expires_at"]}: unless it was claimed, it has been deleted.'
    if claim_url is None:
        return f'Unless it is claimed, it is deleted at {trial["expires_at"]}.'
    return (
        f'Claim it at {claim_url}, or it is deleted at {trial["expires_at"]}.\n'
        'Anyone with this link can claim the organization.'
    )
