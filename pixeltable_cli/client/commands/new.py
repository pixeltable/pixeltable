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
  pxt whoami                    # the trial, its claim link and its expiry
  pxt logout                    # forget the trial on this machine; `pxt new` then creates another

A trial is an organization with one database (main), at most two services and 1 GiB of storage.
Unclaimed, it is deleted 72 hours after it was created. Whoever opens the claim link signs in or
signs up, and becomes the organization's admin.

The trial's API key is cached in your Pixeltable home directory for the control plane that commands
reach (PIXELTABLE_API_URL), and later commands use it. While the trial lasts, running `pxt new` again
prints the same trial and creates nothing. An API key or a `pxt login` session outranks a trial, so
with either one configured, `pxt new` exits 1 and creates nothing.

PIXELTABLE_SITE_URL sets the Pixeltable site that `pxt new` asks for a trial; the default is
https://pixeltable.com."""


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt new', description='get a free Pixeltable Cloud database, no account needed', epilog=EPILOG
    )
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    answer = post_request('/api/trial', {})
    trial = answer['trial']
    # stderr, so that --json leaves one document on stdout
    if answer['warning'] != '':
        print(f'pxt new: warning: {answer["warning"]}', file=sys.stderr)
    if args.json_output:
        fields = ('org', 'org_id', 'db', 'claim_url', 'expires_at')
        print(json.dumps({'created': answer['created'], 'api_url': answer['api_url'], **{f: trial[f] for f in fields}}))
        return

    uri = f'pxt://{trial["org"]}:{trial["db"]}'
    what = f'organization {trial["org"]}, with database {trial["db"]}'
    if answer['created']:
        print(f'Created a free Pixeltable Cloud trial: {what}.')
    else:
        print(f'This machine already has a Pixeltable Cloud trial: {what}.')
    print(f'\n  {uri}\n')
    print(trial_fate(trial))
    print("Whoever opens the claim link becomes the organization's admin.")
    print(f"Later pxt commands on this machine send the trial's API key to {answer['api_url']}.")
    print('\nPoint a project at it in pixeltable.toml (`pxt init` writes the file):\n')
    print(f"  [[pixeltable.database]]\n  name = '{uri}'\n")
    print('Then run:\n')
    print(f'  pxt db update {uri}')
    print(f'  pxt schema update app.py {uri}')
    print(f'  pxt service update app.py {uri}')
    if not answer['created']:
        print('\nTo start over with a new trial, run `pxt logout`, then `pxt new`.')


def trial_fate(trial: dict[str, Any]) -> str:
    """What happens to the trial unless it is claimed."""
    if trial['expired']:
        return f'It expired at {trial["expires_at"]}: unless it was claimed, it has been deleted.'
    return f'Claim it at {trial["claim_url"]}, or it is deleted at {trial["expires_at"]}.'
