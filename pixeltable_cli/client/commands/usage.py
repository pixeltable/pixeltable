"""`pxt usage` - this month's usage of your organization in Pixeltable Cloud, against what its plan includes."""

from __future__ import annotations

import datetime
import json
import math
from typing import Any

from ..parser import Parser
from ..utils import get_request, plural, print_aligned

EPILOG = """\
Examples:
  pxt usage
  pxt usage --json

The organization is the one your credential belongs to. Costs are the month so far at Pro's rates,
after what the plan includes. Compute lags up to an hour; storage and egress, up to a day. On Pro,
the amount past included is an estimate: the invoice is Stripe's, and a first month's credit is
prorated. --json prints the control plane's answer as it is.
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt usage', description="show this month's usage of your organization", epilog=EPILOG)
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)
    usage = get_request('/api/usage')
    if args.json_output:
        print(json.dumps(usage, indent=2))
        return
    print_usage(usage, datetime.datetime.now(datetime.timezone.utc))


def print_usage(usage: dict[str, Any], now: datetime.datetime) -> None:
    """Print a get_usage answer: the period, the money against the plan, the meters, then the resources."""
    start = datetime.datetime.fromisoformat(usage['period_start'])
    end = datetime.datetime.fromisoformat(usage['period_end'])
    days_left = max(0, math.ceil((end - max(now, start)) / datetime.timedelta(days=1)))
    last_day = end - datetime.timedelta(days=1)
    print(
        f'{str(usage["plan"]).title()} plan, {start:%Y-%m-%d} to {last_day:%Y-%m-%d} (UTC), '
        f'{plural(days_left, "day")} left'
    )
    counts_from = start if usage.get('counts_from') is None else datetime.datetime.fromisoformat(usage['counts_from'])
    if counts_from > start:
        print(f'Usage counts from {counts_from:%Y-%m-%d}, the billing start.')
    print(_spend(usage))

    meters = usage.get('meters') or []
    if len(meters) > 0:
        print()
        rows = [[m['label'], f'{_number(m["quantity"])} {m["unit"]}', _usd(m.get('usd'), cents=True)] for m in meters]
        print_aligned(['METER', 'QUANTITY', 'COST'], rows, right_align={2})

    resources = usage.get('resources') or []
    if len(resources) > 0:
        print()
        rows = [
            [r['label'], f'{_amount(r.get("used"), r["unit"])} / {_amount(r.get("limit"), r["unit"])}']
            for r in resources
        ]
        print_aligned(['RESOURCE', 'USED / LIMIT'], rows, right_align=set())


def _spend(usage: dict[str, Any]) -> str:
    """The month's cost against what the plan includes, and what happens past it."""
    used, included = usage.get('used_usd'), usage.get('included_usd')
    if used is None:
        return 'Usage is billed under your contract.'
    if included is None:
        return f'{_usd(used)} used'
    spent = f'{_usd(used)} of {_usd(included)} used'
    if usage.get('stops_at_limit'):
        return f'{spent}; past {_usd(included)} your databases and services stop until the 1st (UTC)'
    return f'{spent}; past included (estimate): {_usd(usage.get("past_included_usd"))}'


def _number(value: float) -> str:
    """At most two decimals, without trailing zeros: 120.5, 241, 0."""
    return f'{value:,.2f}'.rstrip('0').rstrip('.')


def _usd(value: float | None, cents: bool = False) -> str:
    """Dollars, with cents unless they are zero and cents is False: $4.20, $25, $0."""
    if value is None:
        return '-'
    text = f'${value:,.2f}'
    return text if cents else text.removesuffix('.00')


def _amount(value: float | None, unit: str) -> str:
    """A resource's usage or limit: a count as it is, bytes as decimal GB."""
    if value is None:
        return '-'
    if unit == 'count':
        return str(int(value))
    if unit == 'bytes':
        return f'{_number(value / 10**9)} GB'
    return f'{_number(value)} {unit}'
