import collections
import json
import shlex
import sys

from pixeltable_cli import models

from ..parser import Parser
from ..utils import display_path, get_request, plural, print_next, validate_path_arg

EPILOG = """\
Examples:
  pxt errors my_dir/my_table
  pxt errors my_dir/my_table --col embedding
  pxt errors my_dir/my_table --json

Note: the table must have a primary key (defined via primary_key=)."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt errors', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('--col', help='filter to a single computed column')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    path = validate_path_arg(args.path)
    resp = models.ErrorsResponse.model_validate(
        get_request('/api/tables/errors', params={'path': path, 'col': args.col})
    )

    if args.as_json:
        print(json.dumps([e.model_dump() for e in resp.entries], indent=2, default=str))
    else:
        _print_entries(resp.entries, path if args.col is None else f'column {args.col} of {path}')
    sys.stdout.flush()
    _print_recomputes(resp.entries, path)


def _print_entries(entries: list[models.ErrorEntry], where: str) -> None:
    for e in entries:
        pk = '{' + ', '.join(f'{k}: {v!r}' for k, v in e.pk.items()) + '}'
        errmsg = e.errormsg if e.errormsg is not None else ''
        print(f'{pk}\t{e.column}\t{e.errortype}\t{errmsg}')
    if len(entries) == 0:
        print(f'no failed cells in {where}', file=sys.stderr)


def _print_recomputes(entries: list[models.ErrorEntry], path: str) -> None:
    """Names, per failed column, the recompute that reruns its failed cells, on the table that defines the column."""
    failed = collections.Counter((e.defining_base, e.column) for e in entries if e.recomputable)
    for (base, column), n in failed.most_common():
        if base is None:
            command = shlex.join(['pxt', 'recompute', path, column, '--errors-only', '-f'])
            print_next(command, f'reruns {plural(n, "failed cell")} of {column} and the columns that depend on it')
        else:
            # the daemon resolved the base's path, so it is shown absolute, which a working directory leaves alone
            table = display_path(base)
            command = shlex.join(['pxt', 'recompute', table, column, '--errors-only', '-f'])
            print_next(
                command, f'reruns every failed cell of {column} in {table}, which defines it; {n} are listed here'
            )
