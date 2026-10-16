import collections
import json
import shlex
import sys

from pixeltable_cli import models

from ..parser import Parser
from ..utils import get_request, plural, print_next, validate_path_arg

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
        for e in resp.entries:
            pk = '{' + ', '.join(f'{k}: {v!r}' for k, v in e.pk.items()) + '}'
            errmsg = e.errormsg if e.errormsg is not None else ''
            print(f'{pk}\t{e.column}\t{e.errortype}\t{errmsg}')
        if not resp.entries:
            where = path if args.col is None else f'column {args.col} of {path}'
            print(f'no failed cells in {where}', file=sys.stderr)
    sys.stdout.flush()
    for column, n in collections.Counter(e.column for e in resp.entries).most_common():
        command = shlex.join(['pxt', 'recompute', path, column, '--errors-only', '-f'])
        print_next(command, f'reruns {plural(n, "failed cell")} of {column} and the columns that depend on it')
