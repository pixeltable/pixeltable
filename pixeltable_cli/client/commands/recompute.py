import json

from pixeltable_cli import models

from ..parser import Parser
from ..utils import confirm_or_exit, get_request, plural, post_request, validate_path_arg

EPILOG = """\
Examples:
  pxt recompute my_dir/my_table summary -f                  # recompute one column and its dependents
  pxt recompute my_dir/my_table summary embedding -f        # several columns in one pass
  pxt recompute my_dir/my_table summary --no-cascade -f     # leave the columns that depend on it alone
  pxt recompute my_dir/my_table summary --errors-only -f    # only the rows whose value is an error
  pxt recompute my_dir/my_table summary -n                  # dry-run: what it would recompute, and over how many rows

Notes:
  Recomputing evaluates the column over every row, which for a udf that calls a model or an API costs
  real time and money. --errors-only narrows it to the rows that failed, and takes a single column.
  Without -f, confirmation is read from the terminal; non-interactive callers must pass -f."""


def run(argv: list[str]) -> None:
    ap = Parser(prog='pxt recompute', epilog=EPILOG)
    ap.add_argument('path')
    ap.add_argument('columns', nargs='+', metavar='COLUMN', help='the computed column(s) to recompute')
    ap.add_argument('--errors-only', action='store_true', dest='errors_only', help='only rows whose value is an error')
    ap.add_argument(
        '--no-cascade',
        action='store_false',
        dest='cascade',
        help='leave the computed columns that depend on these alone',
    )
    ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
    ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    ap.add_argument('--json', action='store_true', dest='as_json')
    args = ap.parse_args(argv)

    if args.errors_only and len(args.columns) > 1:
        ap.error('--errors-only takes a single column')

    path = validate_path_arg(args.path)
    columns = ', '.join(args.columns)
    if args.dry_run:
        # TODO: compute the total number of affected rows across all transitive views as well, possibly even the
        # exact columns that would be recomputed
        table_rows = models.CountResponse.model_validate(get_request('/api/tables/count', params={'path': path})).count
        if args.as_json:
            plan = {
                'path': path,
                'columns': args.columns,
                'errors_only': args.errors_only,
                'cascade': args.cascade,
                'table_rows': table_rows,
            }
            print(json.dumps(plan, indent=2))
            return
        scope = 'the rows with errors' if args.errors_only else f'{plural(table_rows, "row")} of {path}'
        dependents = ', plus the dependent columns in this table and all of its views' if args.cascade else ''
        print(f'would recompute {columns} on {path} over {scope}{dependents}')
        return

    cascade = ', including all dependents' if args.cascade else ', without their dependents'
    confirm_or_exit(f'recompute {columns} on {path}{cascade}?', args.force)

    resp = models.RecomputeResponse.model_validate(
        post_request(
            '/api/tables/recompute',
            {'path': path, 'columns': args.columns, 'errors_only': args.errors_only, 'cascade': args.cascade},
        )
    )
    if args.as_json:
        print(resp.model_dump_json(indent=2))
        return
    print(
        f'recomputed {", ".join(resp.columns)} on {resp.path}: '
        f'{plural(resp.num_rows, "row")}, {plural(resp.num_computed_values, "computed value")}'
    )
    if resp.num_excs > 0:
        print(f'{plural(resp.num_excs, "error")} in {", ".join(resp.cols_with_excs)}')
