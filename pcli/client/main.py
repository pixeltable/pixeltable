import sys

HELP = """\
usage: pcli <command> [args...]

commands:
  health     show daemon info
  ls         list catalog entries
  describe   show a table's schema and metadata
  errors     list rows where a computed column failed
  history    show a table's version timeline
  columns    list columns across tables (optionally one)
  idxs       list indexes across tables (optionally one)
  rows       peek the first N rows of a table
  get        look up a single row by primary key
  count      count rows in a table
  status     show daemon/runtime state
  env        show pixeltable env vars and active config file
  computed   list computed columns (alias for 'columns --computed')
  drop       drop a table or view (use 'rm' for directories)
  rm         remove a directory (use 'drop' for tables/views)
  rename     rename a table/view/dir in place
  mv         move a table/view/dir to a different directory
  revert     undo the last op(s) on a table

Use 'pcli <command> --help' for subcommand options.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        sys.stdout.write(HELP)
        sys.exit(0 if len(sys.argv) >= 2 else 2)

    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == 'health':
        from .commands import health

        health.run(argv)
    elif cmd == 'ls':
        from .commands import ls

        ls.run(argv)
    elif cmd == 'describe':
        from .commands import describe

        describe.run(argv)
    elif cmd == 'errors':
        from .commands import errors

        errors.run(argv)
    elif cmd == 'history':
        from .commands import history

        history.run(argv)
    elif cmd == 'columns':
        from .commands import columns

        columns.run(argv)
    elif cmd == 'idxs':
        from .commands import idxs

        idxs.run(argv)
    elif cmd == 'rows':
        from .commands import rows

        rows.run(argv)
    elif cmd == 'get':
        from .commands import get

        get.run(argv)
    elif cmd == 'count':
        from .commands import count

        count.run(argv)
    elif cmd == 'status':
        from .commands import status

        status.run(argv)
    elif cmd == 'env':
        from .commands import env

        env.run(argv)
    elif cmd == 'computed':
        from .commands import computed

        computed.run(argv)
    elif cmd == 'drop':
        from .commands import drop

        drop.run(argv)
    elif cmd == 'rm':
        from .commands import rm

        rm.run(argv)
    elif cmd == 'rename':
        from .commands import rename

        rename.run(argv)
    elif cmd == 'mv':
        from .commands import mv

        mv.run(argv)
    elif cmd == 'revert':
        from .commands import revert

        revert.run(argv)
    else:
        print(f'pcli: unknown command: {cmd}', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
