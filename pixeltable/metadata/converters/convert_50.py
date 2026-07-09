"""Replica support has been removed.

Replicas hold partial data with replica-specific version bookkeeping (latest-provable v_max, possibly
non-contiguous versions) and cannot be faithfully converted into regular tables. Rather than silently dropping
them, refuse to migrate a database that still contains replicas. For replica-free databases, drop the now-obsolete
is_replica (TableMd) and is_fragment (VersionMd) fields and remove the system directory used to hide anonymous
replica base tables.
"""

from uuid import UUID

import sqlalchemy as sql

from pixeltable import exceptions as excs
from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Dir, Table, TableVersion


@register_converter(version=50)
def _(conn: sql.Connection) -> None:
    # Check before making any changes, so that a database with replicas is left untouched at its current version.
    replica_paths = __user_replica_paths(conn)
    if len(replica_paths) > 0:
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, __replica_error_message(replica_paths))

    # No replicas: drop the now-obsolete metadata keys.
    conn.execute(sql.update(Table).values(md=Table.md.op('-')('is_replica')))
    conn.execute(sql.update(TableVersion).values(md=TableVersion.md.op('-')('is_fragment')))

    # Remove the system directory used to hide anonymous replica base tables. With no replicas left it is empty;
    # the dirs.id foreign key on tables restricts deletion, so this aborts rather than orphaning any stray table.
    conn.execute(sql.delete(Dir).where(Dir.md['name'].astext == '_system'))


def __user_replica_paths(conn: sql.Connection) -> list[str]:
    """Return the user-visible paths of all replica tables (those not hidden under the _system directory)."""
    # Build a map of dir id -> (name, parent_id) so we can resolve full paths.
    dirs: dict[UUID, tuple[str, UUID | None]] = {
        row.id: (row.md['name'], row.parent_id) for row in conn.execute(sql.select(Dir.id, Dir.parent_id, Dir.md))
    }

    def dir_components(dir_id: UUID) -> list[str]:
        components: list[str] = []
        cur: UUID | None = dir_id
        while cur is not None:
            name, parent = dirs[cur]
            if name == '':  # root directory
                break
            components.insert(0, name)
            cur = parent
        return components

    paths: list[str] = []
    q = sql.select(Table.dir_id, Table.md).where(Table.md['is_replica'].astext == 'true', Table.dir_id.is_not(None))
    for row in conn.execute(q):
        components = dir_components(row.dir_id)
        # Anonymous base tables live under the _system directory; the user drops/converts the visible replica instead.
        if len(components) > 0 and components[0] == '_system':
            continue
        paths.append('/'.join([*components, row.md['name']]))
    return sorted(paths)


def __replica_error_message(replica_paths: list[str]) -> str:
    path_list = '\n'.join(f'    - {p}' for p in replica_paths)
    example = replica_paths[0]
    return (
        f'This Pixeltable database contains {len(replica_paths)} replica table(s), which are no longer supported:\n'
        f'{path_list}\n'
        'Replicas cannot be migrated automatically. Before upgrading, use your previous Pixeltable version to either '
        'materialize a local copy of each replica:\n'
        f"    pxt.create_table('my_local_table', source=pxt.get_table({example!r}).select())\n"
        'or drop each replica:\n'
        f'    pxt.drop_table({example!r})\n'
        'and then re-run the upgrade.'
    )
