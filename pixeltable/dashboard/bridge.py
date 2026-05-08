"""
Bridge layer between Pixeltable internal APIs and the Dashboard REST API.

This module translates Pixeltable's internal data structures into JSON-serializable
formats suitable for the dashboard frontend.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable.catalog.table_metadata import TableMetadata
from pixeltable.config import Config
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')

if TYPE_CHECKING:
    from pixeltable import exprs


def _build_select(
    tbl: pxt.Table, *, include_errors: bool = False
) -> tuple[list[dict[str, Any]], dict[str, exprs.Expr], dict[str, str], dict[str, tuple[str, str]]]:
    """Build column info list, select dict, media URL map, and error column map.

    Unstored columns appear in the returned columns list with is_stored=False, but are
    excluded from select_dict and error_cols.

    Returns (columns, select_dict, media_url_cols, error_cols).
    """
    md = tbl.get_metadata()
    columns: list[dict[str, Any]] = []
    select_dict: dict[str, exprs.Expr] = {}
    media_url_cols: dict[str, str] = {}
    error_cols: dict[str, tuple[str, str]] = {}

    # Columns backed by a B-tree index can be ordered cheaply; the rest cannot.
    sorted_cols: set[str] = {
        c for idx in md['indices'].values() if idx['index_type'] == 'btree' for c in idx['columns']
    }

    for col_name, info in md['columns'].items():
        is_media = info['media_validation'] is not None
        is_computed = info['is_computed']
        is_stored = info['is_stored']
        columns.append(
            {
                'name': col_name,
                'type': info['type_'],
                'is_media': is_media,
                'is_computed': is_computed,
                'is_stored': is_stored,
                'is_sorted': col_name in sorted_cols,
            }
        )

        if not is_stored:
            continue

        col_ref = getattr(tbl, col_name)
        if is_media:
            # only fetch the URL
            url_key = f'{col_name}__url'
            select_dict[url_key] = col_ref.fileurl
            media_url_cols[col_name] = url_key
        else:
            select_dict[col_name] = col_ref

        if include_errors and (is_computed or is_media):
            error_type_key = f'{col_name}__errortype'
            error_msg_key = f'{col_name}__errormsg'
            select_dict[error_type_key] = col_ref.errortype
            select_dict[error_msg_key] = col_ref.errormsg
            error_cols[col_name] = (error_type_key, error_msg_key)

    return columns, select_dict, media_url_cols, error_cols


def _resolve_fileurl(fileurl: str, http_address: str) -> str:
    """Convert a file:// URL to an HTTP URL, or return external URLs as-is."""
    if fileurl.startswith('file:'):
        parsed = urllib.parse.urlparse(fileurl)
        local_path = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        return f'{http_address}{local_path}'
    return fileurl


def get_table_data(
    table_path: str,
    offset: int = 0,
    limit: int = 50,
    order_by: str | None = None,
    order_desc: bool = False,
    errors_only: bool = False,
) -> dict[str, Any]:
    """
    Get paginated data from a table with media URLs resolved:
    - ignores order_by if it is a column without a B-tree index
    - doesn't return data for unstored computed columns
    """
    tbl = pxt.get_table(table_path)
    http_address = Env.get().http_address
    columns, select_dict, media_url_cols, error_cols = _build_select(tbl, include_errors=True)
    query = tbl.select(**select_dict)

    error_predicate: exprs.Expr | None = None
    if errors_only:
        error_predicates = []
        for col_name in error_cols:
            try:
                col_ref = getattr(tbl, col_name)
                error_predicates.append(col_ref.errortype != None)
            except Exception:
                pass
        if len(error_predicates) == 0:
            # 'errors only' was requested but the table has no columns that can carry errors.
            # Short-circuit: nothing to return.
            return {'columns': columns, 'rows': [], 'total_count': 0, 'offset': offset, 'limit': limit}
        error_predicate = error_predicates[0]
        for pred in error_predicates[1:]:
            error_predicate |= pred
        query = query.where(error_predicate)

    if order_by is not None:
        # only sort by columns with a B-tree index; other columns would force a full sort
        order_col = next((c for c in columns if c['name'] == order_by), None)
        if order_col is not None and order_col['is_sorted']:
            col = getattr(tbl, order_by)
            query = query.order_by(col, asc=not order_desc)

    if error_predicate is not None:
        total_count = tbl.where(error_predicate).count()
    else:
        total_count = tbl.count()
    results = list(query.limit(limit, offset=offset if offset != 0 else None).collect())

    rows: list[dict[str, Any]] = []
    for row in results:
        row_data: dict[str, Any] = {}
        cell_errors: dict[str, dict[str, str]] = {}
        for col_info in columns:
            col_name = col_info['name']
            if not col_info['is_stored']:
                continue  # omitted
            value = row.get(col_name)

            if col_info['is_media']:
                fileurl = row.get(media_url_cols.get(col_name))
                row_data[col_name] = _resolve_fileurl(fileurl, http_address) if fileurl is not None else None
            elif value is None or isinstance(value, (int, float, bool, str, list, dict)):
                row_data[col_name] = value
            elif isinstance(value, (datetime.datetime, datetime.date)):
                row_data[col_name] = value.isoformat()
            else:
                row_data[col_name] = str(value)

            if col_name in error_cols:
                error_type_key, error_msg_key = error_cols[col_name]
                error_type = row.get(error_type_key)
                error_msg = row.get(error_msg_key)
                if error_type is not None:
                    cell_errors[col_name] = {
                        'error_type': str(error_type),
                        'error_msg': str(error_msg) if error_msg is not None else '',
                    }

        if len(cell_errors) > 0:
            row_data['_errors'] = cell_errors
        rows.append(row_data)

    return {'columns': columns, 'rows': rows, 'total_count': total_count, 'offset': offset, 'limit': limit}


def export_table_csv(table_path: str, limit: int = 100_000) -> bytes:
    """Export a table as CSV bytes. Media columns export their file URL."""
    tbl = pxt.get_table(table_path)
    http_address = Env.get().http_address
    columns, select_dict, media_url_cols, _ = _build_select(tbl)
    # Unstored columns have no value to export; their cells would be empty anyway.
    col_names = [c['name'] for c in columns if c['is_stored']]

    results = list(tbl.select(**select_dict).limit(limit).collect())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(col_names)

    for row in results:
        csv_row: list[str] = []
        for col_name in col_names:
            if col_name in media_url_cols:
                fileurl = row.get(media_url_cols[col_name])
                csv_row.append(_resolve_fileurl(fileurl, http_address) if fileurl is not None else '')
            else:
                val = row.get(col_name)
                if val is None:
                    csv_row.append('')
                elif isinstance(val, (dict, list)):
                    csv_row.append(json.dumps(val, default=str))
                else:
                    csv_row.append(str(val))
        writer.writerow(csv_row)

    return buf.getvalue().encode('utf-8')


def search(query: str, limit: int = 50) -> dict[str, Any]:
    """
    Search across directories, tables, and columns.
    """
    query_lower = query.lower()

    results: dict[str, Any] = {'query': query, 'directories': [], 'tables': [], 'columns': []}

    # Search directories
    all_dirs = pxt.list_dirs('', recursive=True)
    for dir_path in all_dirs:
        if query_lower in dir_path.lower():
            results['directories'].append({'path': dir_path, 'name': dir_path.split('/')[-1]})
            if len(results['directories']) >= limit:
                break

    # Search tables and their columns (single get_table call per table)
    all_tables = pxt.list_tables('', recursive=True)
    for tbl_path in all_tables:
        tbl_name = tbl_path.split('/')[-1]
        table_matches = query_lower in tbl_path.lower()

        # Only fetch table metadata once, and only when needed
        tbl_md: TableMetadata | None = None
        if table_matches or len(results['columns']) < limit:
            try:
                tbl = pxt.get_table(tbl_path)
                tbl_md = tbl.get_metadata()
            except Exception:
                # If we can't get metadata, record table match with defaults
                if table_matches and len(results['tables']) < limit:
                    results['tables'].append({'path': tbl_path, 'name': tbl_name, 'kind': 'table'})
                continue

        if table_matches and len(results['tables']) < limit and tbl_md:
            results['tables'].append({'path': tbl_path, 'name': tbl_name, 'kind': tbl_md['kind']})

        # Search columns within this table (reuse tbl_md)
        if tbl_md and len(results['columns']) < limit:
            for col_name, col_info in tbl_md['columns'].items():
                if query_lower in col_name.lower():
                    results['columns'].append(
                        {
                            'name': col_name,
                            'table': tbl_path,
                            'type': col_info['type_'],
                            'is_computed': col_info['is_computed'],
                        }
                    )
                    if len(results['columns']) >= limit:
                        break

    return results


# matches the name of the function of the first function call in a display expression
_FIRST_FUNC_RE = re.compile(r'(\w+)\(')


def _collect_tbl_nodes(nodes: list[pxt.TreeNode], out: list[pxt.TableNode]) -> None:
    """Collect all transitively reachable TableNodes in 'nodes' and return them in 'out'"""
    for n in nodes:
        if n['kind'] == 'directory':
            _collect_tbl_nodes(n['entries'], out)
        else:
            out.append(n)


def _split_tbl_path(tbl_path: str) -> tuple[str, int | None]:
    """Split a Pixeltable path of the form 'p' or 'p:N' into (path, version)."""
    head, sep, tail = tbl_path.rpartition(':')
    if sep != '' and tail.isdigit():
        return head, int(tail)
    return tbl_path, None


def _collect_pipeline_paths(table_nodes: list[pxt.TableNode], tbl_path: str) -> set[str] | None:
    """Return the version-free paths of all tables/views transitively connected to tbl_path."""
    by_path = {n['path']: n for n in table_nodes}
    if tbl_path not in by_path:
        return None
    view_map: dict[str, list[str]] = {}  # unpinned base path -> list[view path]
    for n in table_nodes:
        if n['base'] is not None:
            # make sure we record the base path w/o the version suffix
            base, _ = _split_tbl_path(n['base'])
            view_map.setdefault(base, []).append(n['path'])

    connected: set[str] = {tbl_path}
    # ancestors
    current = tbl_path
    while True:
        base = by_path[current]['base']
        if base is None:
            break
        # make sure we record the base path w/o the version suffix
        current, _ = _split_tbl_path(base)
        connected.add(current)

    # descendants
    stack = [tbl_path]
    while stack:
        p = stack.pop()
        for view_path in view_map.get(p, []):
            if view_path not in connected:
                connected.add(view_path)
                stack.append(view_path)
    return connected


def get_pipeline(tbl_path: str | None = None) -> dict[str, Any]:
    """Return DAG metadata for the Pipeline Inspector.

    If tbl_path is None, returns the full catalog. If tbl_path is given, returns only the
    connected component containing that table (transitive ancestors + the table + transitive
    descendants). Returns an empty result if tbl_path is not in the catalog.
    """
    tbl_nodes: list[pxt.TableNode] = []
    _collect_tbl_nodes(pxt.get_dir_tree(), tbl_nodes)

    pipeline_paths: set[str] | None
    if tbl_path is None:
        pipeline_paths = {n['path'] for n in tbl_nodes}
    else:
        pipeline_paths = _collect_pipeline_paths(tbl_nodes, tbl_path)
        if pipeline_paths is None:
            return {'nodes': [], 'edges': []}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for path in sorted(pipeline_paths):
        try:
            tbl = pxt.get_table(path)
            md = tbl.get_metadata()
            column_md = md['columns']
            row_count = tbl.count()

            iterator_name: str | None = None
            if md['is_view'] and md['iterator_call'] is not None:
                m = _FIRST_FUNC_RE.search(md['iterator_call'])
                iterator_name = m.group(1) if m is not None else md['iterator_call']

            columns: list[dict[str, Any]] = []
            computed_cols: list[str] = []

            for col_name, info in column_md.items():
                value_expr = info['computed_with']
                is_iter_col = info['is_iterator_col']

                # Iterator-produced columns: use the iterator name as computed_with
                if is_iter_col and value_expr is None:
                    value_expr = iterator_name

                is_computed = value_expr is not None
                if is_computed:
                    computed_cols.append(col_name)
                defined_in = info['defined_in']

                value_expr = value_expr[:200] if value_expr is not None else None
                func_type: str | None
                if not is_computed and not is_iter_col:
                    func_type = None
                elif is_iter_col:
                    func_type = 'iterator'
                elif info['is_builtin']:
                    func_type = 'builtin'
                else:
                    func_type = 'custom_udf'

                func_name: str | None = None  # the function name of the topmost call
                if is_iter_col:
                    func_name = iterator_name
                elif value_expr is not None:
                    match = _FIRST_FUNC_RE.search(value_expr)
                    if match is not None:
                        func_name = match.group(1)

                col_entry: dict[str, Any] = {
                    'name': col_name,
                    'type': info['type_'],
                    'is_computed': is_computed,
                    'is_iterator_col': is_iter_col,
                    'computed_with': value_expr,
                    'defined_in': defined_in,
                    'defined_in_self': defined_in == md['name'],
                    'func_name': func_name,
                    'func_type': func_type,
                }

                if is_computed and value_expr is not None and not is_iter_col:
                    col_entry['depends_on'] = [d[1] for d in info['depends_on']]

                columns.append(col_entry)

            # indices: surface only embedding indices here, the dashboard doesn't want to know about B-trees
            indices: list[dict[str, Any]] = []
            for idx_name, idx_info in md['indices'].items():
                if idx_info['index_type'] != 'embedding':
                    continue
                params = idx_info['parameters']
                assert params is not None
                indices.append(
                    {
                        'name': idx_name,
                        'columns': idx_info['columns'],
                        'type': idx_info['index_type'],
                        'embedding': str(params['embedding'])[:120],
                    }
                )

            base_path = md['base']

            is_view = md['kind'] == 'view'
            nodes.append(
                {
                    'path': path,
                    'name': md['name'],
                    'is_view': is_view,
                    'base': base_path,
                    'row_count': row_count,
                    'version': md['version'],
                    'columns': columns,
                    'indices': indices,
                    'versions': tbl.get_versions(),
                    'computed_count': len(computed_cols),
                    'insertable_count': len(columns) - len(computed_cols),
                    'iterator_type': iterator_name,
                }
            )

            if base_path is not None:
                source, base_version = _split_tbl_path(base_path)
                assert source in pipeline_paths
                edge_type = md['kind']
                edge: dict[str, Any] = {
                    'source': source,
                    'target': path,
                    'type': edge_type,
                    'label': iterator_name or edge_type,
                }
                if base_version is not None:
                    edge['base_version'] = base_version
                edges.append(edge)

        except Exception as e:
            _logger.warning(f'Pipeline: could not inspect {path}: {e}')
            nodes.append(
                {
                    'path': path,
                    'name': path.rsplit('/', 1)[-1],
                    'is_view': False,
                    'base': None,
                    'row_count': 0,
                    'version': 0,
                    'columns': [],
                    'indices': [],
                    'versions': [],
                    'computed_count': 0,
                    'insertable_count': 0,
                    'iterator_type': None,
                    'error': str(e)[:200],
                }
            )

    return {'nodes': nodes, 'edges': edges}


def get_status() -> dict[str, Any]:
    """
    Get system status including version, environment, connection info, and table count.
    """
    version = pxt.__version__

    total_tables = 0
    total_errors = 0

    def collect_totals(nodes: list[pxt.TreeNode]) -> None:
        nonlocal total_tables, total_errors
        for n in nodes:
            if n['kind'] == 'directory':
                collect_totals(n['entries'])
            else:
                total_tables += 1
                total_errors += n['error_count']

    collect_totals(pxt.get_dir_tree())

    config_info: dict[str, Any] = {}
    try:
        env = Env.get()
        cfg = Config.get()
        config_info = {
            'home': str(cfg.home),
            'db_url': env.db_url,
            'media_dir': str(env.media_dir),
            'file_cache_dir': str(env.file_cache_dir),
            'is_local': env.is_local,
        }
    except Exception:
        pass

    return {
        'version': version,
        'environment': 'local',
        'total_tables': total_tables,
        'total_errors': total_errors,
        'config': config_info,
    }
