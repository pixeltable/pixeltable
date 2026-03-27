"""
Bridge layer between Pixeltable internal APIs and the Dashboard REST API.

This module translates Pixeltable's internal data structures into JSON-serializable
formats suitable for the dashboard frontend.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import urllib.parse
import urllib.request
from typing import Any

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.catalog.table import Table
from pixeltable.catalog.table_metadata import TableMetadata
from pixeltable.config import Config
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')


def _version_error_total(tbl: Table) -> int:
    """Sum errors across all versions of a table (cheap, no row scans)."""
    try:
        return sum(v['errors'] for v in tbl.get_versions())
    except Exception:
        return 0


def _column_error_counts(tbl: Table) -> dict[str, int]:
    """Count rows with errors per computed or media column. Returns {col_name: count}."""
    counts: dict[str, int] = {}
    for col_name in tbl.columns():
        col_ref = getattr(tbl, col_name)
        if not col_ref.col.is_computed and not col_ref.col_type.is_media_type():
            continue
        try:
            counts[col_name] = tbl.where(col_ref.errortype != None).count()
        except Exception:
            counts[col_name] = 0
    return counts


def _build_select(
    tbl: Table, *, include_errors: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str], dict[str, tuple[str, str]]]:
    """Build column info list, select dict, media URL map, and error column map.

    Returns (columns, select_dict, media_url_cols, error_cols).
    """
    columns: list[dict[str, Any]] = []
    select_dict: dict[str, Any] = {}
    media_url_cols: dict[str, str] = {}
    error_cols: dict[str, tuple[str, str]] = {}

    for col_name in tbl.columns():
        col_ref = getattr(tbl, col_name)
        col_type_str = col_ref.col_type._to_str(as_schema=True)
        is_media = col_ref.col_type.is_media_type()
        is_computed = col_ref.col.is_computed
        columns.append({'name': col_name, 'type': col_type_str, 'is_media': is_media, 'is_computed': is_computed})

        if is_media:
            # Only fetch the URL — never download the actual media file
            url_key = f'{col_name}__url'
            select_dict[url_key] = col_ref.fileurl
            media_url_cols[col_name] = url_key
        else:
            select_dict[col_name] = col_ref

        if include_errors and (is_computed or is_media):
            try:
                et_key = f'{col_name}__errortype'
                em_key = f'{col_name}__errormsg'
                select_dict[et_key] = col_ref.errortype
                select_dict[em_key] = col_ref.errormsg
                error_cols[col_name] = (et_key, em_key)
            except Exception:
                pass

    return columns, select_dict, media_url_cols, error_cols


def _resolve_fileurl(fileurl: str, http_address: str) -> str:
    """Convert a file:// URL to an HTTP URL, or return external URLs as-is."""
    if fileurl.startswith('file://'):
        parsed = urllib.parse.urlparse(fileurl)
        local_path = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        return f'{http_address}{local_path}'
    return fileurl


def get_directory_tree() -> list[dict[str, Any]]:
    """
    Get the complete directory tree with all tables/views/snapshots.

    Returns:
        List of directory nodes with nested children.
    """
    all_dirs = pxt.list_dirs('', recursive=True)
    all_tables = pxt.list_tables('', recursive=True)

    root_children: list[dict[str, Any]] = []
    dir_nodes: dict[str, dict[str, Any]] = {}

    # First pass: create all directory nodes
    for dir_path in sorted(all_dirs):
        parts = dir_path.split('/')
        node = {'name': parts[-1], 'path': dir_path, 'kind': 'directory', 'children': []}
        dir_nodes[dir_path] = node

        if len(parts) == 1:
            root_children.append(node)
        else:
            parent_path = '/'.join(parts[:-1])
            if parent_path in dir_nodes:
                dir_nodes[parent_path]['children'].append(node)

    # Second pass: add tables to their directories
    for tbl_path in sorted(all_tables):
        parts = tbl_path.split('/')
        tbl_name = parts[-1]
        parent_path = '/'.join(parts[:-1]) if len(parts) > 1 else ''

        error_count = 0
        try:
            tbl = pxt.get_table(tbl_path)
            md = tbl.get_metadata()
            kind = md['kind']
            version = md['version'] if kind != 'snapshot' else None
            error_count = _version_error_total(tbl)
        except Exception as e:
            _logger.warning(f'Failed to get metadata for {tbl_path}: {e}')
            kind = 'table'
            version = None

        table_node = {'name': tbl_name, 'path': tbl_path, 'kind': kind, 'version': version, 'error_count': error_count}

        if parent_path and parent_path in dir_nodes:
            dir_nodes[parent_path]['children'].append(table_node)
        elif not parent_path:
            root_children.append(table_node)

    return root_children


def get_table_metadata(table_path: str) -> TableMetadata:
    """
    Get detailed metadata for a table including schema, indices, and lineage info.
    """
    tbl = pxt.get_table(table_path)
    return tbl.get_metadata()


def get_table_data(
    table_path: str,
    offset: int = 0,
    limit: int = 50,
    order_by: str | None = None,
    order_desc: bool = False,
    errors_only: bool = False,
) -> dict[str, Any]:
    """
    Get paginated data from a table with media URLs resolved.
    """
    tbl = pxt.get_table(table_path)
    http_address = Env.get().http_address

    columns, select_dict, media_url_cols, error_cols = _build_select(tbl, include_errors=True)

    query = tbl.select(**select_dict)

    if errors_only and error_cols:
        error_predicates = []
        for col_name in error_cols:
            try:
                col_ref = getattr(tbl, col_name)
                error_predicates.append(col_ref.errortype != None)
            except Exception:
                pass
        if error_predicates:
            combined = error_predicates[0]
            for pred in error_predicates[1:]:
                combined |= pred
            query = query.where(combined)

    if order_by and hasattr(tbl, order_by):
        col = getattr(tbl, order_by)
        query = query.order_by(col, asc=not order_desc)

    total_count = tbl.count() if not errors_only else None
    results = list(query.limit(limit, offset=offset if offset else None).collect())

    rows = []
    for row in results:
        row_data: dict[str, Any] = {}
        cell_errors: dict[str, dict[str, str]] = {}
        for col_info in columns:
            col_name = col_info['name']
            value = row.get(col_name)

            if col_info['is_media']:
                fileurl = row.get(media_url_cols.get(col_name, ''))
                row_data[col_name] = _resolve_fileurl(fileurl, http_address) if fileurl else None
            elif hasattr(value, 'isoformat'):
                row_data[col_name] = value.isoformat()
            elif isinstance(value, (list, dict)):
                row_data[col_name] = value
            elif value is not None:
                try:
                    row_data[col_name] = value if isinstance(value, (int, float, bool, str)) else str(value)
                except Exception:
                    row_data[col_name] = str(value)
            else:
                row_data[col_name] = None

            if col_name in error_cols:
                et_key, em_key = error_cols[col_name]
                etype = row.get(et_key)
                emsg = row.get(em_key)
                if etype is not None:
                    cell_errors[col_name] = {'error_type': str(etype), 'error_msg': str(emsg) if emsg else ''}

        if cell_errors:
            row_data['_errors'] = cell_errors
        rows.append(row_data)

    return {
        'columns': columns,
        'rows': rows,
        'total_count': total_count if total_count is not None else len(rows),
        'offset': offset,
        'limit': limit,
    }


def export_table_csv(table_path: str, limit: int = 100_000) -> bytes:
    """Export a table as CSV bytes. Media columns export their file URL."""
    tbl = pxt.get_table(table_path)
    http_address = Env.get().http_address

    columns, select_dict, media_url_cols, _ = _build_select(tbl)
    col_names = [c['name'] for c in columns]

    results = list(tbl.select(**select_dict).limit(limit).collect())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(col_names)

    for row in results:
        csv_row: list[str] = []
        for col_name in col_names:
            if col_name in media_url_cols:
                fileurl = row.get(media_url_cols[col_name])
                csv_row.append(_resolve_fileurl(fileurl, http_address) if fileurl else '')
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
        tbl_md = None
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


def _classify_udf(value_expr: exprs.Expr | None) -> str | None:
    """Classify the salient UDF in an expression as 'builtin' or 'custom_udf'.

    Returns None if the expression contains no UDF call.
    """
    if value_expr is None:
        return None
    fn = value_expr.salient_udf()
    if fn is None:
        return None
    path = fn.self_path
    return 'builtin' if path and path.startswith('pixeltable.') else 'custom_udf'


def _parse_deps(value_expr: exprs.Expr | None, own_name: str = '') -> list[str]:
    """Extract column names referenced in an expression."""
    if value_expr is None:
        return []
    return sorted({ref.col.name for ref in value_expr.subexprs(exprs.ColumnRef) if ref.col.name != own_name})


def _get_iterator_info(tbl: Table) -> tuple[str | None, set[str]]:
    """Return (iterator_class_name, set_of_iterator_column_names) for a view.

    Uses the fixed ``TableVersion.is_iterator_column`` (v0.5.19+) which
    correctly identifies iterator-produced columns by column id.
    """
    try:
        tv = tbl._tbl_version_path.tbl_version.get()
        if tv.iterator_call is not None:
            name = tv.iterator_call.it.name
            iter_cols = {c.name for c in tv.cols if c.name and tv.is_iterator_column(c)}
            return name, iter_cols
    except Exception:
        pass
    return None, set()


def get_pipeline() -> dict[str, Any]:
    """Return the full DAG metadata for the Pipeline Inspector."""
    table_paths = sorted(pxt.list_tables('', recursive=True))

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for path in table_paths:
        try:
            tbl = pxt.get_table(path)
            md = tbl.get_metadata()
            column_md = md['columns']
            row_count = tbl.count()

            col_errors = _column_error_counts(tbl)
            table_error_total = _version_error_total(tbl)

            is_view = md['is_view']
            iterator_name: str | None = None
            iter_col_names: set[str] = set()
            if is_view:
                iterator_name, iter_col_names = _get_iterator_info(tbl)

            columns: list[dict[str, Any]] = []
            computed_cols: list[str] = []

            for col_name, info in column_md.items():
                col_ref: exprs.ColumnRef = getattr(tbl, col_name)
                col = col_ref.col
                cw = info['computed_with']
                is_iter_col = col_name in iter_col_names

                # Iterator-produced columns: use the iterator name as computed_with
                if is_iter_col and cw is None:
                    cw = iterator_name

                is_computed = cw is not None
                if is_computed:
                    computed_cols.append(col_name)
                defined_in = info['defined_in']

                cw_str = str(cw)[:200] if cw else None
                salient_fn = col.value_expr.salient_udf() if col.value_expr is not None else None
                func_name = salient_fn.display_name if salient_fn is not None else None

                col_entry: dict[str, Any] = {
                    'name': col_name,
                    'type': info['type_'],
                    'is_computed': is_computed,
                    'is_iterator_col': is_iter_col,
                    'computed_with': cw_str,
                    'defined_in': defined_in,
                    'defined_in_self': defined_in == tbl._name,
                    'func_name': iterator_name if is_iter_col else func_name,
                    'func_type': 'iterator' if is_iter_col else _classify_udf(col.value_expr),
                    'error_count': col_errors.get(col_name, 0),
                }

                if is_computed and cw_str and not is_iter_col:
                    col_entry['depends_on'] = _parse_deps(col.value_expr, col_name)

                columns.append(col_entry)

            # Indices
            raw_indices = md['indices']
            indices: list[dict[str, Any]] = []
            for idx_name, idx_info in raw_indices.items():
                indices.append(
                    {
                        'name': idx_name,
                        'columns': idx_info['columns'],
                        'type': idx_info['index_type'],
                        'embedding': str(idx_info['parameters']['embedding'])[:120],
                    }
                )

            base_path = md['base']

            nodes.append(
                {
                    'path': path,
                    'name': tbl._name,
                    'is_view': is_view,
                    'base': base_path,
                    'row_count': row_count,
                    'version': md['version'],
                    'total_errors': table_error_total,
                    'columns': columns,
                    'indices': indices,
                    'versions': tbl.get_versions(),
                    'computed_count': len(computed_cols),
                    'insertable_count': len(columns) - len(computed_cols),
                    'iterator_type': iterator_name,
                }
            )

            if is_view and base_path:
                edges.append({'source': base_path, 'target': path, 'type': 'view', 'label': iterator_name or 'view'})

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
                    'total_errors': 0,
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
    all_tables = pxt.list_tables('', recursive=True)

    total_errors = 0
    for path in all_tables:
        try:
            total_errors += _version_error_total(pxt.get_table(path))
        except Exception:
            pass

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
        'total_tables': len(all_tables),
        'total_errors': total_errors,
        'config': config_info,
    }
