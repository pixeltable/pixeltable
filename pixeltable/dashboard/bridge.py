"""
Bridge layer between Pixeltable internal APIs and the Dashboard REST API.

This module translates Pixeltable's internal data structures into JSON-serializable
formats suitable for the dashboard frontend.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

import pixeltable as pxt
from pixeltable.catalog.table import Table
from pixeltable.catalog.table_metadata import TableMetadata
from pixeltable.env import Env

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[assignment,misc]

_logger = logging.getLogger('pixeltable.dashboard')


# ── Helpers ──────────────────────────────────────────────────────────────────


def _version_error_total(tbl: Table) -> int:
    """Sum errors across all versions of a table (cheap, no row scans)."""
    try:
        return sum(v.get('errors', 0) for v in tbl.get_versions())
    except Exception:
        return 0


def _column_error_counts(tbl: Table, col_meta: dict[str, Any]) -> dict[str, int]:
    """Count rows with errors per computed or media column. Returns {col_name: count}."""
    counts: dict[str, int] = {}
    for col_name, info in col_meta.items():
        is_computed = info.get('computed_with') is not None
        is_media = _is_media_type(info.get('type_', ''))
        if not is_computed and not is_media:
            continue
        try:
            col_ref = getattr(tbl, col_name)
            counts[col_name] = tbl.where(col_ref.errortype != None).count()
        except Exception:
            counts[col_name] = 0
    return counts


def _table_kind(md: TableMetadata) -> str:
    """Determine the kind of a table from its metadata dict."""
    if md['is_replica']:
        return 'replica'
    if md['is_snapshot']:
        return 'snapshot'
    if md['is_view']:
        return 'view'
    return 'table'


def _extract_indices(raw_indices: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert raw index metadata into a serialisable list."""
    indices: list[dict[str, Any]] = []
    for idx_name, idx_info in raw_indices.items():
        idx_columns = idx_info.get('columns', [])
        column_str = ', '.join(idx_columns) if idx_columns else idx_info.get('column', '')
        indices.append(
            {
                'name': idx_info.get('name', idx_name),
                'column': column_str,
                'type_': idx_info.get('index_type', idx_info.get('type_', 'Unknown')),
                'parameters': idx_info.get('parameters', {}),
            }
        )
    return indices


def _format_versions(tbl: Table) -> list[dict[str, Any]]:
    """Build a serialisable list of version dicts from a table's version history."""
    versions: list[dict[str, Any]] = []
    try:
        for v in tbl.get_versions():
            versions.append(
                {
                    'version': v['version'],
                    'created_at': v['created_at'].isoformat() if v.get('created_at') else None,
                    'change_type': v.get('change_type'),
                    'inserts': v.get('inserts', 0),
                    'updates': v.get('updates', 0),
                    'deletes': v.get('deletes', 0),
                    'errors': v.get('errors', 0),
                }
            )
    except Exception:
        pass
    return versions


_MEDIA_TYPES = frozenset({'image', 'video', 'audio', 'document'})


def _is_media_type(col_type: str) -> bool:
    """Check if a column type string represents a media type."""
    t = col_type.lower()
    return any(m in t for m in _MEDIA_TYPES)


def _build_select(
    tbl: Table, col_meta: dict[str, Any], *, include_errors: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str], dict[str, tuple[str, str]]]:
    """Build column info list, select dict, media URL map, and error column map.

    Returns (columns, select_dict, media_url_cols, error_cols).
    """
    columns: list[dict[str, Any]] = []
    select_dict: dict[str, Any] = {}
    media_url_cols: dict[str, str] = {}
    error_cols: dict[str, tuple[str, str]] = {}

    for col_name, col_info in col_meta.items():
        col_type = col_info.get('type_', 'Unknown')
        is_media = _is_media_type(col_type)
        is_computed = col_info.get('computed_with') is not None
        columns.append({'name': col_name, 'type': col_type, 'is_media': is_media, 'is_computed': is_computed})

        col_ref = getattr(tbl, col_name)
        select_dict[col_name] = col_ref

        if is_media:
            url_key = f'{col_name}__url'
            select_dict[url_key] = col_ref.fileurl
            media_url_cols[col_name] = url_key

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


# ── Directory / tree ─────────────────────────────────────────────────────────


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
        node = {'name': parts[-1], 'path': dir_path, 'type': 'directory', 'children': []}
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
            kind = _table_kind(md)
            version = md['version'] if kind != 'snapshot' else None
            error_count = _version_error_total(tbl)
        except Exception as e:
            _logger.warning(f'Failed to get metadata for {tbl_path}: {e}')
            kind = 'table'
            version = None

        table_node = {'name': tbl_name, 'path': tbl_path, 'type': kind, 'version': version, 'error_count': error_count}

        if parent_path and parent_path in dir_nodes:
            dir_nodes[parent_path]['children'].append(table_node)
        elif not parent_path:
            root_children.append(table_node)

    return root_children


# ── Table metadata / data ────────────────────────────────────────────────────


def get_table_metadata(table_path: str) -> dict[str, Any]:
    """
    Get detailed metadata for a table including schema, indices, and lineage info.
    """
    tbl = pxt.get_table(table_path)
    md = tbl.get_metadata()
    kind = _table_kind(md)

    columns = []
    for col_name, col_info in md['columns'].items():
        columns.append(
            {
                'name': col_info.get('name', col_name),
                'type': col_info.get('type_', 'Unknown'),
                'is_computed': col_info.get('computed_with') is not None,
                'computed_with': col_info.get('computed_with'),
                'is_stored': col_info.get('is_stored', True),
                'is_primary_key': col_info.get('is_primary_key', False),
                'defined_in': col_info.get('defined_in'),
                'version_added': col_info.get('version_added', 0),
                'comment': col_info.get('comment') or None,
            }
        )

    return {
        'path': md['path'],
        'name': md['name'],
        'type': kind,
        'version': md['version'],
        'schema_version': md['schema_version'],
        'created_at': md['version_created'].isoformat() if md['version_created'] else None,
        'comment': md['comment'],
        'base': md['base'],
        'columns': columns,
        'indices': _extract_indices(md.get('indices', {})),
        'media_validation': md['media_validation'],
        'versions': _format_versions(tbl),
    }


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
    md = tbl.get_metadata()
    http_address = Env.get().http_address

    columns, select_dict, media_url_cols, error_cols = _build_select(tbl, md['columns'], include_errors=True)

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
                if fileurl:
                    row_data[col_name] = _resolve_fileurl(fileurl, http_address)
                elif value is not None and PILImage is not None:
                    try:
                        if isinstance(value, PILImage.Image):
                            buf = io.BytesIO()
                            fmt = 'JPEG' if value.mode == 'RGB' else 'PNG'
                            value.save(buf, format=fmt, quality=80)
                            b64 = base64.b64encode(buf.getvalue()).decode()
                            row_data[col_name] = f'data:image/{fmt.lower()};base64,{b64}'
                        else:
                            row_data[col_name] = None
                    except Exception:
                        row_data[col_name] = None
                else:
                    row_data[col_name] = None
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


# ── CSV export ────────────────────────────────────────────────────────────────


def export_table_csv(table_path: str, limit: int = 100_000) -> bytes:
    """Export a table as CSV bytes. Media columns export their file URL."""
    tbl = pxt.get_table(table_path)
    md = tbl.get_metadata()
    http_address = Env.get().http_address

    columns, select_dict, media_url_cols, _ = _build_select(tbl, md['columns'])
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


# ── Search ───────────────────────────────────────────────────────────────────


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
                    results['tables'].append({'path': tbl_path, 'name': tbl_name, 'type': 'table'})
                continue

        if table_matches and len(results['tables']) < limit and tbl_md:
            results['tables'].append({'path': tbl_path, 'name': tbl_name, 'type': _table_kind(tbl_md)})

        # Search columns within this table (reuse tbl_md)
        if tbl_md and len(results['columns']) < limit:
            for col_name, col_info in tbl_md['columns'].items():
                if query_lower in col_name.lower():
                    results['columns'].append(
                        {
                            'name': col_name,
                            'table': tbl_path,
                            'type': col_info.get('type_', 'Unknown'),
                            'is_computed': col_info.get('computed_with') is not None,
                        }
                    )
                    if len(results['columns']) >= limit:
                        break

    return results


# ── Pipeline helpers ─────────────────────────────────────────────────────────

_FUNC_CALL_RE = re.compile(r'(\w+)\s*\(')
_COL_REF_RE = re.compile(r'\b(\w+)\b')

_SKIP_FUNC_NAMES = frozenset(
    {
        'model',
        'config',
        'type',
        'object',
        'items',
        'str',
        'get',
        'text',
        'int',
        'float',
        'bool',
        'list',
        'dict',
        'set',
        'tuple',
        'len',
        'range',
        'print',
        'format',
        'join',
        'split',
        'strip',
        'lower',
        'upper',
        'replace',
        'append',
        'extend',
        'keys',
        'values',
        'update',
        'pop',
        'map',
        'filter',
        'sorted',
        'enumerate',
        'zip',
        'any',
        'all',
        'sum',
        'min',
        'max',
        'abs',
        'isinstance',
        'hasattr',
        'getattr',
    }
)


def _extract_func_name(computed_with: str | None) -> str | None:
    """Extract the primary function name from a computed_with expression."""
    if not computed_with:
        return None
    for match in _FUNC_CALL_RE.finditer(computed_with):
        name = match.group(1)
        if name not in _SKIP_FUNC_NAMES:
            return name
    return None


_BUILTIN_PREFIXES = frozenset(
    {
        'openai',
        'anthropic',
        'together',
        'fireworks',
        'mistral',
        'replicate',
        'huggingface',
        'bedrock',
        'ollama',
        'whisper',
        'label_studio',
        'string',
        'image',
        'video',
        'audio',
        'timestamp',
        'json',
        'math',
        'nos',
        'sentence_transformer',
        'yolox',
        'detr',
        'clip',
    }
)


def _classify_func(computed_with: str | None) -> str:
    """Classify a computed_with expression as builtin, custom_udf, or unknown."""
    if not computed_with:
        return 'unknown'
    if '.apply(' in computed_with or 'lambda ' in computed_with:
        return 'custom_udf'
    first_call = _FUNC_CALL_RE.search(computed_with)
    if first_call:
        name = first_call.group(1)
        if name.split('.')[0] in _BUILTIN_PREFIXES or name.split('_')[0] in _BUILTIN_PREFIXES:
            return 'builtin'
    return 'unknown'


def _parse_deps(computed_with: str | None, all_cols: set[str], own_name: str = '') -> list[str]:
    """Extract column names referenced in a computed_with expression."""
    if not computed_with:
        return []
    tokens = _COL_REF_RE.findall(computed_with)
    return sorted({t for t in tokens if t in all_cols and t != own_name})


def _detect_iterator(columns: list[dict[str, Any]]) -> str | None:
    """Detect the iterator type used to create a view from its column shapes."""
    own_cols = {c['name'] for c in columns if c.get('defined_in_self')}
    if {'frame_idx', 'pos_frame', 'frame'} & own_cols:
        return 'FrameIterator'
    if {'audio_chunk'} & own_cols and {'start_time_sec', 'end_time_sec'} & own_cols:
        return 'AudioSplitter'
    if {'heading', 'page', 'title'} & own_cols and 'pos' in own_cols:
        return 'DocumentSplitter'
    if 'text' in own_cols and 'pos' in own_cols:
        return 'StringSplitter'
    return None


def get_pipeline() -> dict[str, Any]:
    """Return the full DAG metadata for the Pipeline Inspector."""
    table_paths = sorted(pxt.list_tables('', recursive=True))

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for path in table_paths:
        try:
            tbl = pxt.get_table(path)
            md = tbl.get_metadata()
            col_meta = md.get('columns', {})
            row_count = tbl.count()

            all_col_names = set(col_meta.keys())
            short_name = path.rsplit('/', 1)[-1]

            col_errors = _column_error_counts(tbl, col_meta)
            table_error_total = _version_error_total(tbl)

            columns: list[dict[str, Any]] = []
            computed_cols: list[str] = []

            for col_name, info in col_meta.items():
                cw = info.get('computed_with')
                is_computed = cw is not None
                if is_computed:
                    computed_cols.append(col_name)
                defined_in = info.get('defined_in')

                cw_str = str(cw)[:200] if cw else None
                func_name = _extract_func_name(cw_str) if is_computed else None

                col_entry: dict[str, Any] = {
                    'name': col_name,
                    'type': info.get('type_', 'unknown'),
                    'is_computed': is_computed,
                    'computed_with': cw_str,
                    'defined_in': defined_in,
                    'defined_in_self': defined_in == short_name,
                    'func_name': func_name,
                    'func_type': _classify_func(cw_str) if is_computed else None,
                    'error_count': col_errors.get(col_name, 0),
                }

                if is_computed and cw_str:
                    col_entry['depends_on'] = _parse_deps(cw_str, all_col_names, col_name)

                columns.append(col_entry)

            # Indices
            raw_indices = md.get('indices', {})
            indices: list[dict[str, Any]] = []
            for idx_name, idx_info in raw_indices.items():
                indices.append(
                    {
                        'name': idx_name,
                        'columns': idx_info.get('columns', []),
                        'type': idx_info.get('index_type', 'unknown'),
                        'embedding': str(idx_info.get('parameters', {}).get('embedding', ''))[:120],
                    }
                )

            base_path = md.get('base')
            is_view = md.get('is_view', False)
            iterator_type = _detect_iterator(columns) if is_view else None

            nodes.append(
                {
                    'path': path,
                    'name': short_name,
                    'is_view': is_view,
                    'base': base_path,
                    'row_count': row_count,
                    'version': md.get('version', 0),
                    'total_errors': table_error_total,
                    'columns': columns,
                    'indices': indices,
                    'versions': _format_versions(tbl),
                    'computed_count': len(computed_cols),
                    'insertable_count': len(columns) - len(computed_cols),
                    'iterator_type': iterator_type,
                }
            )

            if is_view and base_path:
                edges.append({'source': base_path, 'target': path, 'type': 'view', 'label': iterator_type or 'view'})

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


# ── Status ───────────────────────────────────────────────────────────────────


def get_status() -> dict[str, Any]:
    """
    Get system status including version, environment, connection info, and table count.
    """
    import pixeltable
    from pixeltable.config import Config

    version = getattr(pixeltable, '__version__', 'unknown')
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
