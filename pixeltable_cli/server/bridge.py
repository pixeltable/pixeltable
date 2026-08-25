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
from typing import TYPE_CHECKING, Any, Literal, cast

import pixeltable as pxt
from pixeltable import exceptions as excs, exprs
from pixeltable.catalog import Path as CatalogPath, model
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.utils.app_module import load_model_bases, load_services
from pixeltable_cli import schema_types, service_types
from pixeltable_cli.utils import PxtPath

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import fastapi

    from pixeltable import exprs
    from pixeltable.serving import FastAPIRouter
    from pixeltable.serving._diff import ServiceChangeOp


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
        c for idx in md['indexes'].values() if idx['index_type'] == 'btree' for c in idx['columns']
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
                'is_iterator_col': info['is_iterator_col'],
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


def _search_catalog(root: str, query_lower: str, results: dict[str, Any]) -> None:
    """Add the matches found under one catalog root to results, in place."""
    for dir_path in pxt.list_dirs(root, recursive=True):
        if query_lower in dir_path.lower():
            results['directories'].append({'path': dir_path, 'name': dir_path.split('/')[-1]})

    for tbl_path in pxt.list_tables(root, recursive=True):
        # we get metadata for every table: a column match is only visible in the table's metadata
        try:
            tbl_md = pxt.get_table(tbl_path).get_metadata()
        except Exception as e:
            _logger.warning(f'Search: could not inspect {tbl_path}: {e}')
            results['unavailable'].append({'path': tbl_path, 'kind': 'table', 'error': f'{type(e).__name__}: {e}'})
            continue

        if query_lower in tbl_path.lower():
            results['tables'].append({'path': tbl_path, 'name': tbl_path.split('/')[-1], 'kind': tbl_md['kind']})

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


def search(query: str, additional_db_uris: list[str] | None = None) -> dict[str, Any]:
    """
    Search across directories, tables, and columns in the local catalog and any additional catalogs.

    The local (in-process) catalog is always searched; additional_db_uris holds hosted db uris to
    search as well. Result paths are full and resolvable in their catalog.

    A catalog that cannot be listed, or an error in get_table(), is reported under 'unavailable'
    rather than failing the search or appearing in the results with fabricated metadata.
    """
    query_lower = query.lower()

    results: dict[str, Any] = {'query': query, 'directories': [], 'tables': [], 'columns': [], 'unavailable': []}

    # The local catalog is the empty root; each additional catalog is searched at its hosted-uri root.
    roots = ['', *(additional_db_uris or [])]

    for root in roots:
        try:
            _search_catalog(root, query_lower, results)
        except Exception as e:
            _logger.warning(f'Search: could not list catalog {root or "local"}: {e}')
            results['unavailable'].append(
                {'path': root or 'local', 'kind': 'catalog', 'error': f'{type(e).__name__}: {e}'}
            )

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
            for idx_name, idx_info in md['indexes'].items():
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
        project_root = Env.project_root
        config_info = {
            'home': str(cfg.home),
            'project_root': None if project_root is None else str(project_root),
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


def service_diff(app_file: str, target: PxtPath) -> service_types.ServicePlan:
    """The changes that reconciling the services deployed at target with the ones app_file declares would make.

    Read-only: nothing is started, stopped or forgotten.

    Args:
        app_file: the application file declaring the services.
        target: the catalog directory the services' models bind against.
    """
    services = load_services(app_file)
    diffs = [_service_diff(name, service, app_file, target) for name, service in sorted(services.items())]
    return _plan_from_service_diffs(diffs, app_file, target)


def _service_diff(
    name: str, service: FastAPIRouter | fastapi.FastAPI, app_file: str, target: PxtPath
) -> service_types.ServiceDiff:
    """How the deployment of one declared service at target differs from its declaration."""
    # imported here rather than at module scope: pixeltable.serving pulls in fastapi, an optional dependency
    from pixeltable.serving import FastAPIRouter
    from pixeltable.serving._diff import blocked_schema_op, compare_specs
    from pixeltable.serving.service_deployment import ServiceDeployment

    deployment = ServiceDeployment.read(name, target)
    ops: list[service_types.ServiceChangeOp] = []
    route_detail: str | None = None
    if isinstance(service, FastAPIRouter):
        kind: Literal['declarative', 'custom'] = 'declarative'
        if deployment is None:
            route_comparison: service_types.RouteComparison = 'unavailable'
            route_detail = 'the service is not deployed at this target'
        else:
            route_comparison = 'declarative'
            ops += [_service_plan_op(op) for op in compare_specs(deployment.spec, service.service_spec(name))]
        # the models the routes name have to describe the tables at the target, whether or not the service is
        # deployed; _validate_model_routes() reports the discrepancy without binding anything
        try:
            service._validate_model_routes(target)
        except excs.Error as e:
            command = f'pxt schema update {app_file}' + ('' if target == '' else f' {target}')
            ops.append(_service_plan_op(blocked_schema_op(name, e.message, command)))
    else:
        kind = 'custom'
        route_comparison = 'unavailable'
        route_detail = 'the file supplies its own application object, whose routes Pixeltable did not declare'

    resolution: service_types.ServiceResolution
    if kind == 'custom':
        resolution = 'unsupported'
    elif any(op['severity'] == 'blocked' for op in ops):
        # the database has to change before this deployment can serve, whether it exists yet or not
        resolution = 'blocked'
    elif deployment is None:
        resolution = 'create'
    elif any(op['destructive'] for op in ops):
        resolution = 'update_destructive'
    elif len(ops) > 0:
        resolution = 'update_additive'
    else:
        resolution = 'up_to_date'

    return {
        'name': name,
        'exists': deployment is not None,
        # a local deployment is running or it is not recorded at all
        'state': None if deployment is None else 'AVAILABLE',
        'endpoint': None if deployment is None else deployment.endpoint,
        'base_path': target,
        'kind': kind,
        'resolution': resolution,
        'route_comparison': route_comparison,
        'route_detail': route_detail,
        'ops': ops,
        'destructive': any(op['destructive'] for op in ops),
        'requires_restart': deployment is not None and any(op['requires_restart'] for op in ops),
    }


def _plan_from_service_diffs(
    diffs: list[service_types.ServiceDiff], app_file: str, target: PxtPath
) -> service_types.ServicePlan:
    """The plan that the given per-service diffs describe."""
    from pixeltable.serving.service_deployment import ServiceDeployment

    declared = {diff['name'] for diff in diffs}
    extras = sorted(d.service_name for d in ServiceDeployment.list(target) if d.service_name not in declared)
    summary: service_types.ServicePlanSummary = {
        'up_to_date': sum(1 for d in diffs if d['resolution'] == 'up_to_date'),
        'create': sum(1 for d in diffs if d['resolution'] == 'create'),
        'update_additive': sum(1 for d in diffs if d['resolution'] == 'update_additive'),
        'update_destructive': sum(1 for d in diffs if d['resolution'] == 'update_destructive'),
        'unsupported': sum(1 for d in diffs if d['resolution'] == 'unsupported'),
        'blocked': sum(1 for d in diffs if d['resolution'] == 'blocked'),
        'extras': len(extras),
        'destructive': sum(1 for d in diffs for op in d['ops'] if op['destructive']),
        'blocked_ops': sum(1 for d in diffs for op in d['ops'] if op['severity'] == 'blocked'),
        'restarts': sum(1 for d in diffs if d['requires_restart']),
    }
    return {
        'app_file': app_file,
        'target': target,
        # extras are excluded: update never removes a deployment, which is what prune is for
        'in_agreement': all(d['resolution'] == 'up_to_date' for d in diffs),
        'services': diffs,
        'extras': extras,
        'summary': summary,
    }


def _service_plan_op(op: ServiceChangeOp) -> service_types.ServiceChangeOp:
    """The CLI-side form of a service operation."""
    return {
        'target': op['target'],
        'name': op['name'],
        'op': op['op'],
        'severity': op['severity'],
        'description': op['description'],
        'details': op['details'],
        'destructive': op['severity'] == 'destructive',
        # a blocked operation is never applied; every other one replaces what the deployment serves
        'requires_restart': op['severity'] != 'blocked',
    }


# close the refusals raised while reconciling, in place of the Python API's wording
_DESTRUCTIVE_HINT = "Re-run 'pxt schema update' with --allow-destructive to apply these changes."
_SERVICE_DESTRUCTIVE_HINT = "Re-run 'pxt service update' with --allow-destructive to apply these changes."


def _path_key(pxt_path: PxtPath) -> tuple[str, ...]:
    """A comparable identity for a table path, so that a pxt:// URI and a bare path denote the same table."""
    return tuple(CatalogPath.parse(pxt_path, allow_empty_path=True).components)


def _list_tables(pxt_path: PxtPath) -> list[PxtPath]:
    """Paths of the tables under the given path, or [] if it does not exist."""
    try:
        return [PxtPath(p) for p in pxt.list_tables(pxt_path, recursive=True)]
    except excs.NotFoundError:
        return []


def schema_diff(schema_file: str, catalog_dir: PxtPath) -> schema_types.SchemaPlan:
    """The changes that schema_update() would make to reconcile the catalog directory with the schema file.

    Read-only: never creates the catalog directory, and never touches an existing table.
    """
    return _schema_plan(load_model_bases(schema_file), schema_file, catalog_dir)


def _schema_plan(bases: list[model.TableModelMeta], schema_file: str, catalog_dir: PxtPath) -> schema_types.SchemaPlan:
    """The plan for reconciling the catalog directory with the models declared by the given bases."""
    diffs = [diff for base in bases for diff in base.get_model_diff(catalog_dir).values()]
    return _plan_from_diffs(diffs, schema_file, catalog_dir)


def _plan_from_diffs(diffs: list[model.TableDiff], schema_file: str, catalog_dir: PxtPath) -> schema_types.SchemaPlan:
    """The plan that the given per-table diffs describe."""
    tables: list[schema_types.TableDiff] = []
    for diff in diffs:
        # a create subsumes the additions that constitute it, so only a migration enumerates operations
        enumerated = [] if diff['resolution'] in ('create', 'up_to_date') else diff['ops']
        ops = [_plan_op(op) for op in enumerated]
        tables.append(
            {
                'path': diff['path'],
                'model_cls': diff['model_cls'],
                'kind': diff['kind'],
                'exists': diff['exists'],
                'resolution': diff['resolution'],
                'ops': ops,
                'destructive': any(op['destructive'] for op in ops),
            }
        )

    # a table's path crosses from the catalog as a plain string
    declared = {_path_key(PxtPath(t['path'])) for t in tables}
    extras = sorted(p for p in _list_tables(catalog_dir) if _path_key(p) not in declared)
    summary: schema_types.SchemaPlanSummary = {
        'up_to_date': sum(1 for t in tables if t['resolution'] == 'up_to_date'),
        'create': sum(1 for t in tables if t['resolution'] == 'create'),
        'update_additive': sum(1 for t in tables if t['resolution'] == 'update_additive'),
        'update_destructive': sum(1 for t in tables if t['resolution'] == 'update_destructive'),
        'unsupported': sum(1 for t in tables if t['resolution'] == 'unsupported'),
        'extras': len(extras),
        'destructive': sum(1 for t in tables for op in t['ops'] if op['destructive']),
    }
    return {
        'schema_file': schema_file,
        'catalog_dir': catalog_dir,
        # extras are excluded: update() never removes them, so their presence is not something it could reconcile
        'in_agreement': all(t['resolution'] == 'up_to_date' for t in tables),
        'tables': tables,
        'extras': extras,
        'summary': summary,
    }


def _plan_op(op: model.SchemaChangeOp) -> schema_types.SchemaChangeOp:
    """The CLI-side form of a model operation: everything but the model-side and catalog-side values."""
    return {
        'target': op['target'],
        'name': op['name'],
        'op': op['op'],
        'severity': op['severity'],
        'description': op['description'],
        'details': op['details'],
        'destructive': op['severity'] == 'destructive',
    }


def schema_prune(schema_file: str, catalog_dir: PxtPath) -> schema_types.SchemaPlan:
    """Drop the tables under catalog_dir that no model in the schema file declares.

    Returns the plan, with one drop_table operation per dropped table. A view is dropped before its base, so that
    pruning a group of related tables does not depend on the order they are listed in. Nothing is force-dropped:
    a table that something outside the pruned set depends on is left in place and its error is raised.
    If this exits with an error, it may have dropped a partial list of tables.
    """
    plan = _schema_plan(load_model_bases(schema_file), schema_file, catalog_dir)
    remaining = list(plan['extras'])
    dropped: list[PxtPath] = []
    while len(remaining) > 0:
        deferred: list[PxtPath] = []
        blocked_by: excs.Error | None = None
        for pxt_path in remaining:
            try:
                pxt.drop_table(pxt_path, if_not_exists='ignore')
            except excs.Error as e:
                blocked_by = e
                deferred.append(pxt_path)
                continue
            dropped.append(pxt_path)
        if len(deferred) == len(remaining):
            assert blocked_by is not None
            if len(dropped) > 0:
                # the drops so far are already committed; name them, so the error doesn't read as though the
                # catalog were untouched. Augmenting in place keeps the exception's type and fields, and
                # blocked_by.message excludes blocked_by.detail, which must not become part of the message.
                names = ', '.join(repr(pxt_path) for pxt_path in dropped)
                blocked_by.args = (f'{blocked_by.message}\n\nThe following table(s) were already dropped: {names}.',)
            raise blocked_by
        remaining = deferred

    plan['ops'] = [schema_types.drop_table_op(pxt_path, 'applied') for pxt_path in dropped]
    return plan


def schema_update(
    schema_file: str, catalog_dir: PxtPath, *, allow_destructive: bool = False
) -> schema_types.SchemaPlan:
    """Reconcile the tree under catalog_dir with the schema file: create missing tables and migrate existing ones.

    Returns the plan that was applied, each operation annotated with its status.
    """
    bases = load_model_bases(schema_file)

    # TODO: refuse a hosted target whose runtime does not hold the modules these udfs live in. A udf is now
    # referred to by a module path, which a hosted runtime resolves from the project it was built from, so
    # what has to be checked is whether that project's build context holds the module.

    # only create catalog_dir when it names an in-catalog path; a bare catalog root (eg '' or 'pxt://org:db')
    # has no directory to create
    if len(CatalogPath.parse(catalog_dir, allow_empty_path=True).components) > 0:
        pxt.create_dir(catalog_dir, parents=True, if_exists='ignore')

    applied: list[model.TableDiff] = []
    for base in bases:
        try:
            diffs = base.update_all(catalog_dir, allow_destructive=allow_destructive)
        except excs.RequestError as e:
            if e.error_code is not excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE:
                raise
            # update_all() closes its refusal with instructions for the Python API; a CLI user needs the flag
            e.args = (e.message.replace(model.PY_DESTRUCTIVE_HINT, _DESTRUCTIVE_HINT),)
            raise
        applied.extend(diffs.values())

    plan = _plan_from_diffs(applied, schema_file, catalog_dir)
    for tbl in plan['tables']:
        tbl['status'] = 'skipped' if tbl['resolution'] == 'up_to_date' else 'applied'
        for op in tbl['ops']:
            op['status'] = 'applied'
    return plan


def service_update(app_file: str, target: PxtPath, *, allow_destructive: bool = False) -> service_types.ServicePlan:
    """Reconcile the deployments at target with the services app_file declares, and leave them running.

    Starts a service that is not deployed, and restarts one whose declaration changed, since a service binds
    its models once per process. A deployment the file does not declare is left alone, which is what prune is
    for.

    Returns the plan that was applied, each service annotated with its status: 'applied' for one that was
    started or restarted, 'skipped' for one already serving its declaration, 'refused' for one whose routes
    the database cannot serve or whose application object Pixeltable did not declare.

    Args:
        app_file: the application file declaring the services.
        target: the catalog directory the services' models bind against.
        allow_destructive: whether to apply changes that stop serving a route contract a caller may be using.
    """
    from pixeltable.serving.service_deployment import ServiceDeployment

    plan = service_diff(app_file, target)
    destructive = [d['name'] for d in plan['services'] if d['resolution'] == 'update_destructive']
    if len(destructive) > 0 and not allow_destructive:
        names = ', '.join(repr(name) for name in destructive)
        for diff in plan['services']:
            diff['status'] = 'refused'
        raise excs.RequestError(
            excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
            f'Reconciling {names} would stop serving a route that callers may be using.\n{_SERVICE_DESTRUCTIVE_HINT}',
        )

    for diff in plan['services']:
        if diff['resolution'] in ('unsupported', 'blocked'):
            diff['status'] = 'refused'
            continue
        if diff['resolution'] == 'up_to_date':
            diff['status'] = 'skipped'
            continue
        if diff['exists']:
            # the deployment serves the old declaration; binding happens once per process, so it is replaced
            deployment = ServiceDeployment.read(diff['name'], target)
            if deployment is not None:
                deployment.stop()
        started = ServiceDeployment.start(app_file, diff['name'], target)
        diff['status'] = 'applied'
        diff['state'] = 'AVAILABLE'
        diff['endpoint'] = started.endpoint
        diff['exists'] = True
        for op in diff['ops']:
            op['status'] = 'skipped' if op['severity'] == 'blocked' else 'applied'
    return plan


def service_prune(app_file: str, target: PxtPath) -> service_types.ServicePlan:
    """Stop and forget the deployments at target that app_file does not declare.

    A stopped service can be started again, so this is not destructive the way dropping a table is.

    Returns the plan, with one drop operation per deployment stopped.
    """
    from pixeltable.serving.service_deployment import ServiceDeployment

    plan = service_diff(app_file, target)
    ops: list[service_types.ServiceChangeOp] = []
    for name in plan['extras']:
        deployment = ServiceDeployment.read(name, target)
        if deployment is None:
            # it stopped between the diff and here; nothing to stop, and it is already forgotten
            ops.append(service_types.delete_service_op(name, None, 'skipped'))
            continue
        deployment.stop()
        ops.append(service_types.delete_service_op(name, deployment.endpoint, 'applied'))
    plan['ops'] = ops
    return plan


def service_stop(names: list[str], target: PxtPath) -> list[service_types.ServiceChangeOp]:
    """Stop the named services deployed at target and forget them.

    A name that is not deployed there yields a 'skipped' operation rather than an error, so that stopping a
    set of services is idempotent.
    """
    from pixeltable.serving.service_deployment import ServiceDeployment

    ops: list[service_types.ServiceChangeOp] = []
    for name in names:
        deployment = ServiceDeployment.read(name, target)
        if deployment is None:
            ops.append(service_types.delete_service_op(name, None, 'skipped'))
            continue
        deployment.stop()
        ops.append(service_types.delete_service_op(name, deployment.endpoint, 'applied'))
    return ops


def service_list(target: PxtPath | None = None) -> list[service_types.ServiceDeployment]:
    """The services running locally: those bound at target and below it, or all of them if target is None."""
    from pixeltable.serving.service_deployment import ServiceDeployment

    deployments = ServiceDeployment.list('' if target is None else target, recursive=True)
    return [
        service_types.ServiceDeployment(
            name=d.service_name,
            base_path=PxtPath(d.base_path),
            endpoint=d.endpoint,
            pid=d.pid,
            process_started_at=d.process_started_at,
            app_file=d.app_file,
            spec=cast(service_types.ServiceSpec, d.spec),
        )
        for d in sorted(deployments, key=lambda d: (d.base_path, d.service_name))
    ]
