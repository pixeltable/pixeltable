import type {
  TreeNode,
  TableMetadata,
  TableData,
  SearchResults,
  PipelineResponse,
} from '@/types';

const API_BASE = '/api';

/** Daemon errors are `{ detail, error_code }` (see pixeltable_cli/server/http_server.py). */
const ERROR_LABELS: Record<string, string> = {
  MISSING_CREDENTIALS:
    'No Pixeltable API key. Set PIXELTABLE_API_KEY or add api_key under [pixeltable] in your Pixeltable config file.',
  INSUFFICIENT_PRIVILEGES: 'Not allowed to open this catalog.',
  INVALID_PATH: 'Invalid path.',
  PATH_NOT_FOUND: 'Not found.',
  DIRECTORY_NOT_FOUND: 'Not found.',
  TABLE_NOT_FOUND: 'Not found.',
};

function apiErrorMessage(body: { detail?: unknown; error?: unknown; error_code?: unknown }, status: number): string {
  const code = typeof body.error_code === 'string' ? body.error_code : '';
  const detail = typeof body.detail === 'string' ? body.detail : typeof body.error === 'string' ? body.error : '';
  const label = ERROR_LABELS[code];
  if (label) return label;
  return detail || `HTTP ${status}`;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(apiErrorMessage(error, response.status));
  }
  return response.json();
}

// GET /api/dirs returns an LsResponse object (see LsResponse in pixeltable_cli/models.py), not a
// bare array. With tree=true the recursive DirectoryNode/TableNode list the UI renders lives under
// tree.entries; the flat entries field is empty in that mode.
interface LsResponse {
  entries: unknown[];
  tree: { path: string; entries: TreeNode[] } | null;
}

// List a directory's contents as a tree. Omit path (or pass 'local') for the in-process catalog root;
// pass a hosted catalog uri (e.g. 'pxt://org:db'), or any directory path in either catalog, to list that
// directory. The daemon reports each node's path relative to the queried directory, re-rooted so a hosted
// path carries its pxt:// prefix.
export async function getDirectoryTree(path?: string): Promise<TreeNode[]> {
  const params = new URLSearchParams({ tree: 'true' });
  if (path !== undefined && path !== '' && path !== 'local') params.set('path', path);
  const res = await fetchJson<LsResponse>(`${API_BASE}/dirs?${params}`);
  return res.tree?.entries ?? [];
}

export async function getTableMetadata(path: string): Promise<TableMetadata> {
  return fetchJson<TableMetadata>(`${API_BASE}/dashboard/tables/meta?path=${encodeURIComponent(path)}`);
}

export async function getTableData(
  path: string,
  options: {
    offset?: number;
    limit?: number;
    orderBy?: string;
    orderDesc?: boolean;
    errorsOnly?: boolean;
  } = {}
): Promise<TableData> {
  const params = new URLSearchParams({ path });
  if (options.offset !== undefined) params.set('offset', String(options.offset));
  if (options.limit !== undefined) params.set('limit', String(options.limit));
  if (options.orderBy) params.set('order_by', options.orderBy);
  if (options.orderDesc) params.set('order_desc', 'true');
  if (options.errorsOnly) params.set('errors_only', 'true');

  return fetchJson<TableData>(`${API_BASE}/dashboard/tables/data?${params.toString()}`);
}

// Search the local catalog plus any additional (hosted) catalogs; the daemon always includes the local
// catalog and returns full, resolvable paths.
export async function search(query: string, additionalCatalogs?: string[]): Promise<SearchResults> {
  const params = new URLSearchParams({ q: query });
  for (const c of additionalCatalogs ?? []) params.append('catalogs', c);
  return fetchJson<SearchResults>(`${API_BASE}/dashboard/search?${params}`);
}

export async function getPipeline(tablePath?: string): Promise<PipelineResponse> {
  const url = tablePath !== undefined
    ? `${API_BASE}/dashboard/tables/pipeline?path=${encodeURIComponent(tablePath)}`
    : `${API_BASE}/dashboard/pipeline`;
  return fetchJson<PipelineResponse>(url);
}

interface SystemConfig {
  home: string;
  db_url: string;
  media_dir: string;
  file_cache_dir: string;
}

export interface SystemStatus {
  version: string;
  total_tables: number;
  total_errors: number;
  config?: SystemConfig;
}

// Flat shape returned by GET /api/status (see StatusResponse in pixeltable_cli/models.py). The UI
// consumes the nested {version, config} shape below, so map the response rather than asserting it.
interface StatusResponse {
  pxt_version: string;
  home: string | null;
  db_url: string | null;
  media_dir: string | null;
  file_cache_dir: string | null;
  total_tables: number;
  total_errors: number;
}

export async function getStatus(): Promise<SystemStatus> {
  const s = await fetchJson<StatusResponse>(`${API_BASE}/status`);
  return {
    version: s.pxt_version,
    total_tables: s.total_tables,
    total_errors: s.total_errors,
    config: {
      home: s.home ?? '',
      db_url: s.db_url ?? '',
      media_dir: s.media_dir ?? '',
      file_cache_dir: s.file_cache_dir ?? '',
    },
  };
}
