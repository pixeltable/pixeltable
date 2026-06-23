import type {
  TreeNode,
  TableMetadata,
  TableData,
  SearchResults,
  PipelineResponse,
} from '@/types';

const API_BASE = '/api';

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(error.error || `HTTP ${response.status}`);
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

export async function getDirectoryTree(): Promise<TreeNode[]> {
  const res = await fetchJson<LsResponse>(`${API_BASE}/dirs?tree=true`);
  return res.tree?.entries ?? [];
}

export async function getTableMetadata(path: string): Promise<TableMetadata> {
  return fetchJson<TableMetadata>(`${API_BASE}/dashboard/tables/${encodeURIComponent(path)}/meta`);
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
  const params = new URLSearchParams();
  if (options.offset !== undefined) params.set('offset', String(options.offset));
  if (options.limit !== undefined) params.set('limit', String(options.limit));
  if (options.orderBy) params.set('order_by', options.orderBy);
  if (options.orderDesc) params.set('order_desc', 'true');
  if (options.errorsOnly) params.set('errors_only', 'true');

  const query = params.toString();
  return fetchJson<TableData>(`${API_BASE}/dashboard/tables/${encodeURIComponent(path)}/data${query ? `?${query}` : ''}`);
}

export async function search(query: string, limit = 50): Promise<SearchResults> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return fetchJson<SearchResults>(`${API_BASE}/dashboard/search?${params}`);
}

export async function getPipeline(tablePath?: string): Promise<PipelineResponse> {
  const url = tablePath !== undefined
    ? `${API_BASE}/dashboard/tables/${encodeURIComponent(tablePath)}/pipeline`
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

