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

export async function getDirectoryTree(): Promise<TreeNode[]> {
  return fetchJson<TreeNode[]>(`${API_BASE}/dirs`);
}

export async function getTableMetadata(path: string): Promise<TableMetadata> {
  return fetchJson<TableMetadata>(`${API_BASE}/tables/meta/${encodeURIComponent(path)}`);
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
  return fetchJson<TableData>(`${API_BASE}/tables/data/${encodeURIComponent(path)}${query ? `?${query}` : ''}`);
}

export async function search(query: string, limit = 50): Promise<SearchResults> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return fetchJson<SearchResults>(`${API_BASE}/search?${params}`);
}

export async function getPipeline(): Promise<PipelineResponse> {
  return fetchJson<PipelineResponse>(`${API_BASE}/pipeline`);
}

interface SystemConfig {
  home: string;
  db_url: string;
  media_dir: string;
  file_cache_dir: string;
  is_local: boolean;
}

export interface SystemStatus {
  version: string;
  environment: 'local' | 'cloud';
  total_tables: number;
  total_errors: number;
  config?: SystemConfig;
}

export async function getStatus(): Promise<SystemStatus> {
  return fetchJson<SystemStatus>(`${API_BASE}/status`);
}

