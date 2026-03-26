// API response types

export interface TreeNode {
  name: string;
  path: string;
  kind: 'directory' | 'table' | 'view' | 'snapshot' | 'replica';
  version?: number | null;
  error_count?: number;
  children?: TreeNode[];
}

// Matches Python ColumnMetadata TypedDict
export interface ColumnInfo {
  name: string;
  type_: string;
  version_added: number;
  is_stored: boolean;
  is_primary_key: boolean;
  media_validation: 'on_read' | 'on_write' | null;
  is_computed: boolean;
  computed_with: string | null;
  defined_in: string | null;
  custom_metadata: unknown;
  comment: string | null;
  is_iterator_col: boolean;
  destination: string | null;
}

// Matches Python EmbeddingIndexParams TypedDict
export interface EmbeddingIndexParams {
  metric: 'cosine' | 'ip' | 'l2';
  embedding: string;
  embedding_functions: string[];
}

// Matches Python IndexMetadata TypedDict
export interface IndexInfo {
  name: string;
  columns: string[];
  index_type: string;
  parameters: EmbeddingIndexParams;
}

// Matches Python TableMetadata TypedDict
export interface TableMetadata {
  name: string;
  path: string;
  columns: Record<string, ColumnInfo>;
  indices: Record<string, IndexInfo>;
  is_replica: boolean;
  is_view: boolean;
  is_snapshot: boolean;
  version: number;
  version_created: string;
  schema_version: number;
  comment: string | null;
  custom_metadata: unknown;
  media_validation: 'on_read' | 'on_write';
  kind: 'table' | 'view' | 'snapshot' | 'replica';
  base: string | null;
  iterator_expr: string | null;
}

export interface DataColumn {
  name: string;
  type: string;
  is_media: boolean;
  is_computed: boolean;
}

export interface CellError {
  error_type: string;
  error_msg: string;
}

export interface DataRow {
  [key: string]: unknown;
  _errors?: Record<string, CellError>;
}

export interface TableData {
  columns: DataColumn[];
  rows: DataRow[];
  total_count: number;
  offset: number;
  limit: number;
}

export interface SearchResults {
  query: string;
  directories: { path: string; name: string }[];
  tables: { path: string; name: string; kind: string }[];
  columns: { name: string; table: string; type: string; is_computed: boolean }[];
}

// ── Pipeline Inspector ──────────────────────────────────────────────────────

export interface PipelineColumn {
  name: string;
  type: string;
  is_computed: boolean;
  is_iterator_col: boolean;
  computed_with: string | null;
  defined_in: string | null;
  defined_in_self: boolean;
  func_name: string | null;
  func_type: 'builtin' | 'custom_udf' | 'query' | 'iterator' | 'unknown' | null;
  error_count: number;
  depends_on?: string[];
  comment?: string;
}

export interface PipelineIndex {
  name: string;
  columns: string[];
  type: string;
  embedding: string;
}

export interface PipelineVersion {
  version: number;
  created_at: string | null;
  change_type: string | null;
  inserts: number;
  updates: number;
  deletes: number;
  errors: number;
}

export interface PipelineNode extends Record<string, unknown> {
  path: string;
  name: string;
  is_view: boolean;
  base: string | null;
  row_count: number;
  version: number;
  total_errors: number;
  columns: PipelineColumn[];
  indices: PipelineIndex[];
  versions: PipelineVersion[];
  computed_count: number;
  insertable_count: number;
  iterator_type: string | null;
  error?: string;
}

export interface PipelineEdge {
  source: string;
  target: string;
  type: string;
  label: string;
}

export interface PipelineResponse {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
}
