// API response types

export interface TreeNode {
  name: string;
  path: string;
  type: 'directory' | 'table' | 'view' | 'snapshot' | 'replica';
  version?: number | null;
  error_count?: number;
  children?: TreeNode[];
}

export interface ColumnInfo {
  name: string;
  type: string;
  is_computed: boolean;
  computed_with: string | null;
  is_stored: boolean;
  is_primary_key: boolean;
  defined_in: string | null;
  version_added: number;
  comment: string | null;
}

export interface IndexInfo {
  name: string;
  column: string;
  type_: string;
  parameters: Record<string, unknown>;
}

export interface TableMetadata {
  path: string;
  name: string;
  type: 'table' | 'view' | 'snapshot' | 'replica';
  version: number;
  schema_version: number;
  created_at: string | null;
  comment: string | null;
  base: string | null;
  columns: ColumnInfo[];
  indices: IndexInfo[];
  media_validation: string;
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
  tables: { path: string; name: string; type: string }[];
  columns: { name: string; table: string; type: string; is_computed: boolean }[];
}

// Directory summary
export interface DirectorySummary {
  path: string;
  table_count: number;
  total_rows: number;
  total_errors: number;
  tables: {
    path: string;
    name: string;
    type: string;
    row_count: number;
    column_count: number;
    error_count: number;
    version: number;
  }[];
}

// Information Schema types
export interface TableErrors {
  table_path: string;
  column_error_counts: Record<string, number>;
  total_errors: number;
  samples: { column: string; errortype: string | null; errormsg: string | null }[];
}

// ── Pipeline Inspector ──────────────────────────────────────────────────────

export interface PipelineColumn {
  name: string;
  type: string;
  is_computed: boolean;
  computed_with: string | null;
  defined_in: string | null;
  defined_in_self: boolean;
  func_name: string | null;
  func_type: 'builtin' | 'custom_udf' | 'query' | 'unknown' | null;
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
