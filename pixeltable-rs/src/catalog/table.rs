//! Table metadata: columns (with ids that survive renames), schema versions and data versions.
//!
//! Every mutation of a data-versioned table creates a new table version; schema changes also create a new schema
//! version (equal to the table version at which they happen). Columns are never removed from `columns`: dropping
//! a column records `schema_version_drop`, so older versions stay readable.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result, py_repr};
use crate::types::ColumnType;
use crate::value::Json;

pub type ColId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TableKind {
    Table,
    View,
    Snapshot,
}

impl TableKind {
    /// The name used in messages (`_display_name()`).
    pub fn display_name(self) -> &'static str {
        match self {
            TableKind::Table => "table",
            TableKind::View => "view",
            TableKind::Snapshot => "snapshot",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaValidation {
    OnRead,
    OnWrite,
}

impl MediaValidation {
    pub fn parse(s: &str, error_prefix: &str) -> Result<MediaValidation> {
        match s.to_ascii_uppercase().as_str() {
            "ON_READ" => Ok(MediaValidation::OnRead),
            "ON_WRITE" => Ok(MediaValidation::OnWrite),
            _ => crate::bail!(InvalidArgument, "{error_prefix} must be one of: ['on_read', 'on_write']"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            MediaValidation::OnRead => "on_read",
            MediaValidation::OnWrite => "on_write",
        }
    }
}

mod coltype_serde {
    use crate::types::ColumnType;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &ColumnType, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&t.to_dict(), s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ColumnType, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        ColumnType::from_dict(&v).map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMd {
    pub id: ColId,
    #[serde(with = "coltype_serde")]
    pub col_type: ColumnType,
    pub is_pk: bool,
    pub stored: bool,
    /// serialized `Expr` (see `expr::serde`) for computed columns
    pub value_expr: Option<Json>,
    pub schema_version_add: i64,
    pub schema_version_drop: Option<i64>,
    pub media_validation: Option<MediaValidation>,
    pub comment: Option<String>,
    pub custom_metadata: Json,
    pub destination: Option<String>,
}

impl ColumnMd {
    pub fn is_computed(&self) -> bool {
        self.value_expr.is_some()
    }
}

/// A column as visible in a schema version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaColumn {
    pub col_id: ColId,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersionMd {
    pub schema_version: i64,
    pub preceding_schema_version: Option<i64>,
    /// visible columns, in schema order
    pub columns: Vec<SchemaColumn>,
    pub comment: Option<String>,
    pub custom_metadata: Json,
    pub media_validation: MediaValidation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowCountStats {
    pub ins_rows: i64,
    pub del_rows: i64,
    pub upd_rows: i64,
    pub num_excs: i64,
    pub computed_values: i64,
}

impl RowCountStats {
    pub fn num_rows(&self) -> i64 {
        self.ins_rows + self.del_rows + self.upd_rows
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMd {
    pub version: i64,
    pub created_at: f64,
    pub schema_version: i64,
    pub user: Option<String>,
    /// "data" or "schema"
    pub change_type: String,
    pub schema_change: Option<String>,
    pub row_count_stats: RowCountStats,
    pub cascade_row_count_stats: RowCountStats,
}

/// The kind of an index and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IndexKind {
    Btree,
    Embedding {
        /// `cosine`, `ip` or `l2`
        metric: String,
        /// `fp16` or `fp32`
        precision: String,
        /// (type name, e.g. `STRING`, serialized unary embedding function `Function.as_dict()`), per supported
        /// query modality
        embeddings: Vec<(String, Json)>,
        /// the hidden stored column with the embeddings of the indexed column's values
        val_col_id: ColId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMd {
    pub id: u32,
    pub name: String,
    /// the indexed column: a column of this table or of one of its bases
    pub col_tbl_id: Uuid,
    pub col_id: ColId,
    pub kind: IndexKind,
    pub schema_version_add: i64,
}

impl IndexMd {
    pub fn is_btree(&self) -> bool {
        matches!(self.kind, IndexKind::Btree)
    }

    pub fn val_col_id(&self) -> Option<ColId> {
        match &self.kind {
            IndexKind::Embedding { val_col_id, .. } => Some(*val_col_id),
            IndexKind::Btree => None,
        }
    }
}

/// The live indexes of a table (dropped indexes are removed).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Indexes {
    pub list: Vec<IndexMd>,
    pub next_id: u32,
}

/// What a view is defined over (filled in by the views work; tables have none).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewMd {
    pub base_id: Uuid,
    /// None: the view tracks the live base; Some(v): pinned to base version v (snapshots)
    pub base_version: Option<i64>,
    /// serialized filter predicate
    pub predicate: Option<Json>,
    pub is_snapshot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMd {
    pub id: Uuid,
    pub dir_id: Uuid,
    pub name: String,
    pub kind: TableKind,
    pub is_data_versioned: bool,
    pub has_default_idxs: bool,
    pub created_at: f64,
    pub current_version: i64,
    pub current_schema_version: i64,
    pub next_col_id: ColId,
    pub next_rowid: i64,
    /// every column ever created, in creation order
    pub columns: Vec<ColumnMd>,
    pub schema_versions: Vec<SchemaVersionMd>,
    pub versions: Vec<VersionMd>,
    pub view: Option<ViewMd>,
    /// file name (in the store) of the row data as of `current_version`
    pub data_file: Option<String>,
    #[serde(default)]
    pub indexes: Indexes,
}

/// A resolved column of a table at some version.
#[derive(Debug, Clone)]
pub struct ResolvedColumn {
    pub tbl_id: Uuid,
    pub name: String,
    pub md: ColumnMd,
}

impl TableMd {
    pub fn column(&self, id: ColId) -> Option<&ColumnMd> {
        self.columns.iter().find(|c| c.id == id)
    }

    pub fn column_mut(&mut self, id: ColId) -> Option<&mut ColumnMd> {
        self.columns.iter_mut().find(|c| c.id == id)
    }

    pub fn schema_version_md(&self, schema_version: i64) -> Result<&SchemaVersionMd> {
        self.schema_versions
            .iter()
            .find(|s| s.schema_version == schema_version)
            .ok_or_else(|| Error::internal(format!("table {} has no schema version {schema_version}", self.id)))
    }

    pub fn current_schema(&self) -> Result<&SchemaVersionMd> {
        self.schema_version_md(self.current_schema_version)
    }

    pub fn version_md(&self, version: i64) -> Option<&VersionMd> {
        self.versions.iter().find(|v| v.version == version)
    }

    /// The schema version in effect at table version `version` (None: current).
    pub fn schema_version_at(&self, version: Option<i64>) -> Result<i64> {
        match version {
            None => Ok(self.current_schema_version),
            Some(v) => self
                .version_md(v)
                .map(|m| m.schema_version)
                .ok_or_else(|| crate::pxt_err!(InvalidArgument, "Table {} has no version {v}", py_repr(&self.name))),
        }
    }

    /// The visible columns at table version `version` (None: current), in schema order.
    pub fn columns_at(&self, version: Option<i64>) -> Result<Vec<ResolvedColumn>> {
        let sv = self.schema_version_md(self.schema_version_at(version)?)?;
        sv.columns
            .iter()
            .map(|sc| {
                let md = self
                    .column(sc.col_id)
                    .ok_or_else(|| Error::internal(format!("schema references unknown column {}", sc.col_id)))?;
                Ok(ResolvedColumn { tbl_id: self.id, name: sc.name.clone(), md: md.clone() })
            })
            .collect()
    }

    /// The visible column with the given (unfolded or folded) name.
    pub fn column_by_name(&self, name: &str, version: Option<i64>) -> Result<Option<ResolvedColumn>> {
        let folded = crate::catalog::path::fold_identifier(name);
        Ok(self.columns_at(version)?.into_iter().find(|c| c.name == folded))
    }

    pub fn current_comment(&self) -> Option<String> {
        self.current_schema().ok().and_then(|s| s.comment.clone())
    }

    pub fn total_error_count(&self) -> i64 {
        self.versions.iter().map(|v| v.row_count_stats.num_excs + v.cascade_row_count_stats.num_excs).sum()
    }

    pub fn is_view(&self) -> bool {
        self.view.is_some()
    }

    pub fn is_snapshot(&self) -> bool {
        self.view.as_ref().is_some_and(|v| v.is_snapshot)
    }

    pub fn index_by_name(&self, name: &str) -> Option<&IndexMd> {
        self.indexes.list.iter().find(|i| i.name == name)
    }

    /// The index whose hidden value column is `col_id`, if any.
    pub fn index_of_value_col(&self, col_id: ColId) -> Option<&IndexMd> {
        self.indexes.list.iter().find(|i| i.val_col_id() == Some(col_id))
    }
}
