//! Table metadata: columns (with ids that survive renames), schema versions and data versions.
//!
//! Every mutation of a data-versioned table creates a new table version; schema changes also create a new schema
//! version (equal to the table version at which they happen). Columns are never removed from `columns`: dropping
//! a column records `schema_version_drop`, so older versions stay readable.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
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

mod opt_coltype_serde {
    use crate::types::ColumnType;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &Option<ColumnType>, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&t.as_ref().map(|t| t.to_dict()), s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<ColumnType>, D::Error> {
        let v = Option::<serde_json::Value>::deserialize(d)?;
        v.map(|v| ColumnType::from_dict(&v).map_err(|e| serde::de::Error::custom(e.to_string()))).transpose()
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

/// A column as visible in a schema version. `col_type` and `value_expr` override the column's `ColumnMd` from
/// this schema version on (`alter_column()`, `alter_computed_column()`), so that older versions keep the old ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaColumn {
    pub col_id: ColId,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none", with = "opt_coltype_serde")]
    pub col_type: Option<ColumnType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_expr: Option<Json>,
}

impl SchemaColumn {
    pub fn new(col_id: ColId, name: String) -> SchemaColumn {
        SchemaColumn { col_id, name, col_type: None, value_expr: None }
    }
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
    /// views: the version of each base (in `ViewMd::bases` order) at the time this version was created; a pinned
    /// version view (`v:3`) reads its bases at these versions
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub base_versions: Vec<i64>,
    /// views: the table version (table id, version) of the operation whose propagation created this version
    /// (base inserts, updates and deletes); reverting that version also reverts this one
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cascade_from: Option<(Uuid, i64)>,
}

impl VersionMd {
    pub fn new(version: i64, created_at: f64, schema_version: i64, change_type: &str, schema_change: Option<String>) -> Self {
        VersionMd {
            version,
            created_at,
            schema_version,
            user: None,
            change_type: change_type.to_string(),
            schema_change,
            row_count_stats: RowCountStats::default(),
            cascade_row_count_stats: RowCountStats::default(),
            base_versions: Vec::new(),
            cascade_from: None,
        }
    }
}

/// The kind of an index and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IndexKind {
    Btree {
        /// the hidden stored column with the indexed values (upstream `BtreeIndex.uses_value_col`, set for
        /// data-versioned tables); string values are truncated to `MAX_STRING_LEN`
        #[serde(default, skip_serializing_if = "Option::is_none")]
        val_col_id: Option<ColId>,
    },
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
        matches!(self.kind, IndexKind::Btree { .. })
    }

    pub fn val_col_id(&self) -> Option<ColId> {
        match &self.kind {
            IndexKind::Embedding { val_col_id, .. } => Some(*val_col_id),
            IndexKind::Btree { val_col_id } => *val_col_id,
        }
    }
}

/// The indexes of a table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Indexes {
    /// the live indexes
    pub list: Vec<IndexMd>,
    pub next_id: u32,
    /// dropped indexes with the schema version that dropped them (for `revert()`)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped: Vec<(i64, IndexMd)>,
}

impl Indexes {
    /// Undoes the index changes of schema version `version`: drops the indexes it added and restores those it dropped.
    pub fn revert(&mut self, version: i64) {
        self.list.retain(|i| i.schema_version_add != version);
        let (restored, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.dropped).into_iter().partition(|(v, _)| *v == version);
        self.dropped = kept;
        self.list.extend(restored.into_iter().map(|(_, i)| i));
        self.list.sort_by_key(|i| i.id);
    }
}

fn default_true() -> bool {
    true
}

/// What a view is defined over.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewMd {
    /// the bases of the view, direct base first, each live (None) or pinned to a version. A pure snapshot (no
    /// columns, no filter) has no rows of its own: its bases are the table path it pins.
    pub bases: Vec<(Uuid, Option<i64>)>,
    /// serialized filter predicate
    pub predicate: Option<Json>,
    pub is_snapshot: bool,
    /// false for a view defined by a query with a select list: only its own columns are visible
    #[serde(default = "default_true")]
    pub include_base_columns: bool,
    /// component views: the serialized iterator call (`functions::iterator::IteratorCall::to_json()`); its outputs
    /// (`pos` first) are the view's first columns
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterator: Option<Json>,
    /// the serialized sample clause (`exec::SampleClause`): a live view samples a fraction of its base rows (by a
    /// seeded hash of their keys); a snapshot's sample was applied when its rows were stored
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<Json>,
}

impl ViewMd {
    pub fn base_id(&self) -> Uuid {
        self.bases[0].0
    }

    pub fn base_version(&self) -> Option<i64> {
        self.bases[0].1
    }
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
    /// the schema versions, oldest first (persisted in the table's history file, not in `catalog.json`; entries are
    /// shared until changed, see `store::history`)
    #[serde(skip)]
    pub schema_versions: Vec<Arc<SchemaVersionMd>>,
    /// the table versions `0..=current_version`, oldest first (persisted like `schema_versions`)
    #[serde(skip)]
    pub versions: Vec<Arc<VersionMd>>,
    pub view: Option<ViewMd>,
    /// where the table's rows and history are in the store
    #[serde(default)]
    pub storage: crate::store::TableStorage,
    #[serde(default)]
    pub indexes: Indexes,
    /// advanced by every change of the table's rows (`Catalog::stage_data()`), also in other processes: an operation
    /// that computed rows from a snapshot commits them only if the data versions it read are still current
    #[serde(default)]
    pub data_version: u64,
}

/// A resolved column of a table at some version.
#[derive(Debug, Clone)]
pub struct ResolvedColumn {
    pub tbl_id: Uuid,
    /// the version the column's table is pinned to in the path it was resolved in (None: live)
    pub tbl_version: Option<i64>,
    pub name: String,
    /// the column's metadata as of the resolved schema version (with `alter_*()` overrides applied)
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
        // sorted by schema version
        let i = self.schema_versions.partition_point(|s| s.schema_version < schema_version);
        self.schema_versions
            .get(i)
            .filter(|s| s.schema_version == schema_version)
            .map(Arc::as_ref)
            .ok_or_else(|| Error::internal(format!("table {} has no schema version {schema_version}", self.id)))
    }

    /// The position of version `version` in `versions` (which holds consecutive versions).
    fn version_pos(&self, version: i64) -> Option<usize> {
        let first = self.versions.first()?.version;
        let i = usize::try_from(version.checked_sub(first)?).ok()?;
        match self.versions.get(i) {
            Some(v) if v.version == version => Some(i),
            _ => self.versions.iter().position(|v| v.version == version),
        }
    }

    /// The current (last) schema version, for changing it.
    pub fn last_schema_version_mut(&mut self) -> Option<&mut SchemaVersionMd> {
        self.schema_versions.last_mut().map(Arc::make_mut)
    }

    /// The metadata of version `version`, for changing it.
    pub fn version_md_mut(&mut self, version: i64) -> Option<&mut VersionMd> {
        let i = self.version_pos(version)?;
        Some(Arc::make_mut(&mut self.versions[i]))
    }

    pub fn current_schema(&self) -> Result<&SchemaVersionMd> {
        self.schema_version_md(self.current_schema_version)
    }

    pub fn version_md(&self, version: i64) -> Option<&VersionMd> {
        self.version_pos(version).map(|i| self.versions[i].as_ref())
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
                let mut md = self
                    .column(sc.col_id)
                    .ok_or_else(|| Error::internal(format!("schema references unknown column {}", sc.col_id)))?
                    .clone();
                if let Some(t) = &sc.col_type {
                    md.col_type = t.clone();
                }
                if let Some(e) = &sc.value_expr {
                    md.value_expr = Some(e.clone());
                }
                Ok(ResolvedColumn { tbl_id: self.id, tbl_version: None, name: sc.name.clone(), md })
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

    /// True for a component view (a view defined by an iterator).
    pub fn is_component_view(&self) -> bool {
        self.view.as_ref().is_some_and(|v| v.iterator.is_some())
    }

    /// The number of iterator output columns (`pos` and the outputs; they are the first columns of a component view).
    pub fn num_iterator_cols(&self) -> usize {
        self.view
            .as_ref()
            .and_then(|v| v.iterator.as_ref())
            .and_then(|i| i.get("outputs"))
            .and_then(|o| o.as_object())
            .map(|o| o.len())
            .unwrap_or(0)
    }

    /// True if column `col_id` is produced by the view's iterator.
    pub fn is_iterator_col(&self, col_id: ColId) -> bool {
        (col_id as usize) < self.num_iterator_cols()
    }

    /// A snapshot without columns or filter: it has no rows of its own and reads its (pinned) bases.
    pub fn is_pure_snapshot(&self) -> bool {
        self.view.as_ref().is_some_and(|v| v.is_snapshot && v.predicate.is_none() && v.sample.is_none())
            && self.columns.is_empty()
    }

    /// Ids of the visible stored columns at the current schema version, in schema order.
    pub fn current_column_ids(&self) -> Vec<ColId> {
        self.current_schema().map(|s| s.columns.iter().map(|c| c.col_id).collect()).unwrap_or_default()
    }

    pub fn index_by_name(&self, name: &str) -> Option<&IndexMd> {
        self.indexes.list.iter().find(|i| i.name == name)
    }

    /// The index whose hidden value column is `col_id`, if any.
    pub fn index_of_value_col(&self, col_id: ColId) -> Option<&IndexMd> {
        self.indexes.list.iter().find(|i| i.val_col_id() == Some(col_id))
    }
}
