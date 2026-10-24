//! Indexes: B-tree indexes (metadata only: every scan is a full scan) and embedding indexes, whose embeddings are
//! materialized in a hidden stored column of the table (`IndexKind::Embedding::val_col_id`, not part of any schema
//! version's visible columns) that inserts compute like any other stored computed column.
//!
//! Adding an embedding index evaluates the embedding of every row, so it is a plan/compute/commit operation
//! (`exec::compute::AddComputedPlan` with an `IndexSpec`); everything else here runs under the lock.

use uuid::Uuid;

use super::Catalog;
use super::path::fold_identifier;
use super::table::{ColId, ColumnMd, IndexKind, IndexMd, ResolvedColumn, TableMd};
use crate::error::{Error, Result, py_repr};
use crate::types::ColumnType;

/// An index about to be created together with its hidden value column (see `AddComputedPlan`).
#[derive(Debug, Clone)]
pub struct IndexSpec {
    pub name: Option<String>,
    pub col_tbl_id: Uuid,
    pub col_id: ColId,
    pub metric: String,
    pub precision: String,
    pub embeddings: Vec<(String, crate::value::Json)>,
}

/// Upstream `BtreeIndex.validate_column()`.
pub fn validate_btree_column(name: &str, md: &ColumnMd, is_iterator_col: bool) -> Result<()> {
    if !md.stored {
        crate::bail!(UnsupportedOperation, "Cannot create a B-tree index on unstored column {}.", py_repr(name));
    }
    let t = &md.col_type;
    if !((t.is_scalar() && !t.is_bool()) || t.is_media()) {
        crate::bail!(
            TypeMismatch,
            "Index on column {name}: B-tree index requires a non-boolean scalar type or a media type, got {}",
            t.repr()
        );
    }
    if t.is_media() {
        if is_iterator_col {
            crate::bail!(
                UnsupportedOperation,
                "Cannot create a B-tree index on column {}, which is produced by an iterator.",
                py_repr(name)
            );
        }
        if md.is_computed() {
            crate::bail!(UnsupportedOperation, "Cannot create a B-tree index on computed media column {}.", py_repr(name));
        }
    }
    Ok(())
}

/// Upstream `BtreeIndex.can_index()`.
pub fn can_btree_index(md: &ColumnMd) -> bool {
    validate_btree_column("", md, false).is_ok()
}

/// Upstream `generate_idx_name()`.
pub fn generate_idx_name(md: &TableMd) -> String {
    (0..).map(|i| format!("idx{i}")).find(|n| md.index_by_name(n).is_none()).unwrap_or_default()
}

/// A column argument of an index operation: a name, or a reference (table id, column id, name).
#[derive(Debug, Clone)]
pub enum ColumnArg {
    Name(String),
    Ref { tbl_id: Uuid, col_id: ColId, name: String },
}

impl Catalog {
    /// Upstream `_resolve_column_parameter()`: the visible column of `tbl_id` named or referenced by `col`.
    pub fn resolve_column_arg(&self, tbl_id: Uuid, col: &ColumnArg) -> Result<ResolvedColumn> {
        let visible = self.resolve_columns(tbl_id, None)?;
        match col {
            ColumnArg::Name(n) => {
                let folded = fold_identifier(n);
                visible
                    .into_iter()
                    .find(|c| c.name == folded)
                    .ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column: {n}"))
            }
            ColumnArg::Ref { tbl_id: ref_tbl, col_id, name } => {
                visible.into_iter().find(|c| c.tbl_id == *ref_tbl && c.md.id == *col_id).ok_or_else(|| {
                    let tbl_name = self.table(*ref_tbl).map(|t| t.name.clone()).unwrap_or_default();
                    crate::pxt_err!(ColumnNotFound, "Unknown column: {tbl_name}.{name}")
                })
            }
        }
    }

    /// Starts a new schema version for an index change (upstream records no schema change description for it).
    fn bump_version_for_index(&mut self, tbl_id: Uuid) -> Result<i64> {
        let v = self.bump_version(tbl_id, Some(String::new()))?;
        let md = self.table_mut(tbl_id)?;
        if let Some(vmd) = md.versions.iter_mut().find(|x| x.version == v) {
            vmd.schema_change = None;
            vmd.change_type = "data".to_string();
        }
        Ok(v)
    }

    /// Records a B-tree index on column `col_id` of `tbl_id` (no version change, no validation).
    pub fn push_btree_index(&mut self, tbl_id: Uuid, col_id: ColId, name: Option<String>) -> Result<()> {
        let md = self.table_mut(tbl_id)?;
        let name = name.unwrap_or_else(|| generate_idx_name(md));
        let id = md.indexes.next_id;
        md.indexes.next_id += 1;
        let schema_version_add = md.current_schema_version;
        md.indexes.list.push(IndexMd { id, name, col_tbl_id: tbl_id, col_id, kind: IndexKind::Btree, schema_version_add });
        Ok(())
    }

    /// The default B-tree index of a new column of a table with `has_default_idxs` (call after the column exists).
    pub fn add_default_index(&mut self, tbl_id: Uuid, col_id: ColId) -> Result<()> {
        let md = self.table(tbl_id)?;
        if !md.has_default_idxs {
            return Ok(());
        }
        let Some(col) = md.column(col_id) else { return Ok(()) };
        if can_btree_index(col) {
            self.push_btree_index(tbl_id, col_id, None)?;
        }
        Ok(())
    }

    /// Upstream checks of an index name (`Column.validate_name()`).
    pub fn validate_index_name(name: &str) -> Result<String> {
        super::tables::validate_column_name(name)?;
        Ok(fold_identifier(name))
    }

    /// `Table.add_btree_index()`
    pub fn add_btree_index(&mut self, tbl_id: Uuid, col: &ColumnArg, idx_name: Option<&str>, ignore: bool) -> Result<()> {
        self.check_mutable(tbl_id, None, "add an index to")?;
        let idx_name = idx_name.map(Self::validate_index_name).transpose()?;
        let col = self.resolve_column_arg(tbl_id, col)?;
        let md = self.table(tbl_id)?.clone();
        let existing_by_name = idx_name.as_deref().and_then(|n| md.index_by_name(n));
        if let Some(existing) = existing_by_name {
            if !existing.is_btree() {
                crate::bail!(
                    UnsupportedOperation,
                    "Index {} already exists, but is not a B-tree index.",
                    py_repr(&existing.name)
                );
            }
        }
        let has_btree = |c: &ResolvedColumn| md.indexes.list.iter().any(|i| i.is_btree() && i.col_tbl_id == c.tbl_id && i.col_id == c.md.id);
        if ignore
            && !md.has_default_idxs
            && existing_by_name.is_none_or(|i| i.col_tbl_id == col.tbl_id && i.col_id == col.md.id)
            && col.tbl_id == tbl_id
            && has_btree(&col)
        {
            return Ok(());
        }
        // upstream `validate_idxs()`
        if md.has_default_idxs {
            crate::bail!(
                UnsupportedOperation,
                "Cannot create an explicit B-tree index on a table with has_default_idxs=True; its eligible columns are indexed automatically."
            );
        }
        if col.tbl_id != tbl_id {
            crate::bail!(
                UnsupportedOperation,
                "Cannot create a B-tree index on column {}: it belongs to a base table. Add the index to the base table instead.",
                py_repr(&col.name)
            );
        }
        validate_btree_column(&col.name, &col.md, false)?;
        if has_btree(&col) {
            crate::bail!(IndexAlreadyExists, "A B-tree index already exists on column {}.", py_repr(&col.name));
        }
        if let Some(existing) = existing_by_name {
            let col_name = self.index_column_name(tbl_id, existing).unwrap_or_default();
            crate::bail!(IndexAlreadyExists, "Index {} already exists on column {}.", py_repr(&existing.name), py_repr(&col_name));
        }
        self.bump_version_for_index(tbl_id)?;
        self.push_btree_index(tbl_id, col.md.id, idx_name)?;
        self.commit()
    }

    /// The current name of an index's column (None if it is not visible).
    pub fn index_column_name(&self, tbl_id: Uuid, idx: &IndexMd) -> Option<String> {
        self.resolve_columns(tbl_id, None)
            .ok()?
            .into_iter()
            .find(|c| c.tbl_id == idx.col_tbl_id && c.md.id == idx.col_id)
            .map(|c| c.name)
    }

    /// `Table.drop_index()` / `Table.drop_embedding_index()` (`embedding_only`).
    pub fn drop_index(
        &mut self,
        tbl_id: Uuid,
        col: Option<&ColumnArg>,
        idx_name: Option<&str>,
        embedding_only: bool,
        ignore_missing: bool,
    ) -> Result<()> {
        self.check_mutable(tbl_id, None, "drop an index from")?;
        let md = self.table(tbl_id)?.clone();
        let idx = match (col, idx_name) {
            (_, Some(name)) => match md.index_by_name(&fold_identifier(name)) {
                Some(i) => i.clone(),
                None if ignore_missing => return Ok(()),
                None => crate::bail!(IndexNotFound, "Index {} does not exist", py_repr(name)),
            },
            (Some(col), None) => {
                let col = self.resolve_column_arg(tbl_id, col)?;
                if col.tbl_id != tbl_id {
                    let base_name = self.table(col.tbl_id).map(|t| t.name.clone()).unwrap_or_default();
                    crate::bail!(
                        UnsupportedOperation,
                        "Column {}: cannot drop index from column that belongs to base table {}",
                        py_repr(&col.name),
                        py_repr(&base_name)
                    );
                }
                let matches: Vec<&IndexMd> = md
                    .indexes
                    .list
                    .iter()
                    .filter(|i| i.col_id == col.md.id && i.col_tbl_id == col.tbl_id)
                    .filter(|i| !embedding_only || !i.is_btree())
                    .collect();
                match matches.as_slice() {
                    [] if ignore_missing => return Ok(()),
                    [] => crate::bail!(IndexNotFound, "Column {} does not have an index", py_repr(&col.name)),
                    [one] => (*one).clone(),
                    _ => crate::bail!(
                        UnsupportedOperation,
                        "Column {} has multiple indices; specify `idx_name` explicitly to drop one",
                        py_repr(&col.name)
                    ),
                }
            }
            (None, None) => crate::bail!(MissingRequired, "Exactly one of 'column' or 'idx_name' must be provided"),
        };
        if md.has_default_idxs && idx.is_btree() {
            crate::bail!(
                UnsupportedOperation,
                "Cannot drop B-tree index {} from a table with has_default_idxs=True",
                py_repr(&idx.name)
            );
        }
        let v = self.bump_version_for_index(tbl_id)?;
        self.remove_index(tbl_id, idx.id, v)?;
        self.commit()
    }

    /// Removes an index (and drops its hidden value column at schema version `version`). Does not commit.
    pub fn remove_index(&mut self, tbl_id: Uuid, idx_id: u32, version: i64) -> Result<()> {
        let md = self.table_mut(tbl_id)?;
        let Some(pos) = md.indexes.list.iter().position(|i| i.id == idx_id) else { return Ok(()) };
        let idx = md.indexes.list.remove(pos);
        if let Some(val_col) = idx.val_col_id() {
            if let Some(c) = md.column_mut(val_col) {
                c.schema_version_drop = Some(version);
            }
        }
        Ok(())
    }

    /// Removes the indexes on column `col_id` of `tbl_id` (when the column is dropped at `version`).
    pub fn remove_indexes_on_column(&mut self, tbl_id: Uuid, col_id: ColId, version: i64) -> Result<()> {
        let ids: Vec<u32> = self
            .table(tbl_id)?
            .indexes
            .list
            .iter()
            .filter(|i| i.col_tbl_id == tbl_id && i.col_id == col_id)
            .map(|i| i.id)
            .collect();
        for id in ids {
            self.remove_index(tbl_id, id, version)?;
        }
        Ok(())
    }

    /// Records an embedding index and its hidden value column (computed by `value_expr`), at a new schema version.
    /// Returns the id of the value column. Does not commit.
    pub fn push_embedding_index(
        &mut self,
        tbl_id: Uuid,
        spec: &IndexSpec,
        value_expr: crate::value::Json,
        val_type: ColumnType,
    ) -> Result<ColId> {
        let version = self.bump_version_for_index(tbl_id)?;
        let md = self.table_mut(tbl_id)?;
        let name = spec.name.clone().unwrap_or_else(|| generate_idx_name(md));
        let val_col_id = md.next_col_id;
        md.next_col_id += 1;
        md.columns.push(ColumnMd {
            id: val_col_id,
            col_type: val_type,
            is_pk: false,
            stored: true,
            value_expr: Some(value_expr),
            schema_version_add: version,
            schema_version_drop: None,
            media_validation: None,
            comment: None,
            custom_metadata: crate::value::Json::Null,
            destination: None,
        });
        let id = md.indexes.next_id;
        md.indexes.next_id += 1;
        md.indexes.list.push(IndexMd {
            id,
            name,
            col_tbl_id: spec.col_tbl_id,
            col_id: spec.col_id,
            kind: IndexKind::Embedding {
                metric: spec.metric.clone(),
                precision: spec.precision.clone(),
                embeddings: spec.embeddings.clone(),
                val_col_id,
            },
            schema_version_add: version,
        });
        Ok(val_col_id)
    }

    /// Upstream `TableVersion.get_idx()` for embedding indexes: the embedding index of a column, by name if given.
    pub fn embedding_index(&self, tbl_id: Uuid, col_tbl_id: Uuid, col_id: ColId, col_name: &str, idx_name: Option<&str>) -> Result<IndexMd> {
        let md = self.table(tbl_id)?;
        if md.is_snapshot() {
            crate::bail!(UnsupportedOperation, "Snapshot does not support indices");
        }
        let candidates: Vec<&IndexMd> = md
            .indexes
            .list
            .iter()
            .filter(|i| !i.is_btree() && i.col_tbl_id == col_tbl_id && i.col_id == col_id)
            .collect();
        if candidates.is_empty() {
            crate::bail!(IndexNotFound, "No embedding index found for column {}", py_repr(col_name));
        }
        match idx_name {
            None if candidates.len() > 1 => crate::bail!(
                UnsupportedOperation,
                "Column {} has multiple embedding indices; specify `idx_name` instead",
                py_repr(col_name)
            ),
            None => Ok(candidates[0].clone()),
            Some(n) => {
                let folded = fold_identifier(n);
                candidates
                    .into_iter()
                    .find(|i| i.name == folded)
                    .cloned()
                    .ok_or_else(|| crate::pxt_err!(IndexNotFound, "Index {} not found for column {}", py_repr(n), py_repr(col_name)))
            }
        }
    }

    /// The indexes of a table as (name, indexed column name, index); hidden-column indexes on invisible columns are
    /// left out.
    pub fn visible_indexes(&self, tbl_id: Uuid) -> Result<Vec<(String, IndexMd)>> {
        let md = self.table(tbl_id)?;
        Ok(md
            .indexes
            .list
            .iter()
            .filter_map(|i| self.index_column_name(tbl_id, i).map(|n| (n, i.clone())))
            .collect())
    }
}

/// An error for an index operation on a table kind that does not support it yet.
pub fn views_unsupported(what: &str) -> Error {
    Error::not_implemented(format!("{what} on a view"))
}
