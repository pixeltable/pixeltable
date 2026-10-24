//! Version operations: `revert()` and the version history (`get_versions()`).

use std::collections::HashMap;
use uuid::Uuid;

use super::Catalog;
use super::table::*;
use crate::error::{Error, Result};

/// One entry of `get_versions()`.
#[derive(Debug, Clone)]
pub struct VersionInfo {
    pub version: i64,
    pub created_at: f64,
    pub user: Option<String>,
    /// "schema" if the version changed the visible columns, else "data"
    pub change_type: String,
    pub inserts: i64,
    pub updates: i64,
    pub deletes: i64,
    pub errors: i64,
    pub schema_change: Option<String>,
}

/// A visible column of a schema version, as the history compares them.
#[derive(Debug, Clone, PartialEq)]
struct HistCol {
    name: String,
    col_type: crate::types::ColumnType,
    value_expr: Option<crate::value::Json>,
}

fn schema_cols(md: &TableMd, schema_version: i64) -> Result<Vec<(ColId, HistCol)>> {
    let sv = md.schema_version_md(schema_version)?;
    sv.columns
        .iter()
        .map(|sc| {
            let c = md.column(sc.col_id).ok_or_else(|| Error::internal("schema references an unknown column"))?;
            Ok((
                sc.col_id,
                HistCol {
                    name: sc.name.clone(),
                    col_type: sc.col_type.clone().unwrap_or_else(|| c.col_type.clone()),
                    value_expr: sc.value_expr.clone().or_else(|| c.value_expr.clone()),
                },
            ))
        })
        .collect()
}

/// Upstream `MetadataUtils._diff_md()`: what changed between two schema versions ('' if nothing visible did).
fn diff_schema(old: Option<&[(ColId, HistCol)]>, new: &[(ColId, HistCol)]) -> String {
    let Some(old) = old else { return "Initial Version".to_string() };
    let old_by_id: HashMap<ColId, &HistCol> = old.iter().map(|(id, c)| (*id, c)).collect();
    let new_by_id: HashMap<ColId, &HistCol> = new.iter().map(|(id, c)| (*id, c)).collect();
    let mut added = Vec::new();
    let mut altered = Vec::new();
    for (id, c) in new {
        match old_by_id.get(id) {
            None => added.push(c.name.clone()),
            Some(o) => {
                let mut diff = Vec::new();
                if o.name != c.name {
                    diff.push(format!("renamed to {}", c.name));
                }
                if o.col_type != c.col_type {
                    diff.push(format!("type changed to {}", c.col_type.repr()));
                }
                if o.value_expr != c.value_expr {
                    diff.push("value expression changed".to_string());
                }
                if !diff.is_empty() {
                    altered.push(format!("{} ({})", o.name, diff.join(", ")));
                }
            }
        }
    }
    let dropped: Vec<String> = old.iter().filter(|(id, _)| !new_by_id.contains_key(id)).map(|(_, c)| c.name.clone()).collect();
    let mut parts = Vec::new();
    if !added.is_empty() {
        parts.push(format!("Added: {}", added.join(", ")));
    }
    if !altered.is_empty() {
        parts.push(format!("Altered: {}", altered.join(", ")));
    }
    if !dropped.is_empty() {
        parts.push(format!("Dropped: {}", dropped.join(", ")));
    }
    parts.join(", ")
}

impl Catalog {
    /// The most recent `n` versions of a table, newest first (upstream `get_versions()`).
    pub fn version_history(&self, tbl_id: Uuid, n: Option<usize>) -> Result<Vec<VersionInfo>> {
        let md = self.table(tbl_id)?;
        let mut versions: Vec<&VersionMd> = md.versions.iter().collect();
        versions.sort_by_key(|v| std::cmp::Reverse(v.version));
        let mut out = Vec::new();
        for (i, v) in versions.iter().enumerate().take(n.unwrap_or(usize::MAX)) {
            let schema_change = if v.version == 0 || v.schema_version == v.version {
                let new = schema_cols(md, v.schema_version)?;
                let old = match versions.get(i + 1) {
                    Some(prev) if v.version != 0 => Some(schema_cols(md, prev.schema_version)?),
                    _ => None,
                };
                let d = diff_schema(old.as_deref(), &new);
                if d.is_empty() { None } else { Some(d) }
            } else {
                None
            };
            let total = super::super::exec::propagate::add_stats(&v.row_count_stats, &v.cascade_row_count_stats);
            out.push(VersionInfo {
                version: v.version,
                created_at: v.created_at,
                user: v.user.clone(),
                change_type: if schema_change.is_some() { "schema".into() } else { "data".into() },
                inserts: total.ins_rows,
                updates: total.upd_rows,
                deletes: total.del_rows,
                errors: total.num_excs,
                schema_change,
            });
        }
        Ok(out)
    }

    /// `revert()`: undoes the current version of a table, and the versions it created in its views. Commits.
    pub fn revert(&mut self, tbl_id: Uuid, version: Option<i64>) -> Result<()> {
        self.check_mutable(tbl_id, version, "revert")?;
        let md = self.table(tbl_id)?.clone();
        if !md.is_data_versioned {
            crate::bail!(UnsupportedOperation, "Revert is supported on data-versioned tables only");
        }
        if md.current_version == 0 {
            crate::bail!(UnsupportedOperation, "Cannot revert version 0");
        }
        let v = md.current_version;
        // snapshots (and views over snapshots) that pin the current version
        let mut pinning: Vec<String> = self
            .all_tables()
            .filter(|t| t.view.as_ref().is_some_and(|vm| vm.bases.contains(&(tbl_id, Some(v)))))
            .map(|t| t.name.clone())
            .collect();
        if !pinning.is_empty() {
            pinning.sort();
            crate::bail!(
                UnsupportedOperation,
                "Current version is needed for {} snapshot{}: ({})",
                pinning.len(),
                if pinning.len() > 1 { "s" } else { "" },
                pinning.join(", ")
            );
        }
        self.revert_version(tbl_id)?;
        self.revert_cascaded(tbl_id, (tbl_id, v))?;
        self.commit()
    }

    /// Reverts the views of `tbl_id` (recursively) whose current version was created by `root`.
    fn revert_cascaded(&mut self, tbl_id: Uuid, root: (Uuid, i64)) -> Result<()> {
        for view in self.mutable_view_ids(tbl_id) {
            let md = self.table(view)?;
            let caused = md.version_md(md.current_version).is_some_and(|vm| vm.cascade_from == Some(root));
            if caused && md.current_version > 0 {
                self.revert_version(view)?;
            }
            self.revert_cascaded(view, root)?;
        }
        Ok(())
    }

    /// Undoes the current version of one table: its row changes, and its schema change if it made one.
    fn revert_version(&mut self, tbl_id: Uuid) -> Result<()> {
        let v = self.table(tbl_id)?.current_version;
        let mut data = self.data(tbl_id)?.as_ref().clone();
        data.revert(v);
        let md = self.table_mut(tbl_id)?;
        if md.current_schema_version == v {
            let added: Vec<ColId> = md.columns.iter().filter(|c| c.schema_version_add == v).map(|c| c.id).collect();
            if let Some(min) = added.iter().min() {
                md.next_col_id = *min;
            }
            md.columns.retain(|c| c.schema_version_add != v);
            for c in md.columns.iter_mut() {
                if c.schema_version_drop == Some(v) {
                    c.schema_version_drop = None;
                }
            }
            let preceding = md.current_schema()?.preceding_schema_version.unwrap_or(0);
            md.schema_versions.retain(|s| s.schema_version != v);
            md.current_schema_version = preceding;
        }
        md.versions.retain(|vm| vm.version != v);
        md.current_version = v - 1;
        self.defer_delete_media_version(tbl_id, v);
        self.stage_data(tbl_id, data)
    }
}
