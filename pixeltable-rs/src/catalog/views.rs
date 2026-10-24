//! Views and snapshots (minimal: a filter over a base, at the live version or pinned to a base version).
//!
//! Extension points for the views work: additional (computed) columns, iterator views, propagation of base
//! updates to stored view columns, and snapshots that materialize their own columns.

use std::sync::Arc;
use uuid::Uuid;

use super::dirs::{Creating, Expected};
use super::path::Path;
use super::table::*;
use super::{Catalog, Entry, IfExists, now};
use crate::error::{Error, Result};
use crate::expr::ExprRef;
use crate::value::Json;

/// What a query over a table reads: the table whose rows are scanned, at which version, with which filters.
#[derive(Debug, Clone)]
pub struct ScanSource {
    pub base_tbl_id: Uuid,
    /// version of the scanned table (None: current)
    pub base_version: Option<i64>,
    pub predicates: Vec<Json>,
}

impl Catalog {
    #[allow(clippy::too_many_arguments)]
    pub fn create_view(
        &mut self,
        path: &Path,
        base_id: Uuid,
        base_version: Option<i64>,
        predicate: Option<ExprRef>,
        is_snapshot: bool,
        comment: Option<String>,
        custom_metadata: Json,
        media_validation: MediaValidation,
        if_exists: IfExists,
    ) -> Result<(Uuid, bool)> {
        if let Some(p) = &predicate {
            crate::functions::udf::validate_storable(p, "A view's filter")?;
        }
        let creating = if is_snapshot { Creating::Snapshot } else { Creating::View };
        // a replacement must not be one of the view's own ancestors
        if matches!(if_exists, IfExists::Replace | IfExists::ReplaceForce) {
            if let Ok((_, Some(Entry::Table(existing)))) = self.prepare_add(path, false) {
                if self.base_chain(base_id).contains(&existing) {
                    crate::bail!(
                        UnsupportedOperation,
                        "Cannot use if_exists='replace' with the same name as one of the view's own ancestors."
                    );
                }
            }
        }
        if let Some(Entry::Table(id)) = self.handle_path_collision(path, creating, if_exists)? {
            let existing_base = self.table(id)?.view.as_ref().map(|v| v.base_id);
            if existing_base != Some(base_id) {
                crate::bail!(
                    PathAlreadyExists,
                    "Path {} already exists as a {} with a different base table",
                    path.repr(),
                    if is_snapshot { "snapshot" } else { "view" }
                );
            }
            return Ok((id, false));
        }
        let Some(Entry::Dir(dir_id)) = self.get_object(&path.parent(), Expected::Dir, false, true)? else {
            return Err(Error::internal("create_view(): parent is not a directory"));
        };
        let base = self.table(base_id)?.clone();
        let pinned_version = if is_snapshot { Some(base_version.unwrap_or(base.current_version)) } else { base_version };
        let ts = now();
        let id = Uuid::new_v4();
        let md = TableMd {
            id,
            dir_id,
            name: path.name().to_string(),
            kind: if is_snapshot { TableKind::Snapshot } else { TableKind::View },
            is_data_versioned: true,
            has_default_idxs: false,
            created_at: ts,
            current_version: 0,
            current_schema_version: 0,
            next_col_id: 0,
            next_rowid: 0,
            columns: Vec::new(),
            schema_versions: vec![SchemaVersionMd {
                schema_version: 0,
                preceding_schema_version: None,
                columns: Vec::new(),
                comment,
                custom_metadata,
                media_validation,
            }],
            versions: vec![VersionMd {
                version: 0,
                created_at: ts,
                schema_version: 0,
                user: None,
                change_type: "schema".to_string(),
                schema_change: None,
                row_count_stats: RowCountStats::default(),
                cascade_row_count_stats: RowCountStats::default(),
            }],
            view: Some(ViewMd {
                base_id,
                base_version: pinned_version,
                predicate: predicate.as_ref().map(|p| crate::expr::serde::to_json(p)).transpose()?,
                is_snapshot,
            }),
            data_file: None,
            indexes: Default::default(),
        };
        self.tables.insert(id, Arc::new(md));
        self.commit()?;
        Ok((id, true))
    }

    /// The table and all its bases, starting with the table itself.
    pub fn base_chain(&self, tbl_id: Uuid) -> Vec<Uuid> {
        let mut out = vec![tbl_id];
        let mut cur = tbl_id;
        while let Some(v) = self.tables.get(&cur).and_then(|t| t.view.clone()) {
            out.push(v.base_id);
            cur = v.base_id;
        }
        out
    }

    /// The visible columns of a table or view (a view sees its base's columns, then its own).
    pub fn resolve_columns(&self, tbl_id: Uuid, version: Option<i64>) -> Result<Vec<ResolvedColumn>> {
        let md = self.table(tbl_id)?;
        let mut cols = Vec::new();
        if let Some(v) = &md.view {
            cols.extend(self.resolve_columns(v.base_id, v.base_version)?);
        }
        cols.extend(md.columns_at(version)?);
        Ok(cols)
    }

    pub fn resolve_column_by_name(&self, tbl_id: Uuid, version: Option<i64>, name: &str) -> Result<Option<ResolvedColumn>> {
        let folded = super::path::fold_identifier(name);
        Ok(self.resolve_columns(tbl_id, version)?.into_iter().find(|c| c.name == folded))
    }

    /// What a query over `tbl_id` (at `version`) scans.
    pub fn scan_source(&self, tbl_id: Uuid, version: Option<i64>) -> Result<ScanSource> {
        let md = self.table(tbl_id)?;
        match &md.view {
            None => Ok(ScanSource { base_tbl_id: tbl_id, base_version: version, predicates: Vec::new() }),
            Some(v) => {
                let mut src = self.scan_source(v.base_id, v.base_version)?;
                if let Some(p) = &v.predicate {
                    src.predicates.push(p.clone());
                }
                Ok(src)
            }
        }
    }

    /// The base path string of a view for listings (`dir1/t1`, or `dir1/t1:0` for a pinned base).
    pub fn base_display(&self, tbl_id: Uuid) -> Result<Option<String>> {
        let md = self.table(tbl_id)?;
        let Some(v) = &md.view else { return Ok(None) };
        let base_path = self.table_path(v.base_id)?;
        Ok(Some(match v.base_version {
            Some(ver) => format!("{base_path}:{ver}"),
            None => base_path.to_string(),
        }))
    }

    pub fn list_views(&self, tbl_id: Uuid, recursive: bool) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for v in self.view_ids(tbl_id) {
            out.push(self.table_path(v)?.to_string());
            if recursive {
                out.extend(self.list_views(v, true)?);
            }
        }
        Ok(out)
    }
}
