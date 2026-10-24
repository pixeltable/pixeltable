//! Views and snapshots, and table paths.
//!
//! A view is defined over a base (a table, a view, or a pinned version of either) by an optional filter, optional
//! columns of its own, or a select list (then only its own columns are visible). It stores its own rows: one per
//! base row that passes the filter, keyed by the base row's key, holding the view's own stored columns, with their
//! own version ranges. Base inserts, updates and deletes propagate to the live views (`exec::propagate`).
//!
//! A table path is the sequence of tables a handle reads, leaf first, each live or pinned to a version
//! (`Catalog::tbl_path()`): a base table is a path of one; a live view adds its bases; a version-pinned view
//! (`v:3`) reads its bases at the versions recorded for that view version; a pure snapshot (no columns, no
//! filter) has no rows of its own and is just the pinned path of its bases.

use std::sync::Arc;
use uuid::Uuid;

use super::dirs::{Creating, Expected};
use super::path::{Path, fold_identifier};
use super::table::*;
use super::tables::{ColumnSpec, validate_column_spec};
use super::{Catalog, Entry, IfExists, now};
use crate::error::{Error, Result, py_repr};
use crate::expr::ExprRef;
use crate::value::Json;

/// One table of a table path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathElem {
    pub tbl_id: Uuid,
    /// the version read (None: live, i.e. the current version)
    pub pinned: Option<i64>,
}

/// The definition of a view to create (from `create_view()`'s arguments).
#[derive(Debug, Clone)]
pub struct ViewDef {
    /// the base handle: a table id and its pinned version
    pub base_id: Uuid,
    pub base_version: Option<i64>,
    pub predicate: Option<ExprRef>,
    /// the select list of the base query (then only the view's own columns are visible)
    pub select: Option<Vec<(ExprRef, Option<String>)>>,
    pub additional_columns: Vec<ColumnSpec>,
    pub is_snapshot: bool,
    pub comment: Option<String>,
    pub custom_metadata: Json,
    pub media_validation: MediaValidation,
    /// component views: the iterator call that produces the view's rows from each base row
    pub iterator: Option<std::sync::Arc<crate::functions::iterator::IteratorCall>>,
    /// the sample clause of the base query
    pub sample: Option<crate::exec::SampleClause>,
    /// a default B-tree index on each eligible column of the view (not on its bases' columns)
    pub has_default_idxs: bool,
}

impl Catalog {
    /// The table path a handle on `tbl_id` (pinned to `version`) reads, leaf first.
    pub fn tbl_path(&self, tbl_id: Uuid, version: Option<i64>) -> Result<Vec<PathElem>> {
        let md = self.table(tbl_id)?;
        let Some(view) = &md.view else { return Ok(vec![PathElem { tbl_id, pinned: version }]) };
        if md.is_pure_snapshot() {
            return Ok(view.bases.iter().map(|(id, pin)| PathElem { tbl_id: *id, pinned: *pin }).collect());
        }
        let recorded = match version {
            Some(v) => Some(
                md.version_md(v)
                    .ok_or_else(|| crate::pxt_err!(InvalidArgument, "Table {} has no version {v}", py_repr(&md.name)))?
                    .base_versions
                    .clone(),
            ),
            None => None,
        };
        let mut out = vec![PathElem { tbl_id, pinned: version }];
        for (i, (id, pin)) in view.bases.iter().enumerate() {
            let pinned = match (pin, &recorded) {
                (Some(p), _) => Some(*p),
                (None, Some(rec)) => Some(
                    *rec.get(i)
                        .ok_or_else(|| Error::internal(format!("version {version:?} of {tbl_id} has no base versions")))?,
                ),
                (None, None) => None,
            };
            out.push(PathElem { tbl_id: *id, pinned });
        }
        Ok(out)
    }

    /// The number of iterator positions in the row keys of `tbl_id` (one per component view in its path).
    pub fn num_pos(&self, tbl_id: Uuid) -> usize {
        self.tbl_path(tbl_id, None)
            .map(|p| p.iter().filter(|e| self.tables.get(&e.tbl_id).is_some_and(|t| t.is_component_view())).count())
            .unwrap_or(0)
    }

    /// The path with every live table pinned to its current version.
    pub fn pin_path(&self, path: &[PathElem]) -> Result<Vec<PathElem>> {
        path.iter()
            .map(|e| {
                Ok(PathElem { tbl_id: e.tbl_id, pinned: Some(e.pinned.unwrap_or(self.table(e.tbl_id)?.current_version)) })
            })
            .collect()
    }

    /// The versions of the bases of view `view_id` right now (in `ViewMd::bases` order).
    pub fn current_base_versions(&self, view_id: Uuid) -> Result<Vec<i64>> {
        let md = self.table(view_id)?;
        let Some(v) = &md.view else { return Ok(Vec::new()) };
        v.bases
            .iter()
            .map(|(id, pin)| Ok(pin.unwrap_or(self.table(*id)?.current_version)))
            .collect()
    }

    /// The table and all its bases, starting with the table itself (the tables of its live path).
    pub fn base_chain(&self, tbl_id: Uuid) -> Vec<Uuid> {
        let mut out = vec![tbl_id];
        if let Ok(path) = self.tbl_path(tbl_id, None) {
            for e in path {
                if !out.contains(&e.tbl_id) {
                    out.push(e.tbl_id);
                }
            }
        }
        out
    }

    /// The columns visible by name through a path: the leaf's own columns, then (unless the leaf is defined by a
    /// select list) the visible columns of its base that it doesn't shadow.
    pub fn path_visible_columns(&self, path: &[PathElem]) -> Result<Vec<ResolvedColumn>> {
        let Some(elem) = path.first() else { return Ok(Vec::new()) };
        let md = self.table(elem.tbl_id)?;
        let mut cols = md.columns_at(elem.pinned)?;
        for c in cols.iter_mut() {
            c.tbl_version = elem.pinned;
        }
        let include_base = md.view.as_ref().is_none_or(|v| v.include_base_columns);
        if path.len() > 1 && include_base {
            let own: std::collections::HashSet<String> = cols.iter().map(|c| c.name.clone()).collect();
            cols.extend(self.path_visible_columns(&path[1..])?.into_iter().filter(|c| !own.contains(&c.name)));
        }
        Ok(cols)
    }

    /// Every column physically present in a path (visible by name or not).
    pub fn path_columns(&self, path: &[PathElem]) -> Result<Vec<ResolvedColumn>> {
        let mut out = Vec::new();
        for e in path {
            let mut cols = self.table(e.tbl_id)?.columns_at(e.pinned)?;
            for c in cols.iter_mut() {
                c.tbl_version = e.pinned;
            }
            out.extend(cols);
        }
        Ok(out)
    }

    /// The visible columns of a table or view (own columns first, then the base columns it doesn't shadow).
    pub fn resolve_columns(&self, tbl_id: Uuid, version: Option<i64>) -> Result<Vec<ResolvedColumn>> {
        self.path_visible_columns(&self.tbl_path(tbl_id, version)?)
    }

    pub fn resolve_column_by_name(&self, tbl_id: Uuid, version: Option<i64>, name: &str) -> Result<Option<ResolvedColumn>> {
        let folded = fold_identifier(name);
        Ok(self.resolve_columns(tbl_id, version)?.into_iter().find(|c| c.name == folded))
    }

    /// True if every column reference of `e` is a column of `path`, pinned as the path pins its table.
    pub fn is_bound_by(&self, e: &crate::expr::Expr, path: &[PathElem]) -> Result<bool> {
        let cols = self.path_columns(path)?;
        Ok(e.column_refs().iter().all(|r| {
            cols.iter().any(|c| c.tbl_id == r.tbl_id && c.md.id == r.col_id && c.tbl_version == r.tbl_version)
        }))
    }

    /// The name of column `col_id` of `tbl_id` as of version `version` of the table (None: current); the latest name
    /// if the column isn't visible at that version.
    pub fn column_name_at(&self, tbl_id: Uuid, col_id: ColId, version: Option<i64>) -> Option<String> {
        let md = self.tables.get(&tbl_id)?;
        let sv = md.schema_version_at(version).ok()?;
        let find = |s: &SchemaVersionMd| s.columns.iter().find(|c| c.col_id == col_id).map(|c| c.name.clone());
        md.schema_version_md(sv)
            .ok()
            .and_then(find)
            .or_else(|| md.schema_versions.iter().rev().find_map(|s| find(s)))
    }

    /// `e` with its column references named as the tables of `path` name them at the versions it reads (tables
    /// not in the path: their current names); for displaying stored expressions after renames.
    pub fn with_column_names(&self, e: &ExprRef, path: &[PathElem]) -> ExprRef {
        e.map_column_refs(&|c| {
            let pin = path.iter().find(|p| p.tbl_id == c.tbl_id).and_then(|p| p.pinned);
            self.column_name_at(c.tbl_id, c.col_id, pin).map(|name| crate::expr::ColumnRef { name, ..c.clone() })
        })
    }

    /// The base path string of a view for listings and metadata (`dir1/t1`, or `dir1/t1:0` for a pinned base).
    pub fn base_display(&self, tbl_id: Uuid) -> Result<Option<String>> {
        let md = self.table(tbl_id)?;
        let Some(v) = &md.view else { return Ok(None) };
        let base_path = self.table_path(v.base_id())?;
        Ok(Some(match v.base_version() {
            Some(ver) => format!("{base_path}:{ver}"),
            None => base_path.to_string(),
        }))
    }

    /// The base string of a version-pinned handle `tbl_id:version` (its base at the recorded version).
    pub fn pinned_base_display(&self, tbl_id: Uuid, version: i64) -> Result<Option<String>> {
        let md = self.table(tbl_id)?;
        if md.view.is_none() || md.is_pure_snapshot() {
            return self.base_display(tbl_id);
        }
        let path = self.tbl_path(tbl_id, Some(version))?;
        let Some(base) = path.get(1) else { return Ok(None) };
        let base_path = self.table_path(base.tbl_id)?;
        Ok(Some(match base.pinned {
            Some(ver) => format!("{base_path}:{ver}"),
            None => base_path.to_string(),
        }))
    }

    pub fn list_views(&self, tbl_id: Uuid, recursive: bool) -> Result<Vec<String>> {
        self.table(tbl_id)?;
        let mut out = Vec::new();
        for v in self.view_ids(tbl_id) {
            out.push(self.table_path(v)?.to_string());
            if recursive {
                out.extend(self.list_views(v, true)?);
            }
        }
        Ok(out)
    }

    /// Ids of the live views over `tbl_id` (the views its data changes propagate to), in creation order.
    pub fn mutable_view_ids(&self, tbl_id: Uuid) -> Vec<Uuid> {
        self.view_ids(tbl_id)
            .into_iter()
            .filter(|v| self.tables.get(v).and_then(|t| t.view.as_ref()).is_some_and(|vm| vm.base_version().is_none() && !vm.is_snapshot))
            .collect()
    }

    /// Validates a view definition and builds the metadata of the new view (not yet registered). Returns None if
    /// the path exists and `if_exists='ignore'` (then the existing view's id is returned by the caller).
    pub fn plan_view(&self, path: &Path, def: &ViewDef, if_exists: IfExists) -> Result<ViewPlanOutcome> {
        let creating = if def.is_snapshot { Creating::Snapshot } else { Creating::View };
        // a replacement must not be one of the view's own ancestors
        let (_, existing) = self.prepare_add(path, false)?;
        if let Some(Entry::Table(existing)) = existing {
            if matches!(if_exists, IfExists::Replace | IfExists::ReplaceForce) {
                let mut chain = vec![def.base_id];
                chain.extend(self.base_chain(def.base_id));
                if chain.contains(&existing) {
                    crate::bail!(
                        UnsupportedOperation,
                        "Cannot use if_exists='replace' with the same name as one of the view's own ancestors."
                    );
                }
            }
        }
        if let Some(obj) = existing {
            match if_exists {
                IfExists::Error => crate::bail!(
                    PathAlreadyExists,
                    "Path {} is an existing {}",
                    path.repr(),
                    self.entry_display_name(obj)
                ),
                IfExists::Ignore => {
                    let type_ok = match obj {
                        Entry::Table(id) => {
                            let k = self.table(id)?.kind;
                            if def.is_snapshot { k == TableKind::Snapshot } else { k != TableKind::Table }
                        }
                        Entry::Dir(_) => false,
                    };
                    if !type_ok {
                        crate::bail!(PathAlreadyExists, "Path {} already exists and is not a {}", path.repr(), creating.name());
                    }
                    let Entry::Table(id) = obj else { return Err(Error::internal("expected a table")) };
                    let existing_base = self.table(id)?.view.as_ref().map(|v| v.base_id());
                    let new_base_path = self.tbl_path(def.base_id, def.base_version)?;
                    let new_base = if self.table(def.base_id)?.is_pure_snapshot() { new_base_path[0].tbl_id } else { def.base_id };
                    if existing_base != Some(new_base) {
                        crate::bail!(
                            PathAlreadyExists,
                            "Path {} already exists as a {} with a different base table",
                            path.repr(),
                            if def.is_snapshot { "snapshot" } else { "view" }
                        );
                    }
                    return Ok(ViewPlanOutcome::Existing(id));
                }
                IfExists::Replace | IfExists::ReplaceForce => {
                    if let Entry::Dir(_) = obj {
                        crate::bail!(
                            PathAlreadyExists,
                            "Path {} already exists as a directory; expected a table, view or snapshot",
                            path.repr()
                        );
                    }
                    if let Entry::Table(id) = obj {
                        self.check_droppable(id, if_exists == IfExists::ReplaceForce, true)?;
                    }
                }
            }
        } else {
            // the parent directory must exist
            self.get_object(&path.parent(), Expected::Dir, false, true)?;
        }

        // the base path (pinned throughout for a snapshot, after the binding checks)
        let live_base_path = self.tbl_path(def.base_id, def.base_version)?;
        let base_path = if def.is_snapshot { self.pin_path(&live_base_path)? } else { live_base_path.clone() };
        let base_md = self.table(base_path[0].tbl_id)?;
        let base_name = base_md.name.clone();
        for e in &base_path {
            if !self.table(e.tbl_id)?.is_data_versioned {
                crate::bail!(
                    UnsupportedOperation,
                    "{}: Views and snapshots are supported on data-versioned tables only",
                    self.display_str(e.tbl_id)?
                );
            }
        }

        // the view's own columns: the select list's items, then the additional columns
        let mut specs: Vec<ColumnSpec> = Vec::new();
        if let Some(items) = &def.select {
            let names = crate::exec::output_names(items);
            for ((e, _), name) in items.iter().zip(names) {
                let stored = e.as_column_ref().is_none();
                specs.push(ColumnSpec {
                    name,
                    col_type: e.col_type.clone(),
                    value_expr: Some(e.clone()),
                    is_pk: false,
                    stored: Some(stored),
                    media_validation: None,
                    comment: None,
                    custom_metadata: Json::Null,
                    destination: None,
                });
            }
        }
        let base_cols = self.path_visible_columns(&live_base_path)?;
        for spec in &def.additional_columns {
            if def.select.is_none() && base_cols.iter().any(|c| c.name == spec.name) {
                crate::bail!(ColumnAlreadyExists, "Column {} already exists in the base table", py_repr(&spec.name));
            }
        }
        specs.extend(def.additional_columns.iter().cloned());
        if let Some(call) = &def.iterator {
            // the iterator's outputs shadow base columns of the same name, but share one namespace with the view's
            // own columns
            let mut duplicates: Vec<&str> =
                call.outputs.iter().filter(|o| specs.iter().any(|s| s.name == o.name)).map(|o| o.name.as_str()).collect();
            duplicates.sort();
            if !duplicates.is_empty() {
                let names: Vec<String> = duplicates.iter().map(|n| py_repr(n)).collect();
                crate::bail!(
                    InvalidSchema,
                    "Column name(s) {} are produced by the iterator and also declared by the view; rename the declared column(s).",
                    names.join(", ")
                );
            }
        }
        let mut seen = std::collections::HashSet::new();
        for spec in &specs {
            validate_column_spec(spec)?;
            if !seen.insert(spec.name.clone()) {
                crate::bail!(ColumnAlreadyExists, "Duplicate column name: {}", spec.name);
            }
            if spec.is_pk {
                crate::bail!(UnsupportedOperation, "Views cannot have primary key columns");
            }
            if spec.value_expr.is_none() && !spec.col_type.nullable {
                crate::bail!(
                    InvalidArgument,
                    "Column {}: Non-computed columns in views must be nullable",
                    py_repr(&spec.name)
                );
            }
        }

        if let Some(sc) = &def.sample {
            if !def.is_snapshot && !sc.is_repeatable() {
                crate::bail!(
                    UnsupportedOperation,
                    "A view that is not a snapshot can only be defined by fractional, unstratified sampling."
                );
            }
            for e in &sc.stratify_by {
                if !self.is_bound_by(e, &live_base_path)? {
                    crate::bail!(
                        UnsupportedOperation,
                        "View sample clause cannot be computed in the context of the base table {}",
                        py_repr(&base_name)
                    );
                }
                crate::functions::udf::validate_storable(e, "A view's sample clause")?;
            }
        }

        // the filter and the value expressions must be computable in the context of the base
        if let Some(p) = &def.predicate {
            crate::functions::udf::validate_storable(p, "A view's filter")?;
            if !self.is_bound_by(p, &live_base_path)? {
                crate::bail!(
                    UnsupportedOperation,
                    "View filter cannot be computed in the context of the base table {}",
                    py_repr(&base_name)
                );
            }
        }
        for spec in &specs {
            if let Some(e) = &spec.value_expr {
                if !self.is_bound_by(e, &live_base_path)? {
                    crate::bail!(
                        UnsupportedOperation,
                        "Column {}: Value expression cannot be computed in the context of the base table {}\n{}",
                        py_repr(&spec.name),
                        py_repr(&base_name),
                        crate::expr::display::display(e)
                    );
                }
            }
        }

        // a snapshot's expressions read the pinned versions
        let retarget = |e: &ExprRef| -> ExprRef {
            if !def.is_snapshot {
                return e.clone();
            }
            e.map_column_refs(&|c| {
                base_path.iter().find(|p| p.tbl_id == c.tbl_id).map(|p| crate::expr::ColumnRef { tbl_version: p.pinned, ..c.clone() })
            })
        };
        let predicate = def.predicate.as_ref().map(retarget);
        let sample = match &def.sample {
            Some(sc) => {
                let mut sc = sc.clone();
                sc.stratify_by = sc.stratify_by.iter().map(retarget).collect();
                if sc.seed.is_none() && !def.is_snapshot {
                    // a live view keeps sampling the same way: fix the seed
                    sc.seed = Some((crate::exec::sample::random_u64() >> 1) as i64);
                }
                Some(sc.to_json()?)
            }
            None => None,
        };
        for spec in specs.iter_mut() {
            spec.value_expr = spec.value_expr.as_ref().map(retarget);
        }
        let iterator = match &def.iterator {
            Some(call) => {
                for e in call.arg_exprs() {
                    if !self.is_bound_by(e, &live_base_path)? {
                        crate::bail!(
                            UnsupportedOperation,
                            "View iterator arguments cannot be computed in the context of the base table {}",
                            py_repr(&base_name)
                        );
                    }
                }
                let retargeted = crate::functions::iterator::IteratorCall {
                    def: call.def.clone(),
                    args: call.args.iter().map(retarget).collect(),
                    kwargs: call.kwargs.iter().map(|(k, e)| (k.clone(), retarget(e))).collect(),
                    binding: call.binding.clone(),
                    outputs: call.outputs.clone(),
                    validation_error: call.validation_error.clone(),
                };
                Some(retargeted.to_json()?)
            }
            None => None,
        };

        let ts = now();
        let id = Uuid::new_v4();
        let mut columns = Vec::new();
        let mut schema_cols = Vec::new();
        // a component view's first columns are the iterator's outputs (`pos` first)
        if let Some(call) = &def.iterator {
            for o in &call.outputs {
                let col_id = columns.len() as ColId;
                columns.push(ColumnMd {
                    id: col_id,
                    col_type: o.col_type.clone(),
                    is_pk: false,
                    stored: o.is_stored,
                    value_expr: None,
                    schema_version_add: 0,
                    schema_version_drop: None,
                    media_validation: None,
                    comment: None,
                    custom_metadata: Json::Null,
                    destination: None,
                });
                schema_cols.push(SchemaColumn::new(col_id, o.name.clone()));
            }
        }
        let num_iter_cols = columns.len();
        for (i, c) in specs.into_iter().enumerate() {
            let col_id = (num_iter_cols + i) as ColId;
            let is_computed = c.value_expr.is_some();
            columns.push(ColumnMd {
                id: col_id,
                col_type: c.col_type,
                is_pk: false,
                stored: if is_computed { c.stored.unwrap_or(true) } else { true },
                value_expr: c.value_expr.as_ref().map(|e| crate::expr::serde::to_json(e)).transpose()?,
                schema_version_add: 0,
                schema_version_drop: None,
                media_validation: c.media_validation,
                comment: c.comment,
                custom_metadata: c.custom_metadata,
                destination: c.destination,
            });
            schema_cols.push(SchemaColumn::new(col_id, c.name));
        }
        let mut version0 = VersionMd::new(0, ts, 0, "schema", None);
        version0.base_versions =
            base_path.iter().map(|e| Ok(e.pinned.unwrap_or(self.table(e.tbl_id)?.current_version))).collect::<Result<_>>()?;
        let mut md = TableMd {
            id,
            dir_id: Uuid::nil(),
            name: path.name().to_string(),
            kind: if def.is_snapshot { TableKind::Snapshot } else { TableKind::View },
            is_data_versioned: true,
            has_default_idxs: def.has_default_idxs,
            created_at: ts,
            current_version: 0,
            current_schema_version: 0,
            next_col_id: columns.len() as ColId,
            next_rowid: 0,
            columns,
            schema_versions: vec![Arc::new(SchemaVersionMd {
                schema_version: 0,
                preceding_schema_version: None,
                columns: schema_cols,
                comment: def.comment.clone(),
                custom_metadata: def.custom_metadata.clone(),
                media_validation: def.media_validation,
            })],
            versions: vec![Arc::new(version0)],
            view: Some(ViewMd {
                bases: base_path.iter().map(|e| (e.tbl_id, e.pinned)).collect(),
                predicate: predicate.as_ref().map(|p| crate::expr::serde::to_json(p)).transpose()?,
                is_snapshot: def.is_snapshot,
                include_base_columns: def.select.is_none(),
                iterator,
                sample,
            }),
            storage: Default::default(),
            indexes: Default::default(),
        };
        if def.has_default_idxs {
            super::indexes::add_default_view_indexes(&mut md)?;
        }
        Ok(ViewPlanOutcome::New(Box::new(md), base_path))
    }

    /// Registers a planned view (`plan_view()`) with its initial rows: replaces an existing object at the path
    /// (`if_exists='replace'`), re-checks that the base path hasn't changed, and commits.
    pub fn commit_view(
        &mut self,
        path: &Path,
        mut md: TableMd,
        base_path: &[PathElem],
        rows: Option<crate::store::rows::TableData>,
        stats: RowCountStats,
        if_exists: IfExists,
    ) -> Result<Uuid> {
        for (e, v) in base_path.iter().zip(md.versions[0].base_versions.iter()) {
            if e.pinned.is_none() && self.table(e.tbl_id)?.current_version != *v {
                crate::bail!(
                    ConcurrentModification,
                    "The base of view {} changed while the view was being created; retry",
                    py_repr(path.name())
                );
            }
        }
        let creating = if md.is_snapshot() { Creating::Snapshot } else { Creating::View };
        if let Some(Entry::Table(id)) = self.handle_path_collision(path, creating, if_exists)? {
            return Ok(id);
        }
        let Some(Entry::Dir(dir_id)) = self.get_object(&path.parent(), Expected::Dir, false, true)? else {
            return Err(Error::internal("create_view(): parent is not a directory"));
        };
        md.dir_id = dir_id;
        Arc::make_mut(&mut md.versions[0]).row_count_stats = stats;
        let id = md.id;
        self.tables.insert(id, Arc::new(md));
        match rows {
            Some(data) => self.write_data(id, data)?,
            None => self.commit()?,
        }
        Ok(id)
    }
}

/// What a query over a table reads when the views in its path are only filters: the root base table and the filters
/// of the views between it and the table (`Table.compute()`).
#[derive(Debug, Clone)]
pub struct ScanSource {
    pub base_tbl_id: Uuid,
    /// version of the root base table (None: current)
    pub base_version: Option<i64>,
    pub predicates: Vec<Json>,
}

impl Catalog {
    pub fn scan_source(&self, tbl_id: Uuid, version: Option<i64>) -> Result<ScanSource> {
        let path = self.tbl_path(tbl_id, version)?;
        let root = path.last().ok_or_else(|| Error::internal("empty table path"))?;
        let mut predicates = Vec::new();
        for e in path.iter().rev() {
            if let Some(p) = self.table(e.tbl_id)?.view.as_ref().and_then(|v| v.predicate.clone()) {
                predicates.push(p);
            }
        }
        Ok(ScanSource { base_tbl_id: root.tbl_id, base_version: root.pinned, predicates })
    }
}

/// The outcome of `Catalog::plan_view()`.
pub enum ViewPlanOutcome {
    /// `if_exists='ignore'` and a matching view exists
    Existing(Uuid),
    /// the new view's metadata and its base path
    New(Box<TableMd>, Vec<PathElem>),
}
