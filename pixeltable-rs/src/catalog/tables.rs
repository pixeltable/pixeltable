//! Table operations: create, get, drop, and simple schema changes (add/drop/rename column).

use std::sync::Arc;
use uuid::Uuid;

use super::dirs::{Creating, Expected};
use super::path::{Path, fold_identifier, is_valid_identifier};
use super::runtime::is_reserved_name;
use super::table::*;
use super::{Catalog, Entry, IfExists, IfNotExists, now};
use crate::error::{Error, Result, py_repr};
use crate::expr::ExprRef;
use crate::types::ColumnType;
use crate::value::Json;

/// A column definition from a schema (`create_table`, `add_column`, `add_computed_column`).
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    /// folded name
    pub name: String,
    pub col_type: ColumnType,
    pub value_expr: Option<ExprRef>,
    pub is_pk: bool,
    pub stored: Option<bool>,
    pub media_validation: Option<MediaValidation>,
    pub comment: Option<String>,
    pub custom_metadata: Json,
    pub destination: Option<String>,
}

/// The outcome of `Catalog::check_existing_column()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExistingColumn {
    /// no column of that name
    Absent,
    /// keep the existing column and skip the new one (`if_exists='ignore'`)
    Ignore,
    /// drop this column before adding the new one (`if_exists='replace'`)
    Replace(ColId),
}

/// Upstream `Column.validate_name()`.
pub fn validate_column_name(name: &str) -> Result<()> {
    let folded = fold_identifier(name);
    if is_reserved_name(&folded) {
        crate::bail!(
            InvalidColumnName,
            "{} is a reserved name in Pixeltable; please choose a different column name.",
            py_repr(name)
        );
    }
    if !is_valid_identifier(name, false) {
        crate::bail!(InvalidColumnName, "Invalid column name: {name}");
    }
    Ok(())
}

/// Upstream `_verify_computed_col_value()`: a computed column cannot read the cell metadata (errortype, errormsg,
/// cellmd) of another column.
pub fn check_no_cellmd_refs(e: &crate::expr::Expr) -> Result<()> {
    use crate::expr::{ColumnProp, ExprKind};
    let mut found = None;
    e.visit(&mut |n| {
        if let ExprKind::ColumnProperty { prop, .. } = &n.kind {
            if matches!(prop, ColumnProp::ErrorType | ColumnProp::ErrorMsg | ColumnProp::CellMd) && found.is_none() {
                found = Some(prop.name());
            }
        }
    });
    if let Some(name) = found {
        crate::bail!(
            UnsupportedOperation,
            "Use of a reference to the {} property of another column is not allowed in a computed column.",
            py_repr(name)
        );
    }
    Ok(())
}

/// Upstream `Column` construction checks.
pub fn validate_column_spec(spec: &ColumnSpec) -> Result<()> {
    validate_column_name(&spec.name)?;
    if spec.stored == Some(false) && spec.value_expr.is_none() {
        crate::bail!(UnsupportedOperation, "Column {}: `stored=False` only applies to computed columns", py_repr(&spec.name));
    }
    if spec.destination.is_some() && (spec.value_expr.is_none() || spec.stored == Some(false)) {
        crate::bail!(
            UnsupportedOperation,
            "Column {}: `destination` property only applies to stored computed columns",
            py_repr(&spec.name)
        );
    }
    // upstream `Column.check_value_expr()`
    if let Some(e) = &spec.value_expr {
        crate::functions::udf::validate_storable(e, &format!("Computed column {}", py_repr(&spec.name)))?;
        check_no_cellmd_refs(e)?;
        let mut has_window_call = false;
        e.visit(&mut |n| {
            if let crate::expr::ExprKind::FunctionCall(fc) = &n.kind {
                has_window_call |= fc.is_window();
            }
        });
        if spec.stored == Some(false) && has_window_call {
            crate::bail!(
                UnsupportedOperation,
                "Column {}: `stored=False` not supported for columns computed with window functions:\n{}",
                py_repr(&spec.name),
                crate::expr::display::display(e)
            );
        }
    }
    Ok(())
}

/// `pxt.Int`-style name of a type, for the nullable-primary-key message.
fn base_type_name(t: &ColumnType) -> String {
    t.with_nullable(false).repr()
}

impl Catalog {
    /// Creates a table; returns its id and whether it was created (false: an existing table was returned for
    /// `if_exists='ignore'`).
    #[allow(clippy::too_many_arguments)]
    pub fn create_table(
        &mut self,
        path: &Path,
        columns: Vec<ColumnSpec>,
        if_exists: IfExists,
        comment: Option<String>,
        custom_metadata: Json,
        media_validation: MediaValidation,
        is_data_versioned: bool,
        has_default_idxs: bool,
    ) -> Result<(Uuid, bool)> {
        for c in &columns {
            validate_column_spec(c)?;
        }
        if let Some(Entry::Table(id)) = self.handle_path_collision(path, Creating::Table, if_exists)? {
            return Ok((id, false));
        }
        let Some(Entry::Dir(dir_id)) = self.get_object(&path.parent(), Expected::Dir, false, true)? else {
            return Err(Error::internal("create_table(): parent is not a directory"));
        };
        for c in &columns {
            if c.is_pk && c.col_type.nullable {
                crate::bail!(
                    UnsupportedOperation,
                    "Primary key column {} cannot be nullable. Declare it as non-nullable instead: `pxt.{}`",
                    py_repr(&c.name),
                    base_type_name(&c.col_type)
                );
            }
        }
        for c in &columns {
            if c.value_expr.as_ref().is_some_and(|e| !e.column_refs().is_empty()) {
                return Err(Error::not_implemented(
                    "a computed column that references other columns in the create_table() schema",
                ));
            }
        }
        let ts = now();
        let tbl_id = Uuid::new_v4();
        let mut col_mds = Vec::new();
        let mut schema_cols = Vec::new();
        for (i, c) in columns.into_iter().enumerate() {
            let id = i as ColId;
            col_mds.push(ColumnMd {
                id,
                col_type: c.col_type,
                is_pk: c.is_pk,
                stored: c.stored.unwrap_or(true),
                value_expr: c.value_expr.as_ref().map(|e| crate::expr::serde::to_json(e)).transpose()?,
                schema_version_add: 0,
                schema_version_drop: None,
                media_validation: c.media_validation,
                comment: c.comment,
                custom_metadata: c.custom_metadata,
                destination: c.destination,
            });
            schema_cols.push(SchemaColumn::new(id, c.name));
        }
        let md = TableMd {
            id: tbl_id,
            dir_id,
            name: path.name().to_string(),
            kind: TableKind::Table,
            is_data_versioned,
            has_default_idxs,
            created_at: ts,
            current_version: 0,
            current_schema_version: 0,
            next_col_id: col_mds.len() as ColId,
            next_rowid: 0,
            data_version: 0,
            columns: col_mds,
            schema_versions: vec![Arc::new(SchemaVersionMd {
                schema_version: 0,
                preceding_schema_version: None,
                columns: schema_cols,
                comment,
                custom_metadata,
                media_validation,
            })],
            versions: vec![Arc::new(VersionMd::new(0, ts, 0, "schema", None))],
            view: None,
            storage: Default::default(),
            indexes: Default::default(),
        };
        let col_ids: Vec<ColId> = md.columns.iter().map(|c| c.id).collect();
        self.tables.insert(tbl_id, Arc::new(md));
        for col_id in col_ids {
            self.add_default_index(tbl_id, col_id)?;
        }
        self.commit()?;
        Ok((tbl_id, true))
    }

    /// Resolves a table path (upstream `get_table()`); None if it does not exist and `if_not_exists='ignore'`.
    pub fn get_table(&self, path: &Path, if_not_exists: IfNotExists) -> Result<Option<Uuid>> {
        match self.get_object(path, Expected::Table, false, if_not_exists == IfNotExists::Error)? {
            Some(Entry::Table(id)) => {
                if let Some(v) = path.version {
                    let md = self.table(id)?;
                    if !md.is_data_versioned {
                        crate::bail!(
                            UnsupportedOperation,
                            "{}: Version-pinned access is supported on data-versioned tables only",
                            self.display_str(id)?
                        );
                    }
                    if md.version_md(v).is_none() {
                        crate::bail!(
                            InvalidArgument,
                            "Table {} has no version {v} (current version is {})",
                            py_repr(&path.without_version().to_string()),
                            md.current_version
                        );
                    }
                }
                Ok(Some(id))
            }
            _ => Ok(None),
        }
    }

    /// `_display_str()`: e.g. `table 'dir1/t1'`.
    pub fn display_str(&self, tbl_id: Uuid) -> Result<String> {
        let kind = self.table(tbl_id)?.kind.display_name();
        Ok(format!("{kind} {}", self.table_path(tbl_id)?.repr()))
    }

    /// Ids of the views (and snapshots) defined directly over `tbl_id`.
    pub fn view_ids(&self, tbl_id: Uuid) -> Vec<Uuid> {
        let mut views: Vec<&Arc<TableMd>> =
            self.tables.values().filter(|t| t.view.as_ref().is_some_and(|v| v.base_id() == tbl_id)).collect();
        views.sort_by(|a, b| a.created_at.total_cmp(&b.created_at));
        views.into_iter().map(|t| t.id).collect()
    }

    fn paths_str(&self, ids: &[Uuid], max_paths: usize) -> String {
        let mut paths: Vec<String> = ids.iter().filter_map(|id| self.table_path(*id).ok()).map(|p| p.to_string()).collect();
        paths.sort();
        if paths.len() <= max_paths {
            return paths.iter().map(|p| py_repr(p)).collect::<Vec<_>>().join(", ");
        }
        let listed = paths[..max_paths].iter().map(|p| py_repr(p)).collect::<Vec<_>>().join(", ");
        format!("{listed} and {} more", paths.len() - max_paths)
    }

    /// Fails unless the table can be dropped: without `force`, nothing may depend on it.
    pub fn check_droppable(&self, tbl_id: Uuid, force: bool, is_replace: bool) -> Result<()> {
        let views = self.view_ids(tbl_id);
        if views.is_empty() || force {
            return Ok(());
        }
        let dependents = format!("the following depend on it: {}", self.paths_str(&views, 5));
        let display = self.display_str(tbl_id)?;
        if is_replace {
            crate::bail!(
                ConstraintViolation,
                "{display} already exists and {dependents}. Use `if_exists='replace_force'` to replace it."
            );
        }
        crate::bail!(ConstraintViolation, "{display} cannot be dropped, {dependents}.");
    }

    /// Drops a table (and, with `force`, every view over it). Does not commit.
    pub fn drop_table_by_id(&mut self, tbl_id: Uuid, force: bool, is_replace: bool) -> Result<()> {
        self.check_droppable(tbl_id, force, is_replace)?;
        for v in self.view_ids(tbl_id) {
            if self.tables.contains_key(&v) {
                self.drop_table_by_id(v, true, is_replace)?;
            }
        }
        self.remove_table_record(tbl_id);
        Ok(())
    }

    pub fn drop_table(&mut self, path: &Path, force: bool, if_not_exists: IfNotExists) -> Result<()> {
        let raise = if_not_exists == IfNotExists::Error && !force;
        if let Some(Entry::Table(id)) = self.get_object(path, Expected::Table, false, raise)? {
            self.drop_table_by_id(id, force, false)?;
            self.commit()?;
        }
        Ok(())
    }

    /// Fails if the table cannot be modified (snapshots, pinned versions): `op_descr` completes
    /// "Cannot <op_descr> a snapshot." (e.g. "add columns to").
    pub fn check_mutable(&self, tbl_id: Uuid, version: Option<i64>, op_descr: &str) -> Result<()> {
        let md = self.table(tbl_id)?;
        if md.is_snapshot() || version.is_some() {
            crate::bail!(UnsupportedOperation, "{}: Cannot {op_descr} a snapshot.", self.display_str(tbl_id)?);
        }
        Ok(())
    }

    /// Starts a new table version (and, for schema changes, a new schema version equal to it).
    /// Returns the new version.
    pub fn bump_version(&mut self, tbl_id: Uuid, schema_change: Option<String>) -> Result<i64> {
        let base_versions = self.current_base_versions(tbl_id)?;
        let md = self.table_mut(tbl_id)?;
        let is_schema = schema_change.is_some();
        if !md.is_data_versioned && !is_schema {
            return Ok(md.current_version);
        }
        let new_version = md.current_version + 1;
        md.current_version = new_version;
        if is_schema {
            let mut sv: SchemaVersionMd = md.current_schema()?.clone();
            sv.preceding_schema_version = Some(sv.schema_version);
            sv.schema_version = new_version;
            md.schema_versions.push(Arc::new(sv));
            md.current_schema_version = new_version;
        }
        let schema_version = md.current_schema_version;
        let mut vmd =
            VersionMd::new(new_version, now(), schema_version, if is_schema { "schema" } else { "data" }, schema_change);
        vmd.base_versions = base_versions;
        md.versions.push(Arc::new(vmd));
        Ok(new_version)
    }

    /// The metadata of the current version of a table.
    pub fn current_version_md_mut(&mut self, tbl_id: Uuid) -> Result<&mut VersionMd> {
        let md = self.table_mut(tbl_id)?;
        let v = md.current_version;
        md.version_md_mut(v).ok_or_else(|| Error::internal("table has no current version record"))
    }

    /// Adds a column at a new schema version; returns its id. Does not commit.
    pub fn push_column(&mut self, tbl_id: Uuid, spec: ColumnSpec, value_expr: Option<Json>) -> Result<ColId> {
        let change = format!("Added {}", py_repr(&spec.name));
        self.bump_version(tbl_id, Some(change))?;
        self.add_column_md(tbl_id, spec, value_expr)
    }

    /// Adds a column to the current schema version (which the caller has created); returns its id.
    pub fn add_column_md(&mut self, tbl_id: Uuid, spec: ColumnSpec, value_expr: Option<Json>) -> Result<ColId> {
        let md = self.table_mut(tbl_id)?;
        let new_version = md.current_schema_version;
        let id = md.next_col_id;
        md.next_col_id += 1;
        let is_computed = value_expr.is_some();
        md.columns.push(ColumnMd {
            id,
            col_type: spec.col_type,
            is_pk: false,
            stored: if is_computed { spec.stored.unwrap_or(true) } else { true },
            value_expr,
            schema_version_add: new_version,
            schema_version_drop: None,
            media_validation: spec.media_validation,
            comment: spec.comment,
            custom_metadata: spec.custom_metadata,
            destination: spec.destination,
        });
        let sv = md.last_schema_version_mut().ok_or_else(|| Error::internal("table has no schema versions"))?;
        sv.columns.push(SchemaColumn::new(id, spec.name));
        self.add_default_index(tbl_id, id)?;
        Ok(id)
    }

    /// Upstream `_ignore_or_drop_existing_columns()`, without the drop: what to do about an existing column of the
    /// name a new column is to get.
    pub fn check_existing_column(&self, tbl_id: Uuid, name: &str, if_exists: IfExists) -> Result<ExistingColumn> {
        if !self.resolve_columns(tbl_id, None)?.iter().any(|c| c.name == name) {
            return Ok(ExistingColumn::Absent);
        }
        match if_exists {
            IfExists::Error => crate::bail!(ColumnAlreadyExists, "Duplicate column name: {name}"),
            IfExists::Ignore => Ok(ExistingColumn::Ignore),
            IfExists::Replace | IfExists::ReplaceForce => {
                let md = self.table(tbl_id)?;
                let Some(col) = md.column_by_name(name, None)? else {
                    crate::bail!(
                        UnsupportedOperation,
                        "Column {} is a base table column. Cannot replace it.",
                        py_repr(name)
                    );
                };
                let dependents = self.dependent_columns(tbl_id, col.md.id)?;
                if !dependents.is_empty() {
                    crate::bail!(
                        ColumnAlreadyExists,
                        "Column {} already exists and the following columns depend on it: {}. Cannot {} it.",
                        py_repr(name),
                        dependents.join(", "),
                        if if_exists == IfExists::Replace { "replace" } else { "replace_force" }
                    );
                }
                Ok(ExistingColumn::Replace(col.md.id))
            }
        }
    }

    /// Names of the visible computed columns of `tbl_id` and of its live views (recursively) whose value expression
    /// references column `col_id` of `tbl_id`.
    pub fn dependent_columns(&self, tbl_id: Uuid, col_id: ColId) -> Result<Vec<String>> {
        let mut tables = vec![tbl_id];
        tables.extend(self.mutable_views_recursive(tbl_id));
        let mut out = Vec::new();
        for t in tables {
            for c in self.table(t)?.columns_at(None)? {
                let refs = c.md.value_expr.as_ref().is_some_and(|e| {
                    crate::expr::serde::from_json(e)
                        .map(|e| e.column_refs().iter().any(|r| r.tbl_id == tbl_id && r.col_id == col_id))
                        .unwrap_or(false)
                });
                if refs {
                    out.push(c.name);
                }
            }
        }
        Ok(out)
    }

    /// Adds non-computed columns (upstream `add_columns()` for plain columns).
    pub fn add_columns(&mut self, tbl_id: Uuid, specs: Vec<ColumnSpec>, if_exists: IfExists) -> Result<Vec<String>> {
        self.check_mutable(tbl_id, None, "add columns to")?;
        let mut to_add = Vec::new();
        for spec in specs {
            match self.check_existing_column(tbl_id, &spec.name, if_exists)? {
                ExistingColumn::Absent => {}
                ExistingColumn::Ignore => continue,
                ExistingColumn::Replace(_) => self.drop_column(tbl_id, &spec.name, IfNotExists::Error)?,
            }
            validate_column_spec(&spec)?;
            if spec.value_expr.is_some() {
                return Err(Error::not_implemented("add_columns() with computed columns; use add_computed_column()"));
            }
            if spec.is_pk {
                crate::bail!(
                    UnsupportedOperation,
                    "Cannot add primary key column {} after table creation",
                    py_repr(&spec.name)
                );
            }
            if !spec.col_type.nullable {
                let md = self.table(tbl_id)?.clone();
                let has_rows = self.data(tbl_id)?.visible_at(md.current_version).next().is_some();
                if has_rows {
                    crate::bail!(
                        UnsupportedOperation,
                        "Cannot add non-nullable column {} to table {} with existing rows",
                        py_repr(&spec.name),
                        py_repr(&md.name)
                    );
                }
            }
            to_add.push(spec);
        }
        let names: Vec<String> = to_add.iter().map(|s| s.name.clone()).collect();
        if !names.is_empty() {
            let quoted: Vec<String> = names.iter().map(|n| py_repr(n)).collect();
            self.bump_version(tbl_id, Some(format!("Added {}", quoted.join(", "))))?;
            for spec in to_add {
                self.add_column_md(tbl_id, spec, None)?;
            }
            self.commit()?;
        }
        Ok(names)
    }

    /// The column `name` of `tbl_id` for a schema change: COLUMN_NOT_FOUND if unknown, UNSUPPORTED_OPERATION
    /// (`Cannot <verb> base table column ...`) if it belongs to a base. None: unknown and `if_not_exists='ignore'`.
    pub fn own_column(&self, tbl_id: Uuid, name: &str, verb: &str, if_not_exists: IfNotExists) -> Result<Option<ResolvedColumn>> {
        let Some(col) = self.resolve_column_by_name(tbl_id, None, name)? else {
            if if_not_exists == IfNotExists::Ignore {
                return Ok(None);
            }
            crate::bail!(ColumnNotFound, "Unknown column: {name}");
        };
        if col.tbl_id != tbl_id {
            crate::bail!(UnsupportedOperation, "Cannot {verb} base table column {}", py_repr(&col.name));
        }
        Ok(Some(col))
    }

    pub fn drop_column(&mut self, tbl_id: Uuid, name: &str, if_not_exists: IfNotExists) -> Result<()> {
        self.check_mutable(tbl_id, None, "drop columns from")?;
        let Some(col) = self.own_column(tbl_id, name, "drop", if_not_exists)? else { return Ok(()) };
        if col.md.is_pk {
            crate::bail!(UnsupportedOperation, "Cannot drop primary key column {}", py_repr(&col.name));
        }
        let dependents = self.dependent_columns(tbl_id, col.md.id)?;
        if !dependents.is_empty() {
            crate::bail!(
                UnsupportedOperation,
                "Cannot drop column {} because the following columns depend on it:\n{}",
                py_repr(&col.name),
                dependents.join(", ")
            );
        }
        let mut dependent_views = Vec::new();
        for v in self.mutable_views_recursive(tbl_id) {
            let vmd = self.table(v)?;
            if let Some(p) = vmd.view.as_ref().and_then(|vm| vm.predicate.as_ref()) {
                let pred = crate::expr::serde::from_json(p)?;
                if pred.column_refs().iter().any(|r| r.tbl_id == tbl_id && r.col_id == col.md.id) {
                    dependent_views.push(format!(
                        "view: {}, predicate: {}",
                        self.table_path(v)?,
                        crate::expr::display::display(&self.with_column_names(&pred, &[]))
                    ));
                }
            }
        }
        if !dependent_views.is_empty() {
            dependent_views.sort();
            crate::bail!(
                UnsupportedOperation,
                "Cannot drop column {} because the following views depend on it:\n{}",
                py_repr(&col.name),
                dependent_views.join("\n")
            );
        }
        let visible = self.resolve_columns(tbl_id, None)?;
        if visible.len() == 1 && visible[0].md.id == col.md.id {
            crate::bail!(
                MissingRequired,
                "Cannot drop column {} because it is the last remaining column in this table. Tables must have at least one column.",
                py_repr(&col.name)
            );
        }
        let new_version = self.bump_version(tbl_id, Some(format!("Dropped {}", py_repr(&col.name))))?;
        let md = self.table_mut(tbl_id)?;
        if let Some(c) = md.column_mut(col.md.id) {
            c.schema_version_drop = Some(new_version);
        }
        if let Some(sv) = md.last_schema_version_mut() {
            sv.columns.retain(|c| c.col_id != col.md.id);
        }
        self.remove_indexes_on_column(tbl_id, col.md.id, new_version)?;
        self.commit()
    }

    /// `alter_column()`: widens the type of a plain column (to nullable) at a new schema version. Commits.
    pub fn alter_column(&mut self, tbl_id: Uuid, col: &ResolvedColumn, new_type: &ColumnType) -> Result<()> {
        if col.tbl_id != tbl_id {
            crate::bail!(UnsupportedOperation, "Cannot alter base table column {}", py_repr(&col.name));
        }
        if col.md.is_computed() {
            crate::bail!(UnsupportedOperation, "Cannot alter the type of computed column {}", py_repr(&col.name));
        }
        if col.md.is_pk {
            crate::bail!(UnsupportedOperation, "Cannot alter the type of primary key column {}", py_repr(&col.name));
        }
        let dependents = self.dependent_columns_qualified(tbl_id, col.md.id)?;
        if !dependents.is_empty() {
            crate::bail!(
                UnsupportedOperation,
                "Cannot alter column {} because the following columns depend on it: {}",
                py_repr(&col.name),
                dependents.join(", ")
            );
        }
        let old = &col.md.col_type;
        if new_type == old {
            return Ok(());
        }
        if !(new_type.matches(old) && !old.nullable && new_type.nullable) {
            crate::bail!(
                UnsupportedOperation,
                "Column {} type cannot be changed from {} to {}",
                py_repr(&col.name),
                old.repr(),
                new_type.repr()
            );
        }
        self.bump_version(tbl_id, Some(format!("Altered {}", py_repr(&col.name))))?;
        let md = self.table_mut(tbl_id)?;
        let sv = md.last_schema_version_mut().ok_or_else(|| Error::internal("table has no schema versions"))?;
        for c in sv.columns.iter_mut().filter(|c| c.col_id == col.md.id) {
            c.col_type = Some(new_type.clone());
        }
        self.commit()
    }

    /// Like `dependent_columns()`, as `table.column` names.
    pub fn dependent_columns_qualified(&self, tbl_id: Uuid, col_id: ColId) -> Result<Vec<String>> {
        let mut tables = vec![tbl_id];
        tables.extend(self.mutable_views_recursive(tbl_id));
        let mut out = Vec::new();
        for t in tables {
            let md = self.table(t)?;
            for c in md.columns_at(None)? {
                let refs = c.md.value_expr.as_ref().is_some_and(|e| {
                    crate::expr::serde::from_json(e)
                        .map(|e| e.column_refs().iter().any(|r| r.tbl_id == tbl_id && r.col_id == col_id))
                        .unwrap_or(false)
                });
                if refs {
                    out.push(format!("{}.{}", md.name, c.name));
                }
            }
        }
        Ok(out)
    }

    /// The live views of `tbl_id` and their live views, recursively.
    pub fn mutable_views_recursive(&self, tbl_id: Uuid) -> Vec<Uuid> {
        let mut out = Vec::new();
        for v in self.mutable_view_ids(tbl_id) {
            out.push(v);
            out.extend(self.mutable_views_recursive(v));
        }
        out
    }

    pub fn rename_column(&mut self, tbl_id: Uuid, old_name: &str, new_name: &str) -> Result<()> {
        self.check_mutable(tbl_id, None, "rename columns of")?;
        validate_column_name(new_name)?;
        let md = self.table(tbl_id)?.clone();
        let Some(col) = self.own_column(tbl_id, old_name, "rename", IfNotExists::Error)? else {
            return Ok(());
        };
        let folded_new = fold_identifier(new_name);
        if folded_new == col.name {
            // a case-only rename: nothing changes
            return Ok(());
        }
        if md.column_by_name(&folded_new, None)?.is_some() {
            crate::bail!(ColumnAlreadyExists, "Column {} already exists", py_repr(new_name));
        }
        self.bump_version(tbl_id, Some(format!("Renamed {} to {}", py_repr(&col.name), py_repr(&folded_new))))?;
        let md = self.table_mut(tbl_id)?;
        if let Some(sv) = md.last_schema_version_mut() {
            for c in sv.columns.iter_mut() {
                if c.col_id == col.md.id {
                    c.name = folded_new.clone();
                }
            }
        }
        self.commit()
    }
}
