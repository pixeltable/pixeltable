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

/// Upstream `Column` construction checks.
pub fn validate_column_spec(spec: &ColumnSpec) -> Result<()> {
    validate_column_name(&spec.name)?;
    if spec.stored.is_some() && spec.value_expr.is_none() {
        crate::bail!(
            UnsupportedOperation,
            "Column {}: `stored={}` only applies to computed columns",
            py_repr(&spec.name),
            if spec.stored == Some(true) { "True" } else { "False" }
        );
    }
    if spec.destination.is_some() && (spec.value_expr.is_none() || spec.stored == Some(false)) {
        crate::bail!(
            UnsupportedOperation,
            "Column {}: `destination` property only applies to stored computed columns",
            py_repr(&spec.name)
        );
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
            schema_cols.push(SchemaColumn { col_id: id, name: c.name });
        }
        let md = TableMd {
            id: tbl_id,
            dir_id,
            name: path.name().to_string(),
            kind: TableKind::Table,
            is_data_versioned,
            has_default_idxs: false,
            created_at: ts,
            current_version: 0,
            current_schema_version: 0,
            next_col_id: col_mds.len() as ColId,
            next_rowid: 0,
            columns: col_mds,
            schema_versions: vec![SchemaVersionMd {
                schema_version: 0,
                preceding_schema_version: None,
                columns: schema_cols,
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
            view: None,
            data_file: None,
        };
        self.tables.insert(tbl_id, Arc::new(md));
        self.commit()?;
        Ok((tbl_id, true))
    }

    /// Resolves a table path (upstream `get_table()`); None if it does not exist and `if_not_exists='ignore'`.
    pub fn get_table(&self, path: &Path, if_not_exists: IfNotExists) -> Result<Option<Uuid>> {
        match self.get_object(path, Expected::Table, false, if_not_exists == IfNotExists::Error)? {
            Some(Entry::Table(id)) => {
                if let Some(v) = path.version {
                    let md = self.table(id)?;
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
            self.tables.values().filter(|t| t.view.as_ref().is_some_and(|v| v.base_id == tbl_id)).collect();
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

    /// Drops a table (and, with `force`, every view over it). Does not commit.
    pub fn drop_table_by_id(&mut self, tbl_id: Uuid, force: bool, is_replace: bool) -> Result<()> {
        let views = self.view_ids(tbl_id);
        if !views.is_empty() {
            if force {
                for v in views {
                    if self.tables.contains_key(&v) {
                        self.drop_table_by_id(v, true, is_replace)?;
                    }
                }
            } else {
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
        }
        if let Some(md) = self.remove_table_record(tbl_id) {
            if let Some(f) = &md.data_file {
                self.store.delete_table_data(f);
            }
            self.store.delete_media(tbl_id);
        }
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

    /// Fails if the table cannot be modified (snapshots, pinned versions).
    pub fn check_mutable(&self, tbl_id: Uuid, version: Option<i64>, op_descr: &str) -> Result<()> {
        let md = self.table(tbl_id)?;
        if md.is_snapshot() || version.is_some() {
            crate::bail!(UnsupportedOperation, "{op_descr} on a snapshot is not allowed.");
        }
        Ok(())
    }

    /// Starts a new table version (and, for schema changes, a new schema version equal to it).
    /// Returns the new version.
    pub fn bump_version(&mut self, tbl_id: Uuid, schema_change: Option<String>) -> Result<i64> {
        let md = self.table_mut(tbl_id)?;
        let is_schema = schema_change.is_some();
        if !md.is_data_versioned && !is_schema {
            return Ok(md.current_version);
        }
        let new_version = md.current_version + 1;
        md.current_version = new_version;
        if is_schema {
            let mut sv = md.current_schema()?.clone();
            sv.preceding_schema_version = Some(sv.schema_version);
            sv.schema_version = new_version;
            md.schema_versions.push(sv);
            md.current_schema_version = new_version;
        }
        let schema_version = md.current_schema_version;
        md.versions.push(VersionMd {
            version: new_version,
            created_at: now(),
            schema_version,
            user: None,
            change_type: if is_schema { "schema".into() } else { "data".into() },
            schema_change,
            row_count_stats: RowCountStats::default(),
            cascade_row_count_stats: RowCountStats::default(),
        });
        Ok(new_version)
    }

    /// Adds non-computed columns (upstream `add_columns()` for plain columns).
    pub fn add_columns(&mut self, tbl_id: Uuid, specs: Vec<ColumnSpec>, if_exists: IfExists) -> Result<Vec<String>> {
        self.check_mutable(tbl_id, None, "add_column()")?;
        let md = self.table(tbl_id)?.clone();
        let existing = md.columns_at(None)?;
        let mut to_add = Vec::new();
        for spec in specs {
            validate_column_spec(&spec)?;
            if spec.value_expr.is_some() {
                return Err(Error::not_implemented("computed columns"));
            }
            if spec.is_pk {
                crate::bail!(UnsupportedOperation, "Cannot add a primary key column to an existing table");
            }
            if existing.iter().any(|c| c.name == spec.name) {
                match if_exists {
                    IfExists::Error => crate::bail!(ColumnAlreadyExists, "Duplicate column name: {}", spec.name),
                    IfExists::Ignore => continue,
                    _ => return Err(Error::not_implemented("add_column() with if_exists='replace'")),
                }
            }
            if !spec.col_type.nullable {
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
        if to_add.is_empty() {
            return Ok(Vec::new());
        }
        let names: Vec<String> = to_add.iter().map(|s| s.name.clone()).collect();
        let change = format!("Added {}", names.iter().map(|n| py_repr(n)).collect::<Vec<_>>().join(", "));
        let new_version = self.bump_version(tbl_id, Some(change))?;
        let md = self.table_mut(tbl_id)?;
        for spec in to_add {
            let id = md.next_col_id;
            md.next_col_id += 1;
            md.columns.push(ColumnMd {
                id,
                col_type: spec.col_type,
                is_pk: false,
                stored: true,
                value_expr: None,
                schema_version_add: new_version,
                schema_version_drop: None,
                media_validation: spec.media_validation,
                comment: spec.comment,
                custom_metadata: spec.custom_metadata,
                destination: None,
            });
            let sv = md
                .schema_versions
                .last_mut()
                .ok_or_else(|| Error::internal("table has no schema versions"))?;
            sv.columns.push(SchemaColumn { col_id: id, name: spec.name });
        }
        self.commit()?;
        Ok(names)
    }

    pub fn drop_column(&mut self, tbl_id: Uuid, name: &str, if_not_exists: IfNotExists) -> Result<()> {
        self.check_mutable(tbl_id, None, "drop_column()")?;
        let md = self.table(tbl_id)?.clone();
        let Some(col) = md.column_by_name(name, None)? else {
            if if_not_exists == IfNotExists::Ignore {
                return Ok(());
            }
            crate::bail!(ColumnNotFound, "Unknown column: {name}");
        };
        if col.md.is_pk {
            crate::bail!(UnsupportedOperation, "Cannot drop primary key column {}", py_repr(&col.name));
        }
        let dependents: Vec<String> = md
            .columns_at(None)?
            .into_iter()
            .filter(|c| {
                c.md.value_expr.as_ref().is_some_and(|e| {
                    crate::expr::serde::from_json(e)
                        .map(|e| e.column_refs().iter().any(|r| r.tbl_id == tbl_id && r.col_id == col.md.id))
                        .unwrap_or(false)
                })
            })
            .map(|c| c.name)
            .collect();
        if !dependents.is_empty() {
            crate::bail!(
                UnsupportedOperation,
                "Cannot drop column {} because the following columns depend on it:\n{}",
                py_repr(&col.name),
                dependents.join(", ")
            );
        }
        let new_version = self.bump_version(tbl_id, Some(format!("Dropped {}", py_repr(&col.name))))?;
        let md = self.table_mut(tbl_id)?;
        if let Some(c) = md.column_mut(col.md.id) {
            c.schema_version_drop = Some(new_version);
        }
        if let Some(sv) = md.schema_versions.last_mut() {
            sv.columns.retain(|c| c.col_id != col.md.id);
        }
        self.commit()
    }

    pub fn rename_column(&mut self, tbl_id: Uuid, old_name: &str, new_name: &str) -> Result<()> {
        self.check_mutable(tbl_id, None, "rename_column()")?;
        let md = self.table(tbl_id)?.clone();
        let Some(col) = md.column_by_name(old_name, None)? else {
            crate::bail!(ColumnNotFound, "Unknown column: {old_name}");
        };
        validate_column_name(new_name)?;
        let folded_new = fold_identifier(new_name);
        if md.column_by_name(&folded_new, None)?.is_some() {
            crate::bail!(ColumnAlreadyExists, "Column {} already exists", py_repr(new_name));
        }
        self.bump_version(tbl_id, Some(format!("Renamed {} to {}", py_repr(&col.name), py_repr(&folded_new))))?;
        let md = self.table_mut(tbl_id)?;
        if let Some(sv) = md.schema_versions.last_mut() {
            for c in sv.columns.iter_mut() {
                if c.col_id == col.md.id {
                    c.name = folded_new.clone();
                }
            }
        }
        self.commit()
    }
}
