//! Inserting rows.
//!
//! An insert runs in three steps (see `catalog::runtime` for why): `InsertPlan::new()` captures the target schema
//! under the lock; the bindings convert the Python rows into values and `build_rows()` evaluates computed columns,
//! both without the lock; `commit()` re-checks the schema version, enforces the primary key, appends the rows at a
//! new table version and persists them.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use super::compute::{ComputedCol, MediaTarget, cell_error, discard_media, eval_computed, unstored_exprs};
use crate::catalog::Catalog;
use crate::catalog::table::{ColId, MediaValidation, RowCountStats};
use crate::error::{Error, Result, py_repr};
use crate::media::{self, MediaKind};
use crate::expr::ExprRef;
use crate::store::rows::{LIVE, StoredRow};
use crate::types::ColumnType;
use crate::value::Value;

#[derive(Debug, Clone)]
pub struct InsertColumn {
    pub col_id: ColId,
    pub name: String,
    pub col_type: ColumnType,
    pub value_expr: Option<ExprRef>,
    pub stored: bool,
    pub is_pk: bool,
    /// the effective validation mode of a media column
    pub media_validation: Option<MediaValidation>,
}

impl InsertColumn {
    pub fn is_computed(&self) -> bool {
        self.value_expr.is_some()
    }
    /// Required on insert: non-nullable and not computed.
    pub fn is_required(&self) -> bool {
        !self.col_type.nullable && !self.is_computed()
    }
}

#[derive(Debug, Clone)]
pub struct InsertPlan {
    pub tbl_id: Uuid,
    pub tbl_name: String,
    pub schema_version: i64,
    pub columns: Vec<InsertColumn>,
    pub unstored: HashMap<ColId, ExprRef>,
    /// where in-memory media inputs and results are written
    pub media: MediaTarget,
}

/// Statistics of a completed operation (becomes a Python `UpdateStatus`).
#[derive(Debug, Clone, Default)]
pub struct UpdateStatus {
    pub updated_cols: Vec<String>,
    pub cols_with_excs: Vec<String>,
    pub row_count_stats: RowCountStats,
    pub cascade_row_count_stats: RowCountStats,
}

impl InsertPlan {
    pub fn new(cat: &Catalog, tbl_id: Uuid, version: Option<i64>) -> Result<InsertPlan> {
        let md = cat.table(tbl_id)?;
        if md.is_view() || version.is_some() {
            let kind = if version.is_some() { "snapshot" } else { md.kind.display_name() };
            crate::bail!(UnsupportedOperation, "{}: Cannot insert into a {kind}.", cat.display_str(tbl_id)?);
        }
        let tbl_media_validation = md.current_schema()?.media_validation;
        let columns: Vec<InsertColumn> = md
            .columns_at(None)?
            .into_iter()
            .map(|c| {
                let media_validation = if c.md.col_type.is_media() {
                    Some(c.md.media_validation.unwrap_or(tbl_media_validation))
                } else {
                    None
                };
                Ok(InsertColumn {
                    col_id: c.md.id,
                    name: c.name,
                    col_type: c.md.col_type.clone(),
                    value_expr: c.md.value_expr.as_ref().map(crate::expr::serde::from_json).transpose()?,
                    stored: c.md.stored,
                    is_pk: c.md.is_pk,
                    media_validation,
                })
            })
            .collect::<Result<Vec<InsertColumn>>>()?;
        let tbl_display = crate::functions::udf::tbl_version_display(md);
        for c in columns.iter().filter(|c| c.stored) {
            if let Some(e) = &c.value_expr {
                crate::functions::udf::check_valid_column(&tbl_display, &c.name, e, "inserted into")?;
            }
        }
        let mut columns: Vec<InsertColumn> = columns;
        // the hidden value columns of embedding indexes are computed like stored computed columns
        for idx in &md.indexes.list {
            let Some(val_col_id) = idx.val_col_id() else { continue };
            let (Some(val_col), Some(indexed)) = (md.column(val_col_id), columns.iter().find(|c| c.col_id == idx.col_id))
            else {
                continue;
            };
            let name = format!("{}:{}", indexed.name, idx.name);
            columns.push(InsertColumn {
                col_id: val_col_id,
                name,
                col_type: val_col.col_type.clone(),
                value_expr: val_col.value_expr.as_ref().map(crate::expr::serde::from_json).transpose()?,
                stored: true,
                is_pk: false,
                media_validation: None,
            });
        }
        let version = if md.is_data_versioned { md.current_version + 1 } else { md.current_version };
        Ok(InsertPlan {
            tbl_id,
            tbl_name: md.name.clone(),
            schema_version: md.current_schema_version,
            columns,
            unstored: unstored_exprs(md)?,
            media: MediaTarget { tbl_id, dir: cat.store().media_dir(tbl_id), version },
        })
    }

    pub fn column(&self, name: &str) -> Option<&InsertColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The stored computed columns, in schema order (a computed column only references columns before it).
    pub fn computed_columns(&self) -> Vec<ComputedCol> {
        self.columns
            .iter()
            .filter(|c| c.stored)
            .filter_map(|c| {
                c.value_expr.as_ref().map(|e| ComputedCol {
                    col_id: c.col_id,
                    name: format!("{}.{}", self.tbl_name, c.name),
                    expr: e.clone(),
                    is_media: c.col_type.is_media(),
                })
            })
            .collect()
    }
}

/// A converted input row: a value for every column of the plan that was given (others are NULL or computed).
pub type InputRow = Vec<(ColId, Value)>;

/// Rows ready to be stored (rowids and versions are assigned at commit).
pub struct BuiltRows {
    pub rows: Vec<StoredRow>,
    pub stats: RowCountStats,
    pub cols_with_excs: Vec<String>,
    /// media files written for these rows (the commit adopts them, see `Catalog::adopt_media()`)
    pub saved_media: Vec<std::path::PathBuf>,
}

/// The error an insert with `on_error='abort'` raises for a media column whose file failed validation.
fn media_abort_error(col_name: &str, err: Error) -> Error {
    let msg = format!(
        "In row 0 the expression {col_name} encountered exception {}:\n{}",
        err.py_type_name(),
        err.message()
    );
    match err.code() {
        Some(code) => Error::new(code, msg),
        None => crate::pxt_err!(UnsupportedOperation, "{msg}"),
    }
}

/// Prepares the media inputs of a row: writes in-memory media to the media directory and validates the files of
/// write-validated columns. A failed validation is an error with `fail_on_exception`, else a NULL cell with the
/// error recorded.
/// `saved`: None to leave in-memory media in memory (`compute()`).
pub fn prepare_media_inputs(
    plan: &InsertPlan,
    row: &mut StoredRow,
    fail_on_exception: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
    mut saved: Option<&mut Vec<std::path::PathBuf>>,
) -> Result<()> {
    for c in plan.columns.iter().filter(|c| c.media_validation.is_some() && !c.is_computed()) {
        if let Some(saved) = saved.as_deref_mut() {
            plan.media.materialize(row, c.col_id, saved)?;
        }
        let (Some(MediaValidation::OnWrite), Some(kind)) = (c.media_validation, MediaKind::of(&c.col_type)) else {
            continue;
        };
        let Value::Media(stored) = row.cell(c.col_id).clone() else { continue };
        if let Err(err) = pyo3::Python::attach(|py| media::validate_stored(py, &stored, kind)) {
            if fail_on_exception {
                return Err(media_abort_error(&c.name, err));
            }
            stats.num_excs += 1;
            if !cols_with_excs.contains(&c.name) {
                cols_with_excs.push(c.name.clone());
            }
            row.set_cell(c.col_id, Value::Null);
            row.set_error(c.col_id, Some(cell_error(&err)));
        }
    }
    Ok(())
}

/// Evaluates computed columns for the input rows (no catalog lock held). Media files written along the way are
/// deleted again if this fails.
pub fn build_rows(plan: &InsertPlan, inputs: Vec<InputRow>, fail_on_exception: bool) -> Result<BuiltRows> {
    let mut saved = Vec::new();
    match build_rows_inner(plan, inputs, fail_on_exception, &mut saved) {
        Ok(mut built) => {
            built.saved_media = saved;
            Ok(built)
        }
        Err(e) => {
            discard_media(&saved);
            Err(e)
        }
    }
}

fn build_rows_inner(
    plan: &InsertPlan,
    inputs: Vec<InputRow>,
    fail_on_exception: bool,
    saved: &mut Vec<std::path::PathBuf>,
) -> Result<BuiltRows> {
    let computed = plan.computed_columns();
    let mut stats = RowCountStats::default();
    let mut cols_with_excs = Vec::new();
    let mut rows = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut row = StoredRow { rowid: 0, v_min: 0, v_max: LIVE, cells: Vec::new(), errors: Vec::new() };
        for (col_id, v) in input {
            row.set_cell(col_id, v);
        }
        prepare_media_inputs(plan, &mut row, fail_on_exception, &mut stats, &mut cols_with_excs, Some(&mut *saved))?;
        eval_computed(&mut row, &computed, Some(&plan.unstored), fail_on_exception, &mut stats, &mut cols_with_excs)?;
        for c in computed.iter().filter(|c| c.is_media) {
            plan.media.materialize(&mut row, c.col_id, saved)?;
        }
        stats.ins_rows += 1;
        rows.push(row);
    }
    Ok(BuiltRows { rows, stats, cols_with_excs, saved_media: Vec::new() })
}

/// Upstream `BtreeIndex.MAX_STRING_LEN`: the primary key index of a data-versioned table holds string values
/// truncated to this many characters, so strings that agree on that prefix collide.
const MAX_PK_STRING_LEN: usize = 256;
/// Postgres's maximum B-tree index row size, which bounds the combined size of a primary key value.
const MAX_INDEX_ROW_BYTES: usize = 2704;

fn pk_part(v: &Value, truncate: bool) -> Value {
    match v {
        Value::String(s) if truncate && s.chars().count() > MAX_PK_STRING_LEN => {
            Value::String(s.chars().take(MAX_PK_STRING_LEN).collect())
        }
        v => v.clone(),
    }
}

fn pk_key(row: &StoredRow, pk_cols: &[ColId], truncate: bool) -> String {
    pk_cols.iter().map(|c| format!("{:?}", pk_part(row.cell(*c), truncate))).collect::<Vec<_>>().join("\u{1f}")
}

/// The size of a row's primary key value in the index (strings by their UTF-8 length, other values 8 bytes).
fn pk_size(row: &StoredRow, pk_cols: &[ColId], truncate: bool) -> usize {
    pk_cols
        .iter()
        .map(|c| match pk_part(row.cell(*c), truncate) {
            Value::String(s) => s.len(),
            _ => 8,
        })
        .sum()
}

fn pk_display(row: &StoredRow, plan: &InsertPlan) -> String {
    let pk: Vec<&InsertColumn> = plan.columns.iter().filter(|c| c.is_pk).collect();
    let names: Vec<&str> = pk.iter().map(|c| c.name.as_str()).collect();
    let vals: Vec<String> = pk
        .iter()
        .map(|c| match row.cell(c.col_id) {
            Value::String(s) => s.clone(),
            v => crate::expr::display::literal_str(v, &c.col_type),
        })
        .collect();
    format!("Key ({})=({}) already exists.", names.join(", "), vals.join(", "))
}

/// Appends the rows at a new table version and persists them.
pub fn commit(cat: &mut Catalog, plan: &InsertPlan, built: BuiltRows) -> Result<UpdateStatus> {
    cat.adopt_media(built.saved_media.clone());
    let md = cat.table(plan.tbl_id)?.clone();
    if md.current_schema_version != plan.schema_version {
        crate::bail!(
            ConcurrentModification,
            "The schema of table {} changed while the insert was being prepared; retry the insert",
            py_repr(&plan.tbl_name)
        );
    }
    let mut data = cat.data(plan.tbl_id)?.as_ref().clone();
    let pk_cols: Vec<ColId> = plan.columns.iter().filter(|c| c.is_pk).map(|c| c.col_id).collect();
    if !pk_cols.is_empty() {
        let truncate = md.is_data_versioned;
        for r in &built.rows {
            if pk_size(r, &pk_cols, truncate) > MAX_INDEX_ROW_BYTES {
                let names: Vec<&str> = plan.columns.iter().filter(|c| c.is_pk).map(|c| c.name.as_str()).collect();
                crate::bail!(
                    ConstraintViolation,
                    "Primary key value too large for index: the combined size of the insert for columns ({}) exceeds the maximum btree index row size",
                    names.join(", ")
                );
            }
        }
        let mut seen: HashSet<String> =
            data.visible_at(md.current_version).map(|r| pk_key(r, &pk_cols, truncate)).collect();
        for r in &built.rows {
            if !seen.insert(pk_key(r, &pk_cols, truncate)) {
                crate::bail!(ConstraintViolation, "Duplicate primary key value: {}", pk_display(r, plan));
            }
        }
    }
    let version = if md.is_data_versioned { md.current_version + 1 } else { md.current_version };
    let mut next_rowid = md.next_rowid;
    for mut r in built.rows {
        r.rowid = next_rowid;
        r.v_min = version;
        next_rowid += 1;
        data.rows.push(r);
    }
    if md.is_data_versioned {
        cat.bump_version(plan.tbl_id, None)?;
    }
    {
        let md = cat.table_mut(plan.tbl_id)?;
        md.next_rowid = next_rowid;
        if let Some(vmd) = md.versions.iter_mut().find(|v| v.version == version) {
            vmd.row_count_stats.ins_rows += built.stats.ins_rows;
            vmd.row_count_stats.num_excs += built.stats.num_excs;
            vmd.row_count_stats.computed_values += built.stats.computed_values;
        }
    }
    cat.write_data(plan.tbl_id, data)?;
    Ok(UpdateStatus {
        updated_cols: Vec::new(),
        cols_with_excs: built.cols_with_excs,
        row_count_stats: built.stats,
        cascade_row_count_stats: RowCountStats::default(),
    })
}
