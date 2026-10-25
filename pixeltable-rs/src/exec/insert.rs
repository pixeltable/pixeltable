//! Inserting rows.
//!
//! An insert runs in three steps (see `catalog::runtime` for why): `InsertPlan::new()` captures the target schema
//! under the lock; the bindings convert the Python rows into values and `build_rows()` evaluates computed columns,
//! both without the lock; `commit()` checks that the rows still fit the table (`check_current()`, else the bindings
//! plan again, see `exec::txn`), enforces the primary key, appends the rows at a new table version and persists them.

use std::collections::HashMap;
use uuid::Uuid;

use super::compute::{ComputedCol, MediaTarget, cell_error, discard_media, errors_in_order, eval_computed_batch, unstored_exprs};
use super::propagate::{InsertedViewRows, ViewNode, add_stats, commit_inserted_rows, compute_inserted_rows, record_version_status};
use super::scan::ElemInfo;
use crate::catalog::Catalog;
use crate::catalog::table::{ColId, MediaValidation, RowCountStats};
use crate::error::{Error, Result, py_repr};
use crate::media::{self, MediaKind};
use crate::expr::ExprRef;
use crate::store::rows::StoredRow;
use crate::store::table_data::{pk_hash, values_identical};
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
    /// the table as a path element, for evaluating the rows' view columns
    pub info: ElemInfo,
    /// the live views the inserted rows propagate to
    pub views: Vec<ViewNode>,
    /// the rowid the first inserted row is expected to get (rows are built with rowids from here; the commit shifts
    /// them if another insert came first)
    pub first_rowid: i64,
}

/// A column of the rows an insert or update returns (`return_rows`, upstream `RowBuilder.create_output_rows()`).
#[derive(Debug, Clone)]
pub struct ReturnCol {
    /// the column name; `<column>:<index>` for the value column of an index
    pub name: String,
    pub col_id: ColId,
    pub col_type: ColumnType,
    /// true if a cell error is reported (as `<name>:md`): computed and media columns, embedding index values
    pub records_errors: bool,
}

/// The stored columns of `tbl_id`'s own rows and the value columns of its indexes, as `return_rows` reports them.
pub fn return_columns(cat: &Catalog, tbl_id: Uuid) -> Result<Vec<ReturnCol>> {
    let md = cat.table(tbl_id)?;
    let mut out: Vec<ReturnCol> = md
        .columns_at(None)?
        .into_iter()
        .filter(|c| c.md.stored)
        .map(|c| ReturnCol {
            records_errors: c.md.is_computed() || c.md.col_type.is_media(),
            name: c.name,
            col_id: c.md.id,
            col_type: c.md.col_type,
        })
        .collect();
    for c in super::compute::index_value_columns(cat, tbl_id, None)? {
        let (Some(col), Some(idx)) = (md.column(c.col_id), md.index_of_value_col(c.col_id)) else { continue };
        out.push(ReturnCol { name: c.name, col_id: c.col_id, col_type: col.col_type.clone(), records_errors: !idx.is_btree() });
    }
    Ok(out)
}

/// Statistics of a completed operation (becomes a Python `UpdateStatus`).
#[derive(Debug, Clone, Default)]
pub struct UpdateStatus {
    pub updated_cols: Vec<String>,
    pub cols_with_excs: Vec<String>,
    pub row_count_stats: RowCountStats,
    pub cascade_row_count_stats: RowCountStats,
}

fn union(a: Vec<String>, b: Vec<String>) -> Vec<String> {
    let mut out = a;
    for x in b {
        if !out.contains(&x) {
            out.push(x);
        }
    }
    out
}

impl std::ops::Add for UpdateStatus {
    type Output = UpdateStatus;

    /// Upstream `UpdateStatus.__add__()`.
    fn add(self, other: UpdateStatus) -> UpdateStatus {
        UpdateStatus {
            updated_cols: union(self.updated_cols, other.updated_cols),
            cols_with_excs: union(self.cols_with_excs, other.cols_with_excs),
            row_count_stats: add_stats(&self.row_count_stats, &other.row_count_stats),
            cascade_row_count_stats: add_stats(&self.cascade_row_count_stats, &other.cascade_row_count_stats),
        }
    }
}

impl UpdateStatus {
    /// Upstream `to_cascade()`: the status of a change cascaded to another table.
    pub fn to_cascade(self) -> UpdateStatus {
        UpdateStatus {
            updated_cols: self.updated_cols,
            cols_with_excs: self.cols_with_excs,
            row_count_stats: RowCountStats::default(),
            cascade_row_count_stats: add_stats(&self.cascade_row_count_stats, &self.row_count_stats),
        }
    }

    /// Upstream `insert_to_update()`: rows (re)inserted by an update count as updated.
    pub fn insert_to_update(mut self) -> UpdateStatus {
        for s in [&mut self.row_count_stats, &mut self.cascade_row_count_stats] {
            s.upd_rows += s.ins_rows;
            s.ins_rows = 0;
        }
        self
    }
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
                crate::functions::udf::check_valid_column(cat, &tbl_display, &c.name, e, "inserted into")?;
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
            info: ElemInfo::of(md, None)?,
            views: ViewNode::tree(cat, tbl_id)?,
            first_rowid: md.next_rowid,
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

/// Rows ready to be stored (versions are assigned at commit): the table's rows (with the rowids the insert expects
/// 0, 1, ...) and the rows they add to the table's views.
pub struct BuiltRows {
    pub rows: Vec<StoredRow>,
    pub stats: RowCountStats,
    pub cols_with_excs: Vec<String>,
    /// media files written for these rows (the commit adopts them, see `Catalog::adopt_media()`)
    pub saved_media: Vec<std::path::PathBuf>,
    pub views: Vec<InsertedViewRows>,
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
                // raised as it is (upstream raises it from `ColumnType.validate_media()`, not as an evaluation failure)
                return Err(err);
            }
            stats.num_excs += 1;
            // `<table>.<column>`, as for computed columns
            let qualified = format!("{}.{}", plan.tbl_name, c.name);
            if !cols_with_excs.contains(&qualified) {
                cols_with_excs.push(qualified);
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
    // with fail_on_exception: the error of the first row whose media inputs failed (rows after it are not built)
    let mut media_error: Option<Error> = None;
    for (i, input) in inputs.into_iter().enumerate() {
        let mut row = StoredRow::new(plan.first_rowid + i as i64, Vec::new());
        for (col_id, v) in input {
            row.set_cell(col_id, v);
        }
        if let Err(err) = prepare_media_inputs(plan, &mut row, fail_on_exception, &mut stats, &mut cols_with_excs, Some(&mut *saved)) {
            if !fail_on_exception {
                return Err(err);
            }
            media_error = Some(err);
            break;
        }
        stats.ins_rows += 1;
        rows.push(row);
    }
    let first = eval_computed_batch(&mut rows, &computed, &plan.unstored, &plan.info, fail_on_exception, &mut stats, &mut cols_with_excs)?;
    if let Some((_, k, err)) = first {
        // the rows before a media failure are all built, so a computed column's failure comes first
        return Err(super::compute::eval_error(&computed[k].expr, err));
    }
    if let Some(err) = media_error {
        return Err(err);
    }
    for row in rows.iter_mut() {
        for c in computed.iter().filter(|c| c.is_media) {
            plan.media.materialize(row, c.col_id, saved)?;
        }
    }
    // the order in which sequential evaluation would have met the errors: per row, media inputs, then computed columns
    let mut error_cols: Vec<(ColId, String)> = plan
        .columns
        .iter()
        .filter(|c| c.media_validation.is_some() && !c.is_computed())
        .map(|c| (c.col_id, format!("{}.{}", plan.tbl_name, c.name)))
        .collect();
    error_cols.extend(computed.iter().map(|c| (c.col_id, c.name.clone())));
    let cols_with_excs = if cols_with_excs.is_empty() { cols_with_excs } else { errors_in_order(&rows, &error_cols) };
    let paths: Vec<Vec<&StoredRow>> = rows.iter().map(|r| vec![r]).collect();
    let views = compute_inserted_rows(&plan.views, &[&plan.info], &paths, saved)?;
    Ok(BuiltRows { rows, stats, cols_with_excs, saved_media: Vec::new(), views })
}

/// Postgres's maximum B-tree index row size, which bounds the combined size of a primary key value.
const MAX_INDEX_ROW_BYTES: usize = 2704;

fn pk_part(v: &Value, truncate: bool) -> Value {
    match v {
        // the primary key index of a data-versioned table holds truncated strings: strings that agree on the
        // prefix collide
        v if truncate => crate::functions::btree::truncate(v),
        v => v.clone(),
    }
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

fn pk_display(row: &StoredRow, plan: &InsertPlan, truncate: bool) -> String {
    use crate::functions::btree::MAX_STRING_LEN;
    let pk: Vec<&InsertColumn> = plan.columns.iter().filter(|c| c.is_pk).collect();
    // the key as Postgres reports it for upstream's index: on a data-versioned table, string columns are indexed by
    // their first `MAX_STRING_LEN` characters
    let truncated = |c: &InsertColumn| truncate && c.col_type.is_string();
    let names: Vec<String> = pk
        .iter()
        .map(|c| if truncated(c) { format!("\"left\"({}::text, {MAX_STRING_LEN})", c.name) } else { c.name.clone() })
        .collect();
    let vals: Vec<String> = pk
        .iter()
        .map(|c| match row.cell(c.col_id) {
            Value::String(s) if truncated(c) => s.chars().take(MAX_STRING_LEN).collect(),
            Value::String(s) => s.clone(),
            v => crate::expr::display::literal_str(v, &c.col_type),
        })
        .collect();
    format!("Key ({})=({}) already exists.", names.join(", "), vals.join(", "))
}

/// Appends the rows at a new table version and persists them.
/// Fails with SERIALIZATION_FAILURE (see `exec::txn`) if the rows built for `plan` no longer fit the table: its schema or
/// its views changed, or a sampled view decided membership by rowids that another insert has taken meanwhile.
pub fn check_current(cat: &Catalog, plan: &InsertPlan) -> Result<()> {
    let md = cat.table(plan.tbl_id)?;
    let rowids_moved = md.next_rowid != plan.first_rowid && plan.views.iter().any(ViewNode::samples);
    if md.current_schema_version != plan.schema_version || !ViewNode::unchanged(cat, plan.tbl_id, &plan.views)? || rowids_moved {
        return Err(super::txn::conflict(&format!("the insert into {}", py_repr(&plan.tbl_name))));
    }
    Ok(())
}

/// The input rows of `built` (rows built for `old`), to build again for `new`, the plan of the same insert after the
/// table changed (`check_current()` failed); None if they no longer fit: an input column is gone, changed its type or
/// became computed, or a required column is missing. Also returns the media files written for the inputs, which the
/// new rows keep; the other files of `built` (computed values) are deleted.
pub fn rebuild_inputs(old: &InsertPlan, new: &InsertPlan, built: BuiltRows) -> Option<(Vec<InputRow>, Vec<std::path::PathBuf>)> {
    let inputs: Vec<&InsertColumn> = old.columns.iter().filter(|c| !c.is_computed()).collect();
    let fits = inputs.iter().all(|c| new.columns.iter().any(|n| n.col_id == c.col_id && !n.is_computed() && n.col_type == c.col_type))
        && new.columns.iter().filter(|n| n.is_required()).all(|n| inputs.iter().any(|c| c.col_id == n.col_id))
        // an input whose validation failed was replaced by NULL
        && !built.rows.iter().any(|r| inputs.iter().any(|c| r.error(c.col_id).is_some()));
    if !fits {
        discard_media(&built.saved_media);
        return None;
    }
    let rows: Vec<InputRow> = built
        .rows
        .into_iter()
        .map(|mut r| {
            inputs
                .iter()
                .map(|c| (c.col_id, r.cells.get_mut(c.col_id as usize).map(|v| std::mem::replace(v, Value::Null)).unwrap_or(Value::Null)))
                .collect()
        })
        .collect();
    let kept: std::collections::HashSet<&std::path::Path> = rows
        .iter()
        .flatten()
        .filter_map(|(_, v)| match v {
            Value::Media(s) => Some(std::path::Path::new(s.as_str())),
            _ => None,
        })
        .collect();
    let (keep, discard): (Vec<_>, Vec<_>) = built.saved_media.into_iter().partition(|f| kept.contains(f.as_path()));
    discard_media(&discard);
    Some((rows, keep))
}

pub fn commit(cat: &mut Catalog, plan: &InsertPlan, built: BuiltRows) -> Result<UpdateStatus> {
    cat.adopt_media(built.saved_media.clone());
    check_current(cat, plan)?;
    let md = cat.table(plan.tbl_id)?.clone();
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
        // against the live rows (the table's primary-key index), and among the new rows
        let keys: Vec<Vec<Value>> = built.rows.iter().map(|r| pk_cols.iter().map(|c| r.cell(*c).clone()).collect()).collect();
        let existing = data.find_live_by_pks(&pk_cols, truncate, &keys, false);
        let mut seen: HashMap<u64, Vec<usize>> = HashMap::with_capacity(keys.len());
        for (i, r) in built.rows.iter().enumerate() {
            let h = pk_hash(keys[i].iter(), truncate);
            let dup_in_batch = seen.get(&h).is_some_and(|js| {
                js.iter().any(|j| keys[*j].iter().zip(&keys[i]).all(|(a, b)| values_identical(&pk_part(a, truncate), &pk_part(b, truncate))))
            });
            if existing[i].is_some() || dup_in_batch {
                crate::bail!(ConstraintViolation, "Duplicate primary key value: {}", pk_display(r, plan, truncate));
            }
            seen.entry(h).or_default().push(i);
        }
    }
    let version = if md.is_data_versioned { md.current_version + 1 } else { md.current_version };
    let rowid_offset = md.next_rowid - plan.first_rowid;
    let num_rows = built.rows.len() as i64;
    let rows: Vec<StoredRow> = built
        .rows
        .into_iter()
        .map(|mut r| {
            r.rowid += rowid_offset;
            r
        })
        .collect();
    data.append(rows, version);
    if md.is_data_versioned {
        cat.bump_version(plan.tbl_id, None)?;
    }
    cat.table_mut(plan.tbl_id)?.next_rowid = md.next_rowid + num_rows;
    let own = UpdateStatus {
        updated_cols: Vec::new(),
        cols_with_excs: built.cols_with_excs,
        row_count_stats: built.stats,
        cascade_row_count_stats: RowCountStats::default(),
    };
    let cascade = commit_inserted_rows(cat, &plan.views, built.views, (plan.tbl_id, version), rowid_offset)?;
    let status = own + cascade;
    record_version_status(cat, plan.tbl_id, &status)?;
    cat.write_data(plan.tbl_id, data)?;
    Ok(status)
}
