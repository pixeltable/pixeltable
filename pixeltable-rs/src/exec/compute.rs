//! Computed columns: evaluating value expressions for new rows and materializing a newly added column.
//!
//! Evaluation runs without the catalog lock (it will call Python UDFs): `AddComputedPlan::new()` snapshots what is
//! needed under the lock, `compute()` evaluates, and `commit()` re-checks the schema version and writes.

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::catalog::table::{ColId, RowCountStats};
use crate::catalog::tables::{ColumnSpec, ExistingColumn};
use crate::catalog::{Catalog, IfExists, IfNotExists};
use crate::error::{Error, Result, py_repr};
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ColumnRef, ExprRef};
use crate::store::rows::{CellError, StoredRow, TableData};
use crate::value::Value;

use super::insert::UpdateStatus;

/// A computed column to evaluate.
#[derive(Debug, Clone)]
pub struct ComputedCol {
    pub col_id: ColId,
    pub name: String,
    pub expr: ExprRef,
    /// true for media columns, whose in-memory results are written to the media directory when stored
    pub is_media: bool,
}

/// Evaluates against the cells of a stored (or about-to-be-stored) row; unstored computed columns are evaluated
/// from their value expressions.
pub struct RowCells<'a> {
    pub row: &'a StoredRow,
    pub unstored: Option<&'a HashMap<ColId, ExprRef>>,
}

impl<'a> RowCells<'a> {
    pub fn new(row: &'a StoredRow) -> Self {
        RowCells { row, unstored: None }
    }
}

impl RowContext for RowCells<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        if let Some(e) = self.unstored.and_then(|u| u.get(&col.col_id)) {
            return eval(e, self);
        }
        Ok(self.row.cell(col.col_id).clone())
    }

    fn column_error(&self, col: &ColumnRef) -> Result<Option<CellError>> {
        Ok(self.row.error(col.col_id).cloned())
    }

    fn python_semantics(&self) -> bool {
        true
    }
}

/// The value expressions of a table's unstored computed columns, by column id.
pub fn unstored_exprs(md: &crate::catalog::table::TableMd) -> Result<HashMap<ColId, ExprRef>> {
    let mut out = HashMap::new();
    for c in &md.columns {
        if !c.stored {
            if let Some(e) = &c.value_expr {
                out.insert(c.id, crate::expr::serde::from_json(e)?);
            }
        }
    }
    Ok(out)
}

/// Where in-memory media values of a table's columns are written (see `media::save_object()`).
#[derive(Debug, Clone)]
pub struct MediaTarget {
    pub tbl_id: Uuid,
    pub dir: std::path::PathBuf,
    /// the table version the values are written for (part of the file names)
    pub version: i64,
}

impl MediaTarget {
    /// Writes the cell of `col_id` to the media directory if it holds an in-memory media object; records the file.
    pub fn materialize(&self, row: &mut StoredRow, col_id: ColId, saved: &mut Vec<std::path::PathBuf>) -> Result<()> {
        if let Value::PyObj(o) = row.cell(col_id) {
            let o = o.clone();
            let path = pyo3::Python::attach(|py| {
                crate::media::save_object(o.bind(py), &self.dir, self.tbl_id, col_id, self.version)
            })?;
            saved.push(std::path::PathBuf::from(&path));
            row.set_cell(col_id, Value::Media(path));
        }
        Ok(())
    }
}

/// Deletes media files written by an operation that did not complete.
pub fn discard_media(files: &[std::path::PathBuf]) {
    for f in files {
        let _ = std::fs::remove_file(f);
    }
}

/// The error an evaluation failure raises with `on_error='abort'` (upstream `raise_from_expr_eval_err()`).
pub fn eval_error(expr: &ExprRef, err: Error) -> Error {
    let msg = format!(
        "In row 0 the expression {} encountered exception {}:\n{}",
        crate::expr::display::display(expr),
        err.py_type_name(),
        err.message()
    );
    match err {
        Error::Pxt { code, .. } => Error::new(code, msg),
        Error::NotImplemented(_) => err,
        _ => crate::pxt_err!(UnsupportedOperation, "{msg}"),
    }
}

pub fn cell_error(err: &Error) -> CellError {
    CellError { errortype: err.py_type_name(), errormsg: err.message() }
}

/// Evaluates `cols` (in order) into `row`. With `fail_on_exception`, the first failure is returned as an error;
/// otherwise failures become NULL cells with recorded errors. A column whose expression reads a cell that has an
/// error gets that error too (upstream propagates an exception to the dependent slots).
pub fn eval_computed(
    row: &mut StoredRow,
    cols: &[ComputedCol],
    unstored: Option<&HashMap<ColId, ExprRef>>,
    fail_on_exception: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<()> {
    for c in cols {
        stats.computed_values += 1;
        let inherited = c.expr.column_refs().iter().find_map(|r| row.error(r.col_id).cloned());
        let outcome: std::result::Result<Value, CellError> = match inherited {
            Some(e) => Err(e),
            None => match eval(&c.expr, &RowCells { row, unstored }) {
                Ok(v) => Ok(v),
                Err(err) => {
                    if fail_on_exception {
                        return Err(eval_error(&c.expr, err));
                    }
                    Err(cell_error(&err))
                }
            },
        };
        match outcome {
            Ok(v) => row.set_cell(c.col_id, v),
            Err(err) => {
                stats.num_excs += 1;
                if !cols_with_excs.contains(&c.name) {
                    cols_with_excs.push(c.name.clone());
                }
                row.set_cell(c.col_id, Value::Null);
                row.set_error(c.col_id, Some(err));
            }
        }
    }
    Ok(())
}

/// Everything `add_computed_column()` needs, captured under the lock.
pub struct AddComputedPlan {
    pub tbl_id: Uuid,
    pub schema_version: i64,
    pub spec: ColumnSpec,
    pub expr: ExprRef,
    pub data: Arc<TableData>,
    /// placeholder column id used while evaluating (the real id is assigned at commit)
    pub eval_col_id: ColId,
    /// an existing column of the same name to drop at commit (`if_exists='replace'`)
    pub replace: Option<ColId>,
    pub unstored: HashMap<ColId, ExprRef>,
    pub media: MediaTarget,
}

impl AddComputedPlan {
    /// None: the column exists and `if_exists='ignore'`.
    pub fn new(cat: &mut Catalog, tbl_id: Uuid, spec: ColumnSpec, if_exists: IfExists) -> Result<Option<AddComputedPlan>> {
        cat.check_mutable(tbl_id, None, "add columns to")?;
        let replace = match cat.check_existing_column(tbl_id, &spec.name, if_exists)? {
            ExistingColumn::Absent => None,
            ExistingColumn::Ignore => return Ok(None),
            ExistingColumn::Replace(col_id) => Some(col_id),
        };
        crate::catalog::tables::validate_column_spec(&spec)?;
        let expr = spec.value_expr.clone().ok_or_else(|| Error::internal("computed column without a value"))?;
        let md = cat.table(tbl_id)?.clone();
        let chain = cat.base_chain(tbl_id);
        let visible: Vec<(Uuid, ColId)> = cat
            .resolve_columns(tbl_id, None)?
            .iter()
            .filter(|c| replace != Some(c.md.id) || c.tbl_id != tbl_id)
            .map(|c| (c.tbl_id, c.md.id))
            .collect();
        let bound = expr.column_refs().iter().all(|r| chain.contains(&r.tbl_id) && visible.contains(&(r.tbl_id, r.col_id)));
        if !bound {
            crate::bail!(
                UnsupportedOperation,
                "The value expression of column {} ({}) is not bound by {}",
                py_repr(&spec.name),
                crate::expr::display::display(&expr),
                cat.display_str(tbl_id)?
            );
        }
        if md.is_view() {
            return Err(Error::not_implemented("computed columns on views"));
        }
        let data = cat.data(tbl_id)?;
        let media = MediaTarget { tbl_id, dir: cat.store().media_dir(tbl_id), version: md.current_version + 1 };
        Ok(Some(AddComputedPlan {
            tbl_id,
            schema_version: md.current_schema_version,
            eval_col_id: md.next_col_id,
            spec,
            expr,
            data,
            replace,
            unstored: unstored_exprs(&md)?,
            media,
        }))
    }

    /// Evaluates the column for every live row: (row index in `data.rows`, value, error). In-memory media results
    /// are written to the media directory and recorded in `saved` (the caller discards them if the operation fails).
    pub fn compute(
        &self,
        fail_on_exception: bool,
        stats: &mut RowCountStats,
        cols_with_excs: &mut Vec<String>,
        saved: &mut Vec<std::path::PathBuf>,
    ) -> Result<Vec<(usize, Value, Option<CellError>)>> {
        let mut out = Vec::new();
        if !self.spec.stored.unwrap_or(true) {
            return Ok(out);
        }
        let col = ComputedCol {
            col_id: self.eval_col_id,
            name: self.spec.name.clone(),
            expr: self.expr.clone(),
            is_media: self.spec.col_type.is_media(),
        };
        if super::query::has_window_calls(&self.expr) {
            // window functions are evaluated over all live rows in window order
            let live: Vec<(usize, &StoredRow)> =
                self.data.rows.iter().enumerate().filter(|(_, r)| r.v_max == crate::store::rows::LIVE).collect();
            let rows: Vec<&StoredRow> = live.iter().map(|(_, r)| *r).collect();
            let results = super::query::eval_rows_with_windows(&rows, &self.expr)?;
            for ((i, _), res) in live.iter().zip(results) {
                stats.upd_rows += 1;
                stats.computed_values += 1;
                match res {
                    Ok(v) => out.push((*i, v, None)),
                    Err(err) => {
                        if fail_on_exception {
                            return Err(eval_error(&self.expr, err));
                        }
                        stats.num_excs += 1;
                        if !cols_with_excs.contains(&col.name) {
                            cols_with_excs.push(col.name.clone());
                        }
                        out.push((*i, Value::Null, Some(cell_error(&err))));
                    }
                }
            }
            return Ok(out);
        }
        for (i, r) in self.data.rows.iter().enumerate() {
            if r.v_max != crate::store::rows::LIVE {
                continue;
            }
            let mut row = r.clone();
            stats.upd_rows += 1;
            eval_computed(&mut row, std::slice::from_ref(&col), Some(&self.unstored), fail_on_exception, stats, cols_with_excs)?;
            if col.is_media {
                self.media.materialize(&mut row, col.col_id, saved)?;
            }
            out.push((i, row.cell(col.col_id).clone(), row.error(col.col_id).cloned()));
        }
        Ok(out)
    }

    pub fn commit(
        &self,
        cat: &mut Catalog,
        values: Vec<(usize, Value, Option<CellError>)>,
        stats: RowCountStats,
        cols_with_excs: Vec<String>,
    ) -> Result<UpdateStatus> {
        if cat.table(self.tbl_id)?.current_schema_version != self.schema_version {
            crate::bail!(
                ConcurrentModification,
                "The schema of the table changed while column {} was being computed; retry",
                py_repr(&self.spec.name)
            );
        }
        if self.replace.is_some() {
            cat.drop_column(self.tbl_id, &self.spec.name, IfNotExists::Error)?;
        }
        let value_json = crate::expr::serde::to_json(&self.expr)?;
        let col_id = cat.push_column(self.tbl_id, self.spec.clone(), Some(value_json))?;
        let mut data = cat.data(self.tbl_id)?.as_ref().clone();
        for (i, v, err) in values {
            if let Some(r) = data.rows.get_mut(i) {
                r.set_cell(col_id, v);
                r.set_error(col_id, err);
            }
        }
        {
            let md = cat.table_mut(self.tbl_id)?;
            let version = md.current_version;
            if let Some(vmd) = md.versions.iter_mut().find(|v| v.version == version) {
                vmd.row_count_stats.upd_rows += stats.upd_rows;
                vmd.row_count_stats.num_excs += stats.num_excs;
                vmd.row_count_stats.computed_values += stats.computed_values;
            }
        }
        cat.write_data(self.tbl_id, data)?;
        Ok(UpdateStatus {
            updated_cols: vec![self.spec.name.clone()],
            cols_with_excs,
            row_count_stats: stats,
            cascade_row_count_stats: RowCountStats::default(),
        })
    }
}
