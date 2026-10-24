//! Computed columns: evaluating value expressions for new rows and materializing a newly added column.
//!
//! Evaluation runs without the catalog lock (it will call Python UDFs): `AddComputedPlan::new()` snapshots what is
//! needed under the lock, `compute()` evaluates, and `commit()` re-checks the schema version and writes.

use std::sync::Arc;
use uuid::Uuid;

use crate::catalog::Catalog;
use crate::catalog::table::{ColId, RowCountStats};
use crate::catalog::tables::ColumnSpec;
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
}

/// Evaluates against the cells of a stored (or about-to-be-stored) row.
pub struct RowCells<'a> {
    pub row: &'a StoredRow,
}

impl RowContext for RowCells<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        Ok(self.row.cell(col.col_id).clone())
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
/// otherwise failures become NULL cells with recorded errors.
pub fn eval_computed(
    row: &mut StoredRow,
    cols: &[ComputedCol],
    fail_on_exception: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<()> {
    for c in cols {
        stats.computed_values += 1;
        match eval(&c.expr, &RowCells { row }) {
            Ok(v) => row.set_cell(c.col_id, v),
            Err(err) => {
                if fail_on_exception {
                    return Err(eval_error(&c.expr, err));
                }
                stats.num_excs += 1;
                if !cols_with_excs.contains(&c.name) {
                    cols_with_excs.push(c.name.clone());
                }
                row.set_cell(c.col_id, Value::Null);
                row.set_error(c.col_id, Some(cell_error(&err)));
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
}

impl AddComputedPlan {
    pub fn new(cat: &mut Catalog, tbl_id: Uuid, spec: ColumnSpec) -> Result<AddComputedPlan> {
        let expr = spec.value_expr.clone().ok_or_else(|| Error::internal("computed column without a value"))?;
        let md = cat.table(tbl_id)?.clone();
        let chain = cat.base_chain(tbl_id);
        let visible: Vec<(Uuid, ColId)> =
            cat.resolve_columns(tbl_id, None)?.iter().map(|c| (c.tbl_id, c.md.id)).collect();
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
        Ok(AddComputedPlan {
            tbl_id,
            schema_version: md.current_schema_version,
            eval_col_id: md.next_col_id,
            spec,
            expr,
            data,
        })
    }

    /// Evaluates the column for every live row: (row index in `data.rows`, value, error).
    pub fn compute(
        &self,
        fail_on_exception: bool,
        stats: &mut RowCountStats,
        cols_with_excs: &mut Vec<String>,
    ) -> Result<Vec<(usize, Value, Option<CellError>)>> {
        let mut out = Vec::new();
        if !self.spec.stored.unwrap_or(true) {
            return Ok(out);
        }
        let col = ComputedCol { col_id: self.eval_col_id, name: self.spec.name.clone(), expr: self.expr.clone() };
        for (i, r) in self.data.rows.iter().enumerate() {
            if r.v_max != crate::store::rows::LIVE {
                continue;
            }
            let mut row = r.clone();
            stats.upd_rows += 1;
            eval_computed(&mut row, std::slice::from_ref(&col), fail_on_exception, stats, cols_with_excs)?;
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
