//! Computed columns: evaluating value expressions for new rows and materializing a newly added column.
//!
//! Evaluation runs without the catalog lock (it will call Python UDFs): `AddComputedPlan::new()` snapshots what is
//! needed under the lock, `compute()` evaluates, and `commit()` re-checks the schema version and writes.

use uuid::Uuid;

use crate::catalog::table::{ColId, RowCountStats};
use crate::catalog::tables::{ColumnSpec, ExistingColumn};
use crate::catalog::{Catalog, IfExists, IfNotExists};
use crate::error::{Error, Result, py_repr};
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ColumnRef, ExprRef};
use crate::store::rows::{CellError, StoredRow};
use crate::value::Value;

use super::insert::UpdateStatus;
use super::propagate::record_version_status;
use super::scan::{ElemInfo, PathCtx, PathInput};

/// A computed column to evaluate.
#[derive(Debug, Clone)]
pub struct ComputedCol {
    pub col_id: ColId,
    pub name: String,
    pub expr: ExprRef,
}

/// Evaluates against the cells of a stored (or about-to-be-stored) row of a base table; unstored computed columns
/// (`info.unstored`) are evaluated from their value expressions.
pub struct RowCells<'a> {
    pub row: &'a StoredRow,
    pub info: &'a ElemInfo,
}

impl RowContext for RowCells<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        match self.info.unstored.get(&col.col_id) {
            Some(e) => eval(e, self),
            None => Ok(self.row.cell(col.col_id).clone()),
        }
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
    info: &ElemInfo,
    cols: &[ComputedCol],
    fail_on_exception: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<()> {
    for c in cols {
        stats.computed_values += 1;
        match eval(&c.expr, &RowCells { row, info }) {
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
    pub tbl_name: String,
    pub schema_version: i64,
    pub spec: ColumnSpec,
    pub expr: ExprRef,
    /// the table's path at its current version (the rows to compute the column for)
    pub path: PathInput,
    /// an existing column of the same name to drop at commit (`if_exists='replace'`)
    pub replace: Option<ColId>,
}

/// A computed value for the row with the given key.
pub type KeyedValue = ((i64, Vec<i64>), Value, Option<CellError>);

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
        let path = cat.tbl_path(tbl_id, None)?;
        let visible: Vec<(Uuid, ColId)> = cat
            .path_columns(&path)?
            .iter()
            .filter(|c| replace != Some(c.md.id) || c.tbl_id != tbl_id)
            .map(|c| (c.tbl_id, c.md.id))
            .collect();
        let pins: Vec<(Uuid, Option<i64>)> = path.iter().map(|e| (e.tbl_id, e.pinned)).collect();
        let bound = expr
            .column_refs()
            .iter()
            .all(|r| visible.contains(&(r.tbl_id, r.col_id)) && pins.contains(&(r.tbl_id, r.tbl_version)));
        if !bound {
            crate::bail!(
                UnsupportedOperation,
                "The value expression of column {} ({}) is not bound by {}",
                py_repr(&spec.name),
                crate::expr::display::display(&expr),
                cat.display_str(tbl_id)?
            );
        }
        let path = PathInput::snapshot(cat, &path)?;
        Ok(Some(AddComputedPlan {
            tbl_id,
            tbl_name: md.name.clone(),
            schema_version: md.current_schema_version,
            spec,
            expr,
            path,
            replace,
        }))
    }

    /// Evaluates the column for every live row.
    pub fn compute(
        &self,
        fail_on_exception: bool,
        stats: &mut RowCountStats,
        cols_with_excs: &mut Vec<String>,
    ) -> Result<Vec<KeyedValue>> {
        let mut out = Vec::new();
        if !self.spec.stored.unwrap_or(true) {
            return Ok(out);
        }
        let qualified = format!("{}.{}", self.tbl_name, self.spec.name);
        let infos = self.path.infos();
        for r in self.path.rows()? {
            stats.upd_rows += 1;
            stats.computed_values += 1;
            let key = (r[0].rowid, r[0].pos.clone());
            match eval(&self.expr, &PathCtx { elems: &infos, rows: &r }) {
                Ok(v) => out.push((key, v, None)),
                Err(err) => {
                    if fail_on_exception {
                        return Err(eval_error(&self.expr, err));
                    }
                    stats.num_excs += 1;
                    if !cols_with_excs.contains(&qualified) {
                        cols_with_excs.push(qualified.clone());
                    }
                    out.push((key, Value::Null, Some(cell_error(&err))));
                }
            }
        }
        Ok(out)
    }

    pub fn commit(
        &self,
        cat: &mut Catalog,
        values: Vec<KeyedValue>,
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
        data.set_live_cells(col_id, values);
        record_version_status(
            cat,
            self.tbl_id,
            &UpdateStatus { row_count_stats: stats, ..Default::default() },
        )?;
        cat.write_data(self.tbl_id, data)?;
        Ok(UpdateStatus {
            updated_cols: vec![format!("{}.{}", self.tbl_name, self.spec.name)],
            cols_with_excs,
            row_count_stats: stats,
            cascade_row_count_stats: RowCountStats::default(),
        })
    }
}
