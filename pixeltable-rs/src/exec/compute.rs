//! Computed columns: evaluating value expressions for new rows and materializing a newly added column.
//!
//! A column whose expression may run user code is evaluated without the catalog lock: `AddComputedPlan::new()`
//! snapshots what is needed under the lock, `compute()` evaluates, and `commit()` checks that the rows and schemas read
//! are unchanged (else the operation is planned again, see `exec::txn`) and writes.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::catalog::table::{ColId, RowCountStats};
use crate::catalog::tables::{ColumnSpec, ExistingColumn};
use crate::catalog::{Catalog, IfExists, IfNotExists};
use crate::error::{Error, Result, py_repr};
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ColumnRef, ExprRef};
use crate::store::rows::{CellError, RowKey, StoredRow};
use crate::value::Value;

use super::insert::UpdateStatus;
use super::propagate::record_version_status;
use super::prefetch::CallSlot;
use super::scan::{PathInput, eval_rows_batch};
use super::txn::ReadSet;

/// A computed column to evaluate.
#[derive(Debug, Clone)]
pub struct ComputedCol {
    pub col_id: ColId,
    pub name: String,
    pub expr: ExprRef,
    /// true for media columns, whose in-memory results are written to the media directory when stored
    pub is_media: bool,
}

impl ComputedCol {
    /// A value of the column as the store holds it: an in-flight Python object in a column that is not a media column
    /// (the rows of a query function that hold images, `functions::udf`) is converted to the column's type, which
    /// fails for what has no stored form.
    pub fn storable(&self, v: Value) -> Result<Value> {
        match v {
            Value::PyObj(o) if !self.is_media => {
                pyo3::Python::attach(|py| crate::value::py::coerce(o.bind(py), &self.expr.col_type))
            }
            v => Ok(v),
        }
    }
}

/// Evaluates against the cells of a stored (or about-to-be-stored) row; unstored computed columns are evaluated
/// from their value expressions.
pub struct RowCells<'a> {
    pub row: &'a StoredRow,
    pub unstored: Option<&'a HashMap<ColId, ExprRef>>,
    /// the outcomes of prefetched calls for this row (`exec::prefetch`)
    pub calls: Option<CallSlot<'a>>,
}

impl<'a> RowCells<'a> {
    pub fn new(row: &'a StoredRow) -> Self {
        RowCells { row, unstored: None, calls: None }
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

    fn call_result(&self, e: &crate::expr::Expr) -> Option<Result<Value>> {
        self.calls?.get(e)
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

/// The hidden value columns of the indexes of `tbl_id` (current schema), as computed columns named
/// `<indexed column>:<index>`; with `on`, only those of indexes on a column in `on`.
pub fn index_value_columns(cat: &Catalog, tbl_id: Uuid, on: Option<&HashSet<(Uuid, ColId)>>) -> Result<Vec<ComputedCol>> {
    index_value_columns_md(cat.table(tbl_id)?, &cat.resolve_columns(tbl_id, None)?, on)
}

/// `index_value_columns()` of `md`, for the indexes on a column in `visible`.
pub fn index_value_columns_md(
    md: &crate::catalog::table::TableMd,
    visible: &[crate::catalog::table::ResolvedColumn],
    on: Option<&HashSet<(Uuid, ColId)>>,
) -> Result<Vec<ComputedCol>> {
    let mut out = Vec::new();
    for idx in &md.indexes.list {
        let Some(val_col_id) = idx.val_col_id() else { continue };
        if on.is_some_and(|on| !on.contains(&(idx.col_tbl_id, idx.col_id))) {
            continue;
        }
        let Some(indexed) = visible.iter().find(|c| c.tbl_id == idx.col_tbl_id && c.md.id == idx.col_id) else { continue };
        let Some(e) = md.column(val_col_id).and_then(|c| c.value_expr.as_ref()) else { continue };
        out.push(ComputedCol {
            col_id: val_col_id,
            name: format!("{}:{}", indexed.name, idx.name),
            expr: crate::expr::serde::from_json(e)?,
            is_media: false,
        });
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

/// The error an evaluation failure raises with `on_error='abort'` (upstream `raise_from_expr_eval_err()`): names
/// the node that failed (`expr` if the failure is not located) and the values of its inputs; a failure of a Python
/// aggregate's method carries the Python stack as the error's detail.
pub fn eval_error(expr: &ExprRef, err: Error) -> Error {
    use crate::error::EvalSite;
    use crate::expr::display::{display, print_value};
    let (what, inputs, err) = match err {
        Error::Located(l) if matches!(l.site, EvalSite::Setup) => return l.err,
        Error::Located(l) => {
            let l = *l;
            let call = |fc: &crate::expr::FunctionCall| crate::functions::display_call(fc);
            let what = match &l.site {
                EvalSite::Expr(e) => format!("expression {}", display(e)),
                EvalSite::AggInit(fc) => format!("init() function of the aggregate {}", call(fc)),
                EvalSite::AggUpdate(fc) => format!("update() function of the aggregate {}", call(fc)),
                EvalSite::AggValue(fc) => format!("expression {}", call(fc)),
                EvalSite::Setup => String::new(), // returned as it is above
            };
            let stack = !matches!(l.site, EvalSite::Expr(_));
            (what, l.inputs.iter().map(|(e, v)| format!("'{}' = {}", display(e), print_value(v, &e.col_type))).collect(), (l.err, stack))
        }
        err => (format!("expression {}", display(expr)), Vec::<String>::new(), (err, false)),
    };
    let (err, stack) = err;
    let mut msg = format!("In row 0 the {what} encountered exception {}:\n{}", err.py_type_name(), err.message());
    if !inputs.is_empty() {
        msg.push_str(&format!("\nwith {}", inputs.join(", ")));
    }
    let code = match &err {
        Error::Pxt { code, .. } => *code,
        Error::NotImplemented(_) => return err,
        _ => crate::error::ErrorCode::UnsupportedOperation,
    };
    if let (true, Error::Python(py_err)) = (stack, &err) {
        // the frames of the user's code, most recent first (upstream drops its own two evaluation frames)
        let detail = pyo3::Python::attach(|py| -> Option<String> {
            use pyo3::prelude::*;
            let tb = py_err.traceback(py)?;
            let frames: Vec<String> = py.import("traceback").ok()?.call_method1("format_tb", (tb,)).ok()?.extract().ok()?;
            (!frames.is_empty()).then(|| format!("Stack:\n{}", frames.iter().rev().cloned().collect::<Vec<_>>().join("\n")))
        });
        if let Some(detail) = detail {
            return pyo3::Python::attach(|py| crate::error::pxt_exception_with_detail(py, code, &msg, &detail, Some(py_err)));
        }
    }
    Error::new(code, msg)
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
    for k in 0..cols.len() {
        if let Some((k, err)) = eval_computed_row(row, cols, &[k], unstored, None, stats, cols_with_excs)? {
            if fail_on_exception {
                return Err(eval_error(&cols[k].expr, err));
            }
        }
    }
    Ok(())
}

/// Evaluates the columns `cols[k]`, `k` in `which` (in order), into `row`: a failure becomes a NULL cell with the
/// error recorded, as does a column that reads a cell with an error (it inherits the error). Returns the first
/// failure that is not inherited (its index in `cols` and the error); an interrupt (a BaseException that is not an
/// Exception: KeyboardInterrupt, SystemExit) is returned as an error.
pub fn eval_computed_row(
    row: &mut StoredRow,
    cols: &[ComputedCol],
    which: &[usize],
    unstored: Option<&HashMap<ColId, ExprRef>>,
    calls: Option<CallSlot<'_>>,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<Option<(usize, Error)>> {
    let mut first: Option<(usize, Error)> = None;
    for &k in which {
        let c = &cols[k];
        stats.computed_values += 1;
        let inherited = c.expr.column_refs().iter().find_map(|r| row.error(r.col_id).cloned());
        let outcome: std::result::Result<Value, CellError> = match inherited {
            Some(e) => Err(e),
            None => match eval(&c.expr, &RowCells { row, unstored, calls }).and_then(|v| c.storable(v)) {
                Ok(v) => Ok(v),
                Err(err) => {
                    if crate::functions::udf::is_interrupt(&err) {
                        return Err(err);
                    }
                    let cell = cell_error(&err);
                    if first.is_none() {
                        first = Some((k, err));
                    }
                    Err(cell)
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
    Ok(first)
}

/// The column ids `e` reads, also through the value expressions of derived (unstored computed) columns.
pub fn reads_columns(e: &ExprRef, derived: &dyn Fn(&ColumnRef) -> Option<ExprRef>, out: &mut HashSet<(Uuid, ColId)>) {
    for r in e.column_refs() {
        if out.insert((r.tbl_id, r.col_id)) {
            if let Some(x) = derived(r) {
                reads_columns(&x, derived, out);
            }
        }
    }
}

/// The evaluation waves of `cols` (evaluated in this order, a column only reads columns before it): each column is in
/// the wave after the latest wave of a column of `cols` it reads (directly or through derived columns), so the columns
/// of a wave do not read each other and their user-function calls can be dispatched together. Each wave lists indices
/// into `cols`, in order.
pub fn waves(cols: &[ComputedCol], derived: &dyn Fn(&ColumnRef) -> Option<ExprRef>) -> Vec<Vec<usize>> {
    let mut wave_of: Vec<usize> = Vec::with_capacity(cols.len());
    for (k, c) in cols.iter().enumerate() {
        let mut read = HashSet::new();
        reads_columns(&c.expr, derived, &mut read);
        let w = (0..k).filter(|j| read.iter().any(|(_, id)| *id == cols[*j].col_id)).map(|j| wave_of[j] + 1).max().unwrap_or(0);
        wave_of.push(w);
    }
    let n = wave_of.iter().copied().max().map_or(0, |w| w + 1);
    (0..n).map(|w| (0..cols.len()).filter(|k| wave_of[*k] == w).collect()).collect()
}

/// `eval_computed()` for a batch of new rows of table `info` (no failure is returned early): wave by wave (`waves()`),
/// the user-function calls of a wave's columns are made for all rows together (`prefetch`), then the rows are
/// evaluated in order; a column with window functions is evaluated over the whole batch in window order. With
/// `fail_on_exception`, rows after the first failing row are skipped in later waves. Returns the first failure in
/// sequential order (row, then column; inherited errors excluded) as (row, index in `cols`, error).
pub fn eval_computed_batch(
    rows: &mut [StoredRow],
    cols: &[ComputedCol],
    unstored: &HashMap<ColId, ExprRef>,
    info: &super::scan::ElemInfo,
    fail_on_exception: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<Option<(usize, usize, Error)>> {
    let derived = |c: &ColumnRef| if c.tbl_id == info.tbl_id { unstored.get(&c.col_id).cloned() } else { None };
    let mut first: Option<(usize, usize, Error)> = None;
    let mut limit = rows.len();
    let earlier = |first: &Option<(usize, usize, Error)>, r: usize, k: usize| first.as_ref().is_none_or(|(fr, fk, _)| (r, k) < (*fr, *fk));
    for wave in waves(cols, &derived) {
        let (windowed, plain): (Vec<usize>, Vec<usize>) =
            wave.into_iter().partition(|k| super::query::has_window_calls(&cols[*k].expr));
        if !plain.is_empty() {
            let exprs: Vec<&ExprRef> = plain.iter().map(|k| &cols[*k].expr).collect();
            let calls = {
                let rows: &[StoredRow] = rows;
                super::prefetch::prefetch(&exprs, &derived, limit, fail_on_exception, &|i, slot, f| {
                    f(&RowCells { row: &rows[i], unstored: Some(unstored), calls: Some(slot) })
                })?
            };
            let pure = super::par::worth_parallel(limit) && exprs.iter().all(|e| cells_expr_pure(e, unstored, &calls, 0));
            if pure {
                // rows are independent: evaluate them on several threads, then account as sequential evaluation would
                let calls_ref = &calls;
                let outcomes = super::par::map_mut(&mut rows[..limit], |i, row| -> Result<(RowCountStats, Option<(usize, Error)>)> {
                    let mut row_stats = RowCountStats::default();
                    let mut names = Vec::new();
                    let slot = calls_ref.as_ref().map(|results| CallSlot { results, row: i });
                    let failed = eval_computed_row(row, cols, &plain, Some(unstored), slot, &mut row_stats, &mut names)?;
                    Ok((row_stats, failed))
                });
                for (i, outcome) in outcomes.into_iter().enumerate() {
                    let (row_stats, failed) = outcome?;
                    *stats = super::propagate::add_stats(stats, &row_stats);
                    if let Some((k, err)) = failed {
                        if !cols_with_excs.contains(&cols[k].name) {
                            cols_with_excs.push(cols[k].name.clone());
                        }
                        if fail_on_exception && earlier(&first, i, k) {
                            first = Some((i, k, err));
                            limit = limit.min(i + 1);
                        }
                    }
                }
            }
            for i in 0..if pure { 0 } else { limit } {
                let slot = calls.as_ref().map(|results| CallSlot { results, row: i });
                if let Some((k, err)) = eval_computed_row(&mut rows[i], cols, &plain, Some(unstored), slot, stats, cols_with_excs)? {
                    if fail_on_exception {
                        if earlier(&first, i, k) {
                            first = Some((i, k, err));
                        }
                        limit = i + 1;
                        break;
                    }
                }
            }
        }
        for k in windowed {
            let col = &cols[k];
            let results = {
                let paths: Vec<Vec<&StoredRow>> = rows.iter().map(|r| vec![r]).collect();
                super::query::eval_rows_with_windows(&[info], &paths, &col.expr, true)?
            };
            for (i, (row, res)) in rows.iter_mut().zip(results).enumerate() {
                stats.computed_values += 1;
                match res {
                    Ok(v) => row.set_cell(col.col_id, v),
                    Err(err) => {
                        if crate::functions::udf::is_interrupt(&err) {
                            return Err(err);
                        }
                        stats.num_excs += 1;
                        if !cols_with_excs.contains(&col.name) {
                            cols_with_excs.push(col.name.clone());
                        }
                        row.set_cell(col.col_id, Value::Null);
                        row.set_error(col.col_id, Some(cell_error(&err)));
                        if fail_on_exception && earlier(&first, i, k) {
                            first = Some((i, k, err));
                            limit = limit.min(i + 1);
                        }
                    }
                }
            }
        }
    }
    Ok(first)
}

/// True if `e` can be evaluated for many new rows on several threads (`par::is_pure()`), with the calls in `calls` made
/// (cells are read as they are; unstored columns are evaluated).
fn cells_expr_pure(e: &ExprRef, unstored: &HashMap<ColId, ExprRef>, calls: &Option<super::prefetch::CallResults>, depth: usize) -> bool {
    let col_pure = |c: &ColumnRef| match unstored.get(&c.col_id) {
        Some(x) => depth < 64 && cells_expr_pure(x, unstored, calls, depth + 1),
        None => true,
    };
    super::par::is_pure(e, &col_pure, &|n| calls.as_ref().is_some_and(|r| r.complete(n)))
}

/// The names of the columns with errors in `rows`, in the order in which row-by-row evaluation meets them (row by row,
/// columns in the order of `cols`): what `cols_with_excs` holds after sequential evaluation.
pub fn errors_in_order(rows: &[StoredRow], cols: &[(ColId, String)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for row in rows {
        for (col_id, name) in cols {
            if row.error(*col_id).is_some() && !out.contains(name) {
                out.push(name.clone());
            }
        }
    }
    out
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
    /// placeholder column id used while evaluating (the real id is assigned at commit)
    pub eval_col_id: ColId,
    /// an existing column of the same name to drop at commit (`if_exists='replace'`)
    pub replace: Option<ColId>,
    pub media: MediaTarget,
    /// what the plan read (the table's path), validated by the commit
    pub read: ReadSet,
}

/// A computed value for the live row with the given key.
pub type KeyedValue = (RowKey, Value, Option<CellError>);

/// The result of `AddComputedPlan::compute_column()`.
pub struct ComputedColumn {
    pub values: Vec<KeyedValue>,
    pub stats: RowCountStats,
    pub cols_with_excs: Vec<String>,
    /// media files written for the values
    pub saved: Vec<std::path::PathBuf>,
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
        let media = MediaTarget { tbl_id, dir: cat.store().media_dir(tbl_id), version: md.current_version + 1 };
        let read = ReadSet::of(cat, path.iter().filter(|e| e.pinned.is_none()).map(|e| e.tbl_id))?;
        let path = PathInput::snapshot(cat, &path)?;
        Ok(Some(AddComputedPlan {
            tbl_id,
            tbl_name: md.name.clone(),
            schema_version: md.current_schema_version,
            eval_col_id: md.next_col_id,
            spec,
            expr,
            path,
            replace,
            media,
            read,
        }))
    }

    /// True if computing the column may run user code (so it must run without the catalog lock).
    pub fn runs_user_code(&self) -> bool {
        self.spec.stored.unwrap_or(true) && super::txn::runs_user_code(&self.expr, &self.path.infos())
    }

    /// `compute()`, collecting its results; media files written are deleted again if it fails.
    pub fn compute_column(&self, fail_on_exception: bool) -> Result<ComputedColumn> {
        let mut out = ComputedColumn { values: Vec::new(), stats: RowCountStats::default(), cols_with_excs: Vec::new(), saved: Vec::new() };
        match self.compute(fail_on_exception, &mut out.stats, &mut out.cols_with_excs, &mut out.saved) {
            Ok(values) => {
                out.values = values;
                Ok(out)
            }
            Err(e) => {
                discard_media(&out.saved);
                Err(e)
            }
        }
    }

    /// `commit()` of the result of `compute_column()`.
    pub fn commit_column(&self, cat: &mut Catalog, c: ComputedColumn) -> Result<UpdateStatus> {
        cat.adopt_media(c.saved);
        self.commit(cat, c.values, c.stats, c.cols_with_excs)
    }

    /// Evaluates the column for every live row. In-memory media results are written to the media directory and
    /// recorded in `saved` (the caller discards them if the operation fails).
    pub fn compute(
        &self,
        fail_on_exception: bool,
        stats: &mut RowCountStats,
        cols_with_excs: &mut Vec<String>,
        saved: &mut Vec<std::path::PathBuf>,
    ) -> Result<Vec<KeyedValue>> {
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
        let infos = self.path.infos();
        let path_rows = self.path.rows()?;
        if super::query::has_window_calls(&self.expr) {
            // window functions are evaluated over all visible rows in window order
            let results = super::query::eval_rows_with_windows(&infos, &path_rows, &self.expr, true)?;
            for (r, res) in path_rows.iter().zip(results) {
                stats.upd_rows += 1;
                stats.computed_values += 1;
                let key = (r[0].rowid, r[0].pos.clone());
                match res {
                    Ok(v) => out.push((key, v, None)),
                    Err(err) => {
                        if crate::functions::udf::is_interrupt(&err) {
                            return Err(err);
                        }
                        if fail_on_exception {
                            return Err(self.abort_error(&cell_error(&err)));
                        }
                        stats.num_excs += 1;
                        if !cols_with_excs.contains(&col.name) {
                            cols_with_excs.push(col.name.clone());
                        }
                        out.push((key, Value::Null, Some(cell_error(&err))));
                    }
                }
            }
            return Ok(out);
        }
        let cols = std::slice::from_ref(&col);
        let mut rows = super::scan::rows_for_eval(&path_rows, &infos, cols);
        let bases: Vec<&[&StoredRow]> = path_rows.iter().map(|r| &r[1..]).collect();
        stats.upd_rows += rows.len() as i64;
        eval_rows_batch(&mut rows, &bases, cols, &infos, &self.tbl_name, fail_on_exception, stats, cols_with_excs)?;
        if fail_on_exception {
            // the error of the first row that failed (also an error it inherits from a cell it reads)
            if let Some(err) = rows.iter().find_map(|r| r.error(col.col_id)) {
                return Err(self.abort_error(err));
            }
        }
        out.reserve(rows.len());
        for mut row in rows {
            if col.is_media {
                self.media.materialize(&mut row, col.col_id, saved)?;
            }
            let (v, err) = super::scan::take_cell(&mut row, col.col_id);
            out.push(((row.rowid, std::mem::take(&mut row.pos)), v, err));
        }
        Ok(out)
    }

    /// The error a failed evaluation raises with `on_error='abort'` (upstream `StoreBase.write_column()`): the
    /// failure of the row, an exception of the column's expression or one it inherits from a cell it reads.
    fn abort_error(&self, err: &CellError) -> Error {
        crate::pxt_err!(
            UnsupportedOperation,
            "Error while evaluating computed column {}:\n{}",
            py_repr(&self.spec.name),
            err.errormsg
        )
    }

    pub fn commit(
        &self,
        cat: &mut Catalog,
        values: Vec<KeyedValue>,
        mut stats: RowCountStats,
        mut cols_with_excs: Vec<String>,
    ) -> Result<UpdateStatus> {
        self.read.check(cat, &format!("the computation of column {}", py_repr(&self.spec.name)))?;
        if self.replace.is_some() {
            cat.drop_column(self.tbl_id, &self.spec.name, IfNotExists::Error)?;
        }
        let value_json = crate::expr::serde::to_json(&self.expr)?;
        let col_id = cat.push_column(self.tbl_id, self.spec.clone(), Some(value_json))?;
        let mut data = cat.data(self.tbl_id)?.as_ref().clone();
        // the value column of the column's default B-tree index holds its values, and inherits its errors
        let on_new_col: HashSet<(Uuid, ColId)> = [(self.tbl_id, col_id)].into_iter().collect();
        for idx_col in index_value_columns(cat, self.tbl_id, Some(&on_new_col))? {
            let idx_values: Vec<KeyedValue> =
                values.iter().map(|(k, v, e)| (k.clone(), crate::functions::btree::truncate(v), e.clone())).collect();
            stats.computed_values += idx_values.len() as i64;
            let num_excs = idx_values.iter().filter(|(_, _, e)| e.is_some()).count() as i64;
            if num_excs > 0 {
                stats.num_excs += num_excs;
                cols_with_excs.push(format!("{}.{}", self.tbl_name, idx_col.name));
            }
            data.set_live_cells(idx_col.col_id, idx_values);
        }
        data.set_live_cells(col_id, values);
        record_version_status(cat, self.tbl_id, &UpdateStatus { row_count_stats: stats, ..Default::default() })?;
        cat.write_data(self.tbl_id, data)?;
        Ok(UpdateStatus {
            updated_cols: vec![format!("{}.{}", self.tbl_name, self.spec.name)],
            cols_with_excs,
            row_count_stats: stats,
            cascade_row_count_stats: RowCountStats::default(),
        })
    }
}
