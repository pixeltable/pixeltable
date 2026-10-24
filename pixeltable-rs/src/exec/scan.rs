//! Scanning a table path: the rows of a table, or of a view joined with the rows of its bases.
//!
//! A view stores its own rows (its own columns, keyed by the key of the base row they belong to, see
//! `store::rows::StoredRow::key`), so a query over a view joins, for each visible view row, the base rows with the
//! matching key at the versions the path pins. `PathInput::snapshot()` captures the metadata and row data under the
//! catalog lock; `rows()` runs without it. `PathCtx` evaluates expressions against one joined row: a column
//! reference reads the cell of the row of its table (or evaluates the value expression of an unstored column).

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use super::compute::{ComputedCol, cell_error, eval_error};
use super::prefetch::CallSlot;
use crate::catalog::Catalog;
use crate::functions::iterator::IteratorCall;
use crate::catalog::table::{ColId, RowCountStats, TableMd};
use crate::catalog::views::PathElem;
use crate::error::{Error, Result};
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ColumnRef, ExprKind, ExprRef};
use crate::store::rows::{CellError, StoredRow, TableData};
use crate::value::Value;

/// What evaluation needs to know about one table of a path.
#[derive(Clone, Default)]
pub struct ElemInfo {
    pub tbl_id: Uuid,
    /// value expressions of the table's unstored computed columns (at the schema version read)
    pub unstored: HashMap<ColId, ExprRef>,
    /// the table's media columns that are validated on read
    pub read_validation: Option<Arc<crate::media::ReadValidation>>,
    /// component views: the iterator, whose `pos` column reads the row key and whose unstored outputs are produced
    /// again on read (`IteratorCall::seek_row`)
    pub iterator: Option<Arc<IterInfo>>,
}

/// The iterator of a component view, as a scan reads its columns.
pub struct IterInfo {
    pub call: Arc<IteratorCall>,
    /// the column of each value output (`call.value_outputs()`), with its output index and whether it is stored
    pub outputs: Vec<(ColId, usize, bool)>,
}

impl IterInfo {
    pub fn of(md: &TableMd) -> Result<Option<IterInfo>> {
        let Some(j) = md.view.as_ref().and_then(|v| v.iterator.as_ref()) else { return Ok(None) };
        let call = Arc::new(IteratorCall::from_json(j)?);
        let outputs = call.value_outputs().iter().enumerate().map(|(i, o)| ((i + 1) as ColId, i, o.is_stored)).collect();
        Ok(Some(IterInfo { call, outputs }))
    }
}

impl std::fmt::Debug for ElemInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ElemInfo({})", self.tbl_id)
    }
}

impl ElemInfo {
    /// The info of table `md` as of its version `version`.
    pub fn of(md: &TableMd, version: Option<i64>) -> Result<ElemInfo> {
        let mut unstored = HashMap::new();
        for c in md.columns_at(version)? {
            if !c.md.stored {
                if let Some(e) = &c.md.value_expr {
                    unstored.insert(c.md.id, crate::expr::serde::from_json(e)?);
                }
            }
        }
        Ok(ElemInfo {
            tbl_id: md.id,
            unstored,
            read_validation: Some(Arc::new(crate::media::ReadValidation::for_table(md)?)),
            iterator: IterInfo::of(md)?.map(Arc::new),
        })
    }

    /// The same table, with the unstored iterator outputs read from the row (while a row is being computed, all
    /// outputs are in its cells).
    pub fn with_materialized_outputs(&self) -> ElemInfo {
        let mut out = self.clone();
        if let Some(it) = &self.iterator {
            out.iterator = Some(Arc::new(IterInfo { call: it.call.clone(), outputs: it.outputs.iter().map(|(c, i, _)| (*c, *i, true)).collect() }));
        }
        out
    }

    /// For a read-validated media column with a file: the validation error, if any.
    pub(crate) fn read_validation_error(&self, row: &StoredRow, col_id: ColId) -> Option<Option<CellError>> {
        let rv = self.read_validation.as_ref()?;
        let kind = rv.kind(col_id)?;
        match row.cell(col_id) {
            Value::Media(s) => Some(rv.error(s, kind)),
            _ => Some(None),
        }
    }
}

/// One table of a path, captured for a scan.
pub struct ElemInput {
    pub info: ElemInfo,
    /// the version read
    pub version: i64,
    /// the version the path pins (None: live); column references of a pinned table carry it
    pub pinned: Option<i64>,
    pub data: Arc<TableData>,
    /// the number of iterator positions in this table's row keys
    pub num_pos: usize,
    /// the primary-key columns of a base table, with their types
    pub pk: Vec<(ColId, crate::types::ColumnType)>,
    /// true if the table's primary-key index truncates strings (data-versioned tables)
    pub pk_truncates: bool,
}

/// A table path (leaf first) captured for a scan.
pub struct PathInput {
    pub elems: Vec<ElemInput>,
}

impl PathInput {
    pub fn snapshot(cat: &mut Catalog, path: &[PathElem]) -> Result<PathInput> {
        let mut elems = Vec::with_capacity(path.len());
        for e in path {
            let md = cat.table(e.tbl_id)?.clone();
            let version = e.pinned.unwrap_or(md.current_version);
            let info = ElemInfo::of(&md, e.pinned)?;
            let data = cat.data(e.tbl_id)?;
            let pk = if md.is_view() {
                Vec::new()
            } else {
                md.columns.iter().filter(|c| c.is_pk).map(|c| (c.id, c.col_type.clone())).collect()
            };
            let pk_truncates = md.is_data_versioned;
            elems.push(ElemInput { info, version, pinned: e.pinned, data, num_pos: 0, pk, pk_truncates });
        }
        if elems.is_empty() {
            return Err(Error::internal("empty table path"));
        }
        // key lengths: one iterator position per component view from the element to the root
        let mut n = 0;
        for e in elems.iter_mut().rev() {
            if e.info.iterator.is_some() {
                n += 1;
            }
            e.num_pos = n;
        }
        Ok(PathInput { elems })
    }

    pub fn infos(&self) -> Vec<&ElemInfo> {
        self.elems.iter().map(|e| &e.info).collect()
    }

    /// The visible joined rows whose leaf rows have the given keys, in key order (keys without a visible row are
    /// skipped).
    pub fn rows_for_keys<'k>(&self, keys: impl IntoIterator<Item = (i64, &'k [i64])>) -> Vec<Vec<&StoredRow>> {
        let leaf = &self.elems[0];
        let mut leaf_rows: Vec<&StoredRow> = keys.into_iter().filter_map(|k| leaf.data.row_at(k, leaf.version)).collect();
        leaf_rows.sort_by(|a, b| a.key().cmp(&b.key()));
        leaf_rows.dedup_by(|a, b| a.key() == b.key());
        let mut out = Vec::with_capacity(leaf_rows.len());
        'rows: for r in leaf_rows {
            let mut joined = Vec::with_capacity(self.elems.len());
            joined.push(r);
            for e in &self.elems[1..] {
                let n = e.num_pos.min(r.pos.len());
                match e.data.row_at((r.rowid, &r.pos[..n]), e.version) {
                    Some(b) => joined.push(b),
                    None => continue 'rows,
                }
            }
            out.push(joined);
        }
        out
    }

    /// The rows a filter `pred` can select, a superset of those that pass it (the caller evaluates it): when `pred`
    /// fixes the primary key of a base table with equality (or `isin()`) conjuncts on literals, only the rows with
    /// such a key (found by the table's primary-key index); else all visible rows. In key order.
    pub fn rows_where(&self, pred: Option<&ExprRef>) -> Result<Vec<Vec<&StoredRow>>> {
        match pred.and_then(|p| self.pk_point_rowids(p)) {
            Some(rowids) => Ok(self.rows_for_keys(rowids.iter().map(|r| (*r, &[] as &[i64])))),
            None => self.rows(),
        }
    }

    /// For a filter that fixes the primary key of the path's table (a base table), the rowids of the rows that may
    /// pass it (from the primary-key index).
    fn pk_point_rowids(&self, pred: &ExprRef) -> Option<Vec<i64>> {
        let leaf = &self.elems[0];
        if self.elems.len() != 1 || leaf.pk.is_empty() {
            return None;
        }
        // per primary-key column, the values the filter allows
        let mut allowed: Vec<Option<Vec<Value>>> = vec![None; leaf.pk.len()];
        let mut todo = vec![pred.clone()];
        while let Some(e) = todo.pop() {
            match &e.kind {
                ExprKind::Compound { op: crate::expr::LogicalOp::And, operands } => todo.extend(operands.iter().cloned()),
                ExprKind::Comparison { op: crate::expr::CmpOp::Eq, lhs, rhs } => {
                    let (col, lit) = match (&lhs.kind, &rhs.kind) {
                        (ExprKind::ColumnRef(c), ExprKind::Literal(v)) | (ExprKind::Literal(v), ExprKind::ColumnRef(c)) => (c, v),
                        _ => continue,
                    };
                    if let Some(i) = self.pk_position(col, lit) {
                        allowed[i] = Some(vec![lit.clone()]);
                    }
                }
                ExprKind::InList { lhs, values, .. } => {
                    let ExprKind::ColumnRef(col) = &lhs.kind else { continue };
                    if let Some(i) = values.first().and_then(|v| self.pk_position(col, v)) {
                        if values.iter().all(|v| self.pk_position(col, v) == Some(i)) && values.len() <= 10_000 {
                            allowed[i] = Some(values.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        let allowed: Vec<Vec<Value>> = allowed.into_iter().collect::<Option<_>>()?;
        if allowed.iter().map(Vec::len).product::<usize>() > 10_000 {
            return None;
        }
        // every combination of allowed values
        let mut keys: Vec<Vec<Value>> = vec![Vec::new()];
        for vals in &allowed {
            keys = keys.into_iter().flat_map(|k| vals.iter().map(move |v| [k.clone(), vec![v.clone()]].concat())).collect();
        }
        let cols: Vec<ColId> = leaf.pk.iter().map(|(c, _)| *c).collect();
        let mut rowids: Vec<i64> = keys.iter().flat_map(|k| leaf.data.pk_candidates(&cols, leaf.pk_truncates, k)).collect();
        rowids.sort();
        rowids.dedup();
        Some(rowids)
    }

    /// The position of `col` among the primary-key columns of the path's table, if `col` is one and `v` is a value
    /// of its type that the index can look up (for others, SQL equality may hold between values of different kinds).
    fn pk_position(&self, col: &ColumnRef, v: &Value) -> Option<usize> {
        let leaf = &self.elems[0];
        if col.tbl_id != leaf.info.tbl_id {
            return None;
        }
        let i = leaf.pk.iter().position(|(c, _)| *c == col.col_id)?;
        let t = &leaf.pk[i].1;
        let fits = match v {
            Value::Int(_) => t.is_int(),
            Value::String(_) => t.is_string(),
            Value::Bool(_) => t.is_bool(),
            Value::Timestamp(_) => t.is_timestamp(),
            Value::Date(_) => t.is_date(),
            Value::Uuid(_) => t.is_uuid(),
            _ => false,
        };
        fits.then_some(i)
    }

    /// The visible joined rows, in key order: each entry has one row per table of the path (leaf first).
    pub fn rows(&self) -> Result<Vec<Vec<&StoredRow>>> {
        let leaf = &self.elems[0];
        let mut leaf_rows: Vec<&StoredRow> = leaf.data.visible_at(leaf.version).collect();
        leaf_rows.sort_by(|a, b| a.key().cmp(&b.key()));
        if self.elems.len() == 1 {
            return Ok(leaf_rows.into_iter().map(|r| vec![r]).collect());
        }
        // a base's rows by key: a map of all of them, unless looking up the few needed ones costs less
        let indexes: Vec<Option<HashMap<(i64, &[i64]), &StoredRow>>> = self.elems[1..]
            .iter()
            .map(|e| {
                (leaf_rows.len().saturating_mul(64) > e.data.num_live()).then(|| e.data.visible_at(e.version).map(|r| (r.key(), r)).collect())
            })
            .collect();
        let mut out = Vec::with_capacity(leaf_rows.len());
        'rows: for r in leaf_rows {
            let mut joined = Vec::with_capacity(self.elems.len());
            joined.push(r);
            for (i, idx) in indexes.iter().enumerate() {
                let e = &self.elems[i + 1];
                let n = e.num_pos.min(r.pos.len());
                let base = match idx {
                    Some(idx) => idx.get(&(r.rowid, &r.pos[..n])).copied(),
                    None => e.data.row_at((r.rowid, &r.pos[..n]), e.version),
                };
                match base {
                    Some(b) => joined.push(b),
                    // a view row whose base row is not visible at the base version read (cannot happen for a
                    // consistent path); skip it rather than fail the query
                    None => continue 'rows,
                }
            }
            out.push(joined);
        }
        Ok(out)
    }
}

/// Evaluation context for one joined row of a path: `rows[i]` is the row of table `elems[i]`. Query evaluation uses
/// SQL semantics; computed columns are materialized with Python semantics (`python`, see `RowContext`).
pub struct PathCtx<'a> {
    pub elems: &'a [&'a ElemInfo],
    pub rows: &'a [&'a StoredRow],
    pub python: bool,
    /// the outcomes of prefetched calls for this row (`exec::prefetch`)
    pub calls: Option<CallSlot<'a>>,
}

impl<'a> PathCtx<'a> {
    pub fn new(elems: &'a [&'a ElemInfo], rows: &'a [&'a StoredRow]) -> Self {
        PathCtx { elems, rows, python: false, calls: None }
    }

    fn elem(&self, col: &ColumnRef) -> Result<usize> {
        self.elems
            .iter()
            .position(|e| e.tbl_id == col.tbl_id)
            .ok_or_else(|| Error::internal(format!("column {} is not in the table path", col.name)))
    }
}

impl RowContext for PathCtx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        let i = self.elem(col)?;
        if let Some(e) = self.elems[i].unstored.get(&col.col_id) {
            return eval(e, self);
        }
        let Some(row) = self.rows.get(i) else { return Ok(Value::Null) };
        if let Some(it) = &self.elems[i].iterator {
            if col.col_id == 0 {
                // `pos`: the row's own iterator position
                return Ok(row.pos.last().map(|p| Value::Int(*p)).unwrap_or(Value::Null));
            }
            if let Some((_, idx, false)) = it.outputs.iter().find(|(c, _, _)| *c == col.col_id) {
                let stored: Vec<(String, Value)> = it
                    .outputs
                    .iter()
                    .filter(|(_, _, s)| *s)
                    .map(|(c, j, _)| (it.call.value_outputs()[*j].name.clone(), row.cell(*c).clone()))
                    .collect();
                let base = PathCtx { elems: &self.elems[i + 1..], rows: &self.rows[i + 1..], python: self.python, calls: self.calls };
                let pos = row.pos.last().copied().unwrap_or(0);
                let values = it.call.seek_row(&base, pos, &stored)?;
                return Ok(values.get(*idx).cloned().unwrap_or(Value::Null));
            }
        }
        if let Some(Some(_)) = self.elems[i].read_validation_error(row, col.col_id) {
            // an invalid file reads as NULL; its error is the cell's error
            return Ok(Value::Null);
        }
        Ok(row.cell(col.col_id).clone())
    }

    fn column_error(&self, col: &ColumnRef) -> Result<Option<CellError>> {
        let i = self.elem(col)?;
        let Some(row) = self.rows.get(i) else { return Ok(None) };
        if let Some(e) = self.elems[i].read_validation_error(row, col.col_id) {
            return Ok(e);
        }
        Ok(row.error(col.col_id).cloned())
    }

    fn python_semantics(&self) -> bool {
        self.python
    }

    fn row_key(&self, tbl_id: Uuid) -> Result<Option<(i64, Vec<i64>)>> {
        let Some(i) = self.elems.iter().position(|e| e.tbl_id == tbl_id) else {
            return Err(Error::internal(format!("table {tbl_id} is not in the table path")));
        };
        Ok(self.rows.get(i).map(|r| (r.rowid, r.pos.clone())))
    }

    fn call_result(&self, e: &crate::expr::Expr) -> Option<Result<Value>> {
        self.calls?.get(e)
    }
}

/// Where `eval_into()` records evaluation results and failures.
pub struct EvalSink<'a> {
    /// the table name `cols_with_excs` qualifies column names with
    pub tbl_name: &'a str,
    /// true: an evaluation failure is an error, else a NULL cell with the error recorded
    pub fail_on_exception: bool,
    pub stats: &'a mut RowCountStats,
    pub cols_with_excs: &'a mut Vec<String>,
}

impl EvalSink<'_> {
    /// Stores the outcome of evaluating column `c` of `row`; a failure, or a value the column cannot store, becomes a
    /// NULL cell with the error recorded and is returned (with `fail_on_exception` it is returned as the abort error
    /// instead, and an interrupt is always returned as an error).
    fn record(&mut self, row: &mut StoredRow, c: &ComputedCol, outcome: Result<Value>) -> Result<Option<Error>> {
        self.stats.computed_values += 1;
        match outcome.and_then(|v| c.storable(v)) {
            Ok(v) => {
                row.set_cell(c.col_id, v);
                row.set_error(c.col_id, None);
                Ok(None)
            }
            Err(err) => {
                if crate::functions::udf::is_interrupt(&err) {
                    // a BaseException that is not an Exception (KeyboardInterrupt, SystemExit) aborts as is
                    return Err(err);
                }
                if self.fail_on_exception {
                    return Err(eval_error(&c.expr, err));
                }
                self.store_error(row, c, cell_error(&err));
                Ok(Some(err))
            }
        }
    }

    fn store_error(&mut self, row: &mut StoredRow, c: &ComputedCol, err: CellError) {
        self.stats.num_excs += 1;
        let qualified = format!("{}.{}", self.tbl_name, c.name);
        if !self.cols_with_excs.contains(&qualified) {
            self.cols_with_excs.push(qualified);
        }
        row.set_cell(c.col_id, Value::Null);
        row.set_error(c.col_id, Some(err));
    }
}

/// Evaluates the computed columns `cols` (in order) into `row`, a row of table `infos[0]` whose base path rows are
/// `base` (rows of `infos[1..]`), with upstream's materialization semantics: a column whose expression reads a cell
/// that this evaluation set to an error gets that error; an evaluation failure is an error with `fail_on_exception`,
/// else a NULL cell with the error recorded (and counted in `stats`, the column named `<tbl_name>.<col>` in
/// `cols_with_excs`).
#[allow(clippy::too_many_arguments)]
pub fn eval_into(
    row: &mut StoredRow,
    cols: &[ComputedCol],
    infos: &[&ElemInfo],
    base: &[&StoredRow],
    tbl_name: &str,
    fail_on_exception: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<()> {
    let mut sink = EvalSink { tbl_name, fail_on_exception, stats, cols_with_excs };
    for k in 0..cols.len() {
        eval_col_into(row, cols, k, infos, base, &mut sink, None)?;
    }
    Ok(())
}

/// Evaluates `cols[k]` into `row` (see `eval_into()`); the columns of `cols[..k]` it reads were evaluated into it
/// before. Returns the failure it recorded, if any (not an inherited error).
fn eval_col_into(
    row: &mut StoredRow,
    cols: &[ComputedCol],
    k: usize,
    infos: &[&ElemInfo],
    base: &[&StoredRow],
    sink: &mut EvalSink<'_>,
    calls: Option<CallSlot<'_>>,
) -> Result<Option<Error>> {
    let c = &cols[k];
    let outcome = {
        let mut rows: Vec<&StoredRow> = Vec::with_capacity(base.len() + 1);
        rows.push(row);
        rows.extend_from_slice(base);
        let ctx = PathCtx { elems: infos, rows: &rows, python: true, calls };
        match inherited_error(&ctx, cols, k)? {
            Some(e) => Err(e),
            None => Ok(eval(&c.expr, &ctx)),
        }
    };
    match outcome {
        Ok(result) => sink.record(row, c, result),
        Err(inherited) => {
            sink.stats.computed_values += 1;
            sink.store_error(row, c, inherited);
            Ok(None)
        }
    }
}

/// The error `cols[k]` inherits from the cells it reads in `ctx`: errors raised by this evaluation (earlier columns of
/// `cols`, media validated on read) propagate; an error stored earlier does not (its cell reads as NULL, as upstream
/// reads it from the store).
fn inherited_error(ctx: &PathCtx<'_>, cols: &[ComputedCol], k: usize) -> Result<Option<CellError>> {
    for r in cols[k].expr.column_refs() {
        let i = ctx.elem(r)?;
        let Some(r_row) = ctx.rows.get(i) else { continue };
        let err = if i == 0 && cols[..k].iter().any(|p| p.col_id == r.col_id) {
            r_row.error(r.col_id).cloned()
        } else {
            ctx.elems[i].read_validation_error(r_row, r.col_id).flatten()
        };
        if err.is_some() {
            return Ok(err);
        }
    }
    Ok(None)
}

/// `eval_into()` for a batch of rows of table `infos[0]` (`rows[i]` belongs to the base path rows `bases[i]`); with
/// `fail_on_exception`, the first failure in sequential order (row, then column) is the error. See
/// `eval_rows_batch()`.
#[allow(clippy::too_many_arguments)]
pub fn eval_rows_into(
    rows: &mut [StoredRow],
    bases: &[&[&StoredRow]],
    cols: &[ComputedCol],
    infos: &[&ElemInfo],
    tbl_name: &str,
    fail_on_exception: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<()> {
    match eval_rows_batch(rows, bases, cols, infos, tbl_name, fail_on_exception, stats, cols_with_excs)? {
        Some((_, k, err)) if fail_on_exception => Err(eval_error(&cols[k].expr, err)),
        _ => Ok(()),
    }
}

/// Evaluates the computed columns `cols` into `rows` (rows of table `infos[0]`; `rows[i]` belongs to the base path
/// rows `bases[i]`), recording failures as cell errors (`eval_into()` semantics). Wave by wave (`compute::waves()`),
/// the user-function calls of a wave's columns are made for all rows together (`prefetch`), then the rows are
/// evaluated in order; a column with window functions is evaluated over the whole batch in window order (upstream
/// evaluates computed columns with a query plan over the rows being inserted or updated). With `abort`, rows after
/// the first failing row are skipped in later waves. Returns the first failure in sequential order (row, then column;
/// inherited errors excluded) as (row, index in `cols`, error). `cols_with_excs` gets the failing columns in the
/// order row-by-row evaluation meets them.
#[allow(clippy::too_many_arguments)]
pub fn eval_rows_batch(
    rows: &mut [StoredRow],
    bases: &[&[&StoredRow]],
    cols: &[ComputedCol],
    infos: &[&ElemInfo],
    tbl_name: &str,
    abort: bool,
    stats: &mut RowCountStats,
    cols_with_excs: &mut Vec<String>,
) -> Result<Option<(usize, usize, Error)>> {
    let known_errors = cols_with_excs.len();
    let derived = |c: &ColumnRef| infos.iter().find(|e| e.tbl_id == c.tbl_id).and_then(|e| e.unstored.get(&c.col_id).cloned());
    let mut sink = EvalSink { tbl_name, fail_on_exception: false, stats, cols_with_excs };
    let mut first: Option<(usize, usize, Error)> = None;
    let mut limit = rows.len();
    let earlier = |first: &Option<(usize, usize, Error)>, r: usize, k: usize| first.as_ref().is_none_or(|(fr, fk, _)| (r, k) < (*fr, *fk));
    for wave in super::compute::waves(cols, &derived) {
        let (windowed, plain): (Vec<usize>, Vec<usize>) =
            wave.into_iter().partition(|k| super::query::has_window_calls(&cols[*k].expr));
        if !plain.is_empty() {
            let exprs: Vec<&ExprRef> = plain.iter().map(|k| &cols[*k].expr).collect();
            let calls = {
                let rows: &[StoredRow] = rows;
                super::prefetch::prefetch(&exprs, &derived, limit, abort, &|i, slot, f| {
                    let path: Vec<&StoredRow> = std::iter::once(&rows[i]).chain(bases[i].iter().copied()).collect();
                    f(&PathCtx { elems: infos, rows: &path, python: true, calls: Some(slot) })
                })?
            };
            'rows: for i in 0..limit {
                let slot = calls.as_ref().map(|results| CallSlot { results, row: i });
                for &k in &plain {
                    if let Some(err) = eval_col_into(&mut rows[i], cols, k, infos, bases[i], &mut sink, slot)? {
                        if abort {
                            if earlier(&first, i, k) {
                                first = Some((i, k, err));
                            }
                            limit = i + 1;
                            // the rest of this row's columns are evaluated (an earlier column may fail in a later wave)
                            for &k2 in plain.iter().filter(|k2| **k2 > k) {
                                eval_col_into(&mut rows[i], cols, k2, infos, bases[i], &mut sink, slot)?;
                            }
                            break 'rows;
                        }
                    }
                }
            }
        }
        for k in windowed {
            let c = &cols[k];
            let results = {
                let paths: Vec<Vec<&StoredRow>> =
                    rows.iter().zip(bases).map(|(r, b)| std::iter::once(r).chain(b.iter().copied()).collect()).collect();
                super::query::eval_rows_with_windows(infos, &paths, &c.expr, true)?
            };
            for (i, (row, result)) in rows.iter_mut().zip(results).enumerate() {
                if let Some(err) = sink.record(row, c, result)? {
                    if abort && earlier(&first, i, k) {
                        first = Some((i, k, err));
                        limit = limit.min(i + 1);
                    }
                }
            }
        }
    }
    if cols_with_excs.len() > known_errors {
        let names: Vec<(ColId, String)> = cols.iter().map(|c| (c.col_id, format!("{tbl_name}.{}", c.name))).collect();
        let ordered = super::compute::errors_in_order(rows, &names);
        cols_with_excs.truncate(known_errors);
        for n in ordered {
            if !cols_with_excs.contains(&n) {
                cols_with_excs.push(n);
            }
        }
    }
    Ok(first)
}
