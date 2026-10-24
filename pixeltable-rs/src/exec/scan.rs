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
use crate::catalog::Catalog;
use crate::functions::iterator::IteratorCall;
use crate::catalog::table::{ColId, RowCountStats, TableMd};
use crate::catalog::views::PathElem;
use crate::error::{Error, Result};
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ColumnRef, ExprRef};
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
            elems.push(ElemInput { info, version, pinned: e.pinned, data, num_pos: 0 });
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

    /// The visible joined rows, in key order: each entry has one row per table of the path (leaf first).
    pub fn rows(&self) -> Result<Vec<Vec<&StoredRow>>> {
        let leaf = &self.elems[0];
        let mut leaf_rows: Vec<&StoredRow> = leaf.data.visible_at(leaf.version).collect();
        leaf_rows.sort_by(|a, b| a.key().cmp(&b.key()));
        if self.elems.len() == 1 {
            return Ok(leaf_rows.into_iter().map(|r| vec![r]).collect());
        }
        let indexes: Vec<HashMap<(i64, &[i64]), &StoredRow>> = self.elems[1..]
            .iter()
            .map(|e| e.data.visible_at(e.version).map(|r| (r.key(), r)).collect())
            .collect();
        let mut out = Vec::with_capacity(leaf_rows.len());
        'rows: for r in leaf_rows {
            let mut joined = Vec::with_capacity(self.elems.len());
            joined.push(r);
            for (i, idx) in indexes.iter().enumerate() {
                let n = self.elems[i + 1].num_pos.min(r.pos.len());
                match idx.get(&(r.rowid, &r.pos[..n])) {
                    Some(b) => joined.push(*b),
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
}

impl<'a> PathCtx<'a> {
    pub fn new(elems: &'a [&'a ElemInfo], rows: &'a [&'a StoredRow]) -> Self {
        PathCtx { elems, rows, python: false }
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
                let base = PathCtx { elems: &self.elems[i + 1..], rows: &self.rows[i + 1..], python: self.python };
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
}

/// Evaluates the computed columns `cols` (in order) into `row`, a row of table `infos[0]` whose base path rows are
/// `base` (rows of `infos[1..]`), with upstream's materialization semantics: a column whose expression reads a cell
/// that this evaluation set to an error gets that error; an evaluation failure is an error with `fail_on_exception`, else a NULL cell
/// with the error recorded (and counted in `stats`, the column named `<tbl_name>.<col>` in `cols_with_excs`).
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
    for (k, c) in cols.iter().enumerate() {
        stats.computed_values += 1;
        let outcome: std::result::Result<Value, CellError> = {
            let mut rows: Vec<&StoredRow> = Vec::with_capacity(base.len() + 1);
            rows.push(row);
            rows.extend_from_slice(base);
            let ctx = PathCtx { elems: infos, rows: &rows, python: true };
            let mut inherited = None;
            for r in c.expr.column_refs() {
                let i = ctx.elem(r)?;
                let Some(r_row) = rows.get(i) else { continue };
                // errors raised by this evaluation (earlier columns of `cols`, media validated on read) propagate;
                // an error stored earlier does not (its cell reads as NULL, as upstream reads it from the store)
                let err = if i == 0 && cols[..k].iter().any(|p| p.col_id == r.col_id) {
                    r_row.error(r.col_id).cloned()
                } else {
                    infos[i].read_validation_error(r_row, r.col_id).flatten()
                };
                if let Some(e) = err {
                    inherited = Some(e);
                    break;
                }
            }
            match inherited {
                Some(e) => Err(e),
                None => match eval(&c.expr, &ctx).and_then(|v| c.storable(v)) {
                    Ok(v) => Ok(v),
                    Err(err) => {
                        if crate::functions::udf::is_interrupt(&err) {
                            // a BaseException that is not an Exception (KeyboardInterrupt, SystemExit) aborts as is
                            return Err(err);
                        }
                        if fail_on_exception {
                            return Err(eval_error(&c.expr, err));
                        }
                        Err(cell_error(&err))
                    }
                },
            }
        };
        match outcome {
            Ok(v) => {
                row.set_cell(c.col_id, v);
                row.set_error(c.col_id, None);
            }
            Err(err) => {
                stats.num_excs += 1;
                let qualified = format!("{tbl_name}.{}", c.name);
                if !cols_with_excs.contains(&qualified) {
                    cols_with_excs.push(qualified);
                }
                row.set_cell(c.col_id, Value::Null);
                row.set_error(c.col_id, Some(err));
            }
        }
    }
    Ok(())
}
