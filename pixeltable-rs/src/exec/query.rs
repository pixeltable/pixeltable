//! Query execution: scan -> join -> filter -> sample -> aggregate -> window functions -> sort -> limit -> project.
//!
//! `QueryInput::snapshot()` captures, under the catalog lock, the table path (`scan::PathInput`: the table and the
//! bases its rows join with) of every from-table. `execute()` and `count()` run without the lock.
//!
//! Rows flow as `Rows`: one optional stored row per slot, where each from-table occupies one slot per table of its
//! path (leaf first); None is the missing side of an outer join. A query without joins scans its base table by address
//! (`Scan`), fetching (and decoding) only the rows it reads.
//! Aggregate and window function results are computed outside of expression evaluation and handed to it through
//! `RowContext::precomputed()`: `Plan::slots` maps each aggregate/window call node (by address) to an index into
//! the per-output-row `pre` values.
//!
//! Semantics that upstream gets from evaluating part of a query in SQL are reproduced where tests observe them:
//! grouping keys must be "SQL-expressible" (`is_sql_expr()`), `count()` rejects non-SQL filters, and division by
//! zero in a SQL-expressible arithmetic expression yields NULL (unless `RowContext::python_semantics()`).

use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;
use uuid::Uuid;

use super::fast::{Fast, VPred};
use super::prefetch::{CallResults, CallSlot};
use super::scan::{ElemInfo, PathCtx, PathInput};
use super::{JoinType, QuerySpec, ResultData, SampleClause};
use crate::catalog::Catalog;
use crate::catalog::table::ColId;
use crate::error::{Error, Result};
use crate::expr::display::display;
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ArithOp, CmpOp, ColumnRef, Expr, ExprKind, ExprRef, LogicalOp};
use crate::functions::agg::{AggPlan, AggState};
use crate::store::rows::{RowAddr, StoredRow, TableData};
use crate::store::table_data::ColChunk;
use crate::value::{Value, cmp_values};

/// One from-table of a query: its table path, captured under the catalog lock.
pub struct Source {
    pub path: PathInput,
    /// the slot of the path's first (leaf) table in a row
    pub offset: usize,
}

/// Everything a query reads, captured under the catalog lock.
pub struct QueryInput {
    pub q: QuerySpec,
    pub sources: Vec<Source>,
    pub select: Vec<(ExprRef, String)>,
    /// value expressions of unstored computed columns, evaluated on read
    pub unstored: HashMap<(Uuid, ColId), ExprRef>,
    /// number of slots of a row
    width: usize,
    /// table id -> (source index, slot) of the first slot that provides its columns
    slot_of: HashMap<Uuid, (usize, usize)>,
    /// (table id, pinned version) -> (source index, slot): a table can appear at two versions (a table joined with
    /// a snapshot of itself); a column reference picks the one its own pinned version names
    pinned_slot_of: HashMap<(Uuid, Option<i64>), (usize, usize)>,
    /// how the column references of the query's expressions read their columns, resolved once
    reads: FxHashMap<(Uuid, ColId, Option<i64>), ColRead>,
}

/// How a query reads a column.
#[derive(Clone, Copy)]
enum ColRead {
    /// the cell of the row in this slot: a stored column without read validation, not an iterator's
    Slot(usize),
    /// through the `PathCtx` of the from-table that provides it (unstored computed columns, iterator outputs, media
    /// validated on read)
    Path,
}

impl QueryInput {
    pub fn snapshot(cat: &mut Catalog, q: &QuerySpec) -> Result<QueryInput> {
        // where() and order_by() accept any expression; it must be computable from the query's tables
        for e in q.where_.iter().chain(q.order_by.iter().map(|(e, _)| e)) {
            q.check_bound(cat, e)?;
        }
        q.check_columns_exist(cat)?;
        let select = q.effective_select(cat)?;
        let mut sources = Vec::new();
        let mut unstored = HashMap::new();
        let mut slot_of = HashMap::new();
        let mut pinned_slot_of = HashMap::new();
        let mut width = 0;
        for (i, (tbl_id, version)) in q.from_tbls().into_iter().enumerate() {
            let path = PathInput::snapshot(cat, &cat.tbl_path(tbl_id, version)?)?;
            let offset = width;
            for (k, e) in path.elems.iter().enumerate() {
                for (col_id, x) in &e.info.unstored {
                    unstored.insert((e.info.tbl_id, *col_id), x.clone());
                }
                slot_of.entry(e.info.tbl_id).or_insert((i, offset + k));
                pinned_slot_of.entry((e.info.tbl_id, e.pinned)).or_insert((i, offset + k));
            }
            width += path.elems.len();
            sources.push(Source { path, offset });
        }
        let mut input =
            QueryInput { q: q.clone(), sources, select, unstored, width, slot_of, pinned_slot_of, reads: FxHashMap::default() };
        let mut exprs = q.exprs();
        exprs.extend(input.select.iter().map(|(e, _)| e.clone()));
        let mut reads = FxHashMap::default();
        for e in &exprs {
            for c in e.column_refs() {
                reads.entry((c.tbl_id, c.col_id, c.tbl_version)).or_insert_with(|| input.col_read(c));
            }
        }
        input.reads = reads;
        Ok(input)
    }

    /// True if `e` can be evaluated over rows on several threads (`par::is_pure()`), the calls in `calls` made.
    fn expr_pure(&self, e: &Expr, calls: &Option<CallResults>) -> bool {
        self.expr_pure_with(e, &|n| calls.as_ref().is_some_and(|r| r.complete(n)), 0)
    }

    /// `expr_pure()`, the calls `done` says have their outcomes made.
    fn expr_pure_with(&self, e: &Expr, done: &dyn Fn(&Expr) -> bool, depth: usize) -> bool {
        let col_pure = |c: &ColumnRef| {
            let read = self.reads.get(&(c.tbl_id, c.col_id, c.tbl_version)).copied().unwrap_or_else(|| self.col_read(c));
            match (read, self.unstored.get(&(c.tbl_id, c.col_id))) {
                (ColRead::Slot(_), _) => true,
                // an unstored computed column is pure if its expression is (iterator outputs and media validated on
                // read are not)
                (ColRead::Path, Some(x)) => depth < 64 && self.expr_pure_with(x, done, depth + 1),
                (ColRead::Path, None) => false,
            }
        };
        super::par::is_pure(e, &col_pure, done)
    }

    fn col_read(&self, col: &ColumnRef) -> ColRead {
        let Some(src) = self.source_of(col) else { return ColRead::Path };
        let source = &self.sources[src];
        let Some(i) = source.path.elems.iter().position(|e| e.info.tbl_id == col.tbl_id) else { return ColRead::Path };
        let info = &source.path.elems[i].info;
        let validated = info.read_validation.as_ref().is_some_and(|rv| rv.kind(col.col_id).is_some());
        if info.unstored.contains_key(&col.col_id) || info.iterator.is_some() || validated {
            return ColRead::Path;
        }
        ColRead::Slot(source.offset + i)
    }

    fn slot(&self, tbl_id: &Uuid) -> Option<usize> {
        self.slot_of.get(tbl_id).map(|(_, slot)| *slot)
    }

    /// The from-table (source index) that provides the column `col`: the one that reads its table at the version
    /// the reference is pinned to, else the first one that has its table.
    fn source_of(&self, col: &ColumnRef) -> Option<usize> {
        self.pinned_slot_of
            .get(&(col.tbl_id, col.tbl_version))
            .or_else(|| self.slot_of.get(&col.tbl_id))
            .map(|(src, _)| *src)
    }

    /// The from-table that provides the rows of `tbl_id` (the first one that has its table).
    fn source_of_tbl(&self, tbl_id: &Uuid) -> Option<usize> {
        self.slot_of.get(tbl_id).map(|(src, _)| *src)
    }

    /// Sets the slots of source `i` in `row` (a row of `width` slots).
    fn place<'a>(&self, row: &mut [Option<&'a StoredRow>], i: usize, rows: &[&'a StoredRow]) {
        let off = self.sources[i].offset;
        for (k, r) in rows.iter().enumerate() {
            row[off + k] = Some(*r);
        }
    }

    /// `e` compiled for evaluation over this query's rows (`exec::fast`): column references read their slot's cell
    /// where the query reads the column that way (`ColRead::Slot`).
    fn compile(&self, e: &ExprRef) -> Fast {
        Fast::compile(e, &|c| {
            match self.reads.get(&(c.tbl_id, c.col_id, c.tbl_version)).copied().unwrap_or_else(|| self.col_read(c)) {
                ColRead::Slot(slot) => Some(slot),
                ColRead::Path => None,
            }
        })
    }
}

/// Rows flowing through a query: `width` slots per row (one per table of each from-table's path, leaf first; None is
/// the missing side of an outer join), stored flat.
#[derive(Clone)]
struct Rows<'a> {
    width: usize,
    slots: Vec<Option<&'a StoredRow>>,
}

impl<'a> Rows<'a> {
    fn new(width: usize) -> Rows<'a> {
        Rows { width, slots: Vec::new() }
    }

    fn with_capacity(width: usize, n: usize) -> Rows<'a> {
        Rows { width, slots: Vec::with_capacity(width * n) }
    }

    fn len(&self) -> usize {
        self.slots.len() / self.width
    }

    fn row(&self, i: usize) -> &[Option<&'a StoredRow>] {
        &self.slots[i * self.width..(i + 1) * self.width]
    }

    fn push(&mut self, row: &[Option<&'a StoredRow>]) {
        self.slots.extend_from_slice(row);
    }

    /// The rows `idx`, in that order.
    fn gather(&self, idx: impl ExactSizeIterator<Item = usize>) -> Rows<'a> {
        let mut out = Rows::with_capacity(self.width, idx.len());
        for i in idx {
            out.push(self.row(i));
        }
        out
    }

    fn reverse(&mut self) {
        let n = self.len();
        *self = self.gather((0..n).rev());
    }

    fn append(&mut self, other: Rows<'a>) {
        if self.slots.is_empty() {
            *self = other;
        } else {
            self.slots.extend(other.slots);
        }
    }
}

/// Output rows: the (representative) input rows (a base table's rows by address, or materialized rows), and for each
/// the precomputed aggregate/window values (`pre`, empty if the query has none).
struct OutRows<'a> {
    rows: Scan<'a>,
    pre: Vec<Vec<Value>>,
}

impl<'a> OutRows<'a> {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn pre(&self, i: usize) -> &[Value] {
        self.pre.get(i).map_or(&[], Vec::as_slice)
    }

    /// The rows `order` (indexes), in that order.
    fn permute(mut self, order: &[u32]) -> OutRows<'a> {
        let rows = self.rows.pick(order.iter().map(|i| *i as usize));
        let pre = if self.pre.is_empty() {
            Vec::new()
        } else {
            order.iter().map(|i| std::mem::take(&mut self.pre[*i as usize])).collect()
        };
        OutRows { rows, pre }
    }

    /// The rows after the first `offset`, at most `limit` of them.
    fn window(mut self, offset: usize, limit: Option<usize>) -> OutRows<'a> {
        let n = self.len();
        let end = limit.map_or(n, |l| offset.saturating_add(l).min(n));
        if offset == 0 && end == n {
            return self;
        }
        let start = offset.min(end);
        let rows = self.rows.pick(start..end);
        let pre = if self.pre.is_empty() { Vec::new() } else { self.pre.drain(start..end).collect() };
        OutRows { rows, pre }
    }
}

/// Row context for query evaluation.
struct Ctx<'a> {
    input: &'a QueryInput,
    row: &'a [Option<&'a StoredRow>],
    pre: &'a [Value],
    slots: &'a HashMap<usize, usize>,
    /// the outcomes of prefetched calls for this row (`exec::prefetch`)
    calls: Option<CallSlot<'a>>,
}

fn node_key(e: &Expr) -> usize {
    e as *const Expr as usize
}

impl Ctx<'_> {
    /// Runs `f` with the path context of the from-table that provides `col` (None: the missing side of an outer
    /// join). Column reads go through `PathCtx`: unstored and iterator-produced columns, read validation.
    fn with_path<T>(&self, src: Option<usize>, f: impl FnOnce(&PathCtx<'_>) -> Result<T>) -> Result<Option<T>> {
        let Some(src) = src else {
            return Err(Error::internal("a table of the expression is not provided by the query's tables"));
        };
        let source = &self.input.sources[src];
        let n = source.path.elems.len();
        let mut rows: SmallVec<[&StoredRow; 4]> = SmallVec::with_capacity(n);
        for r in &self.row[source.offset..source.offset + n] {
            match r {
                Some(r) => rows.push(*r),
                None => return Ok(None),
            }
        }
        let infos: SmallVec<[&ElemInfo; 4]> = source.path.elems.iter().map(|e| &e.info).collect();
        f(&PathCtx { elems: &infos, rows: &rows, python: false, calls: self.calls }).map(Some)
    }
}

impl RowContext for Ctx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        if let Some(ColRead::Slot(slot)) = self.input.reads.get(&(col.tbl_id, col.col_id, col.tbl_version)) {
            return Ok(self.row[*slot].map_or(Value::Null, |r| r.cell(col.col_id).clone()));
        }
        Ok(self.with_path(self.input.source_of(col), |ctx| ctx.column_value(col))?.unwrap_or(Value::Null))
    }

    fn column_error(&self, col: &ColumnRef) -> Result<Option<crate::store::rows::CellError>> {
        Ok(self.with_path(self.input.source_of(col), |ctx| ctx.column_error(col))?.flatten())
    }

    fn row_key(&self, tbl_id: Uuid) -> Result<Option<(i64, Vec<i64>)>> {
        Ok(self.with_path(self.input.source_of_tbl(&tbl_id), |ctx| ctx.row_key(tbl_id))?.flatten())
    }

    fn precomputed(&self, e: &Expr) -> Option<Value> {
        if self.slots.is_empty() {
            return None;
        }
        self.slots.get(&node_key(e)).map(|i| self.pre[*i].clone())
    }

    fn call_result(&self, e: &Expr) -> Option<Result<Value>> {
        self.calls?.get(e)
    }

}

// ------------------------------------------------------------------------------------------- SQL expressibility

/// True if upstream can evaluate `e` in SQL (upstream `SqlElementCache.contains()`): column references, literals,
/// operators over such, and builtins with a SQL translation. JSON paths, inline lists/dicts, casts, array slices,
/// mappers and Python UDFs are not.
pub fn is_sql_expr(e: &Expr, unstored: &HashMap<(Uuid, ColId), ExprRef>) -> bool {
    match &e.kind {
        ExprKind::ColumnRef(c) => match unstored.get(&(c.tbl_id, c.col_id)) {
            Some(v) => is_sql_expr(v, unstored),
            None => true,
        },
        ExprKind::Literal(_) | ExprKind::RowidRef { .. } => true,
        // properties of stored columns read the stored value or its cell metadata
        ExprKind::ColumnProperty { col, prop } => *prop != crate::expr::ColumnProp::LocalPath && is_sql_expr(col, unstored),
        // upstream evaluates a float `%` and an int `**` in Python
        ExprKind::Arithmetic { op: ArithOp::Mod, .. } if e.col_type.is_float() => false,
        ExprKind::Arithmetic { op: ArithOp::Pow, .. } if !e.col_type.is_float() => false,
        ExprKind::Arithmetic { lhs, rhs, .. } | ExprKind::StringOp { lhs, rhs, .. } | ExprKind::Comparison { lhs, rhs, .. } => {
            is_sql_expr(lhs, unstored) && is_sql_expr(rhs, unstored)
        }
        ExprKind::Compound { operands, .. } => operands.iter().all(|o| is_sql_expr(o, unstored)),
        ExprKind::IsNull(x) | ExprKind::InList { lhs: x, .. } => is_sql_expr(x, unstored),
        ExprKind::FunctionCall(fc) => {
            fc.group_by.is_empty()
                && fc.order_by.is_empty()
                && fc.args.iter().chain(fc.kwargs.iter().map(|(_, a)| a)).all(|a| is_sql_expr(a, unstored))
                && fc.func.is_sql(&fc.args.iter().map(|a| a.col_type.clone()).collect::<Vec<_>>())
        }
        _ => false,
    }
}

/// Splits a conjunction into its conjuncts.
fn conjuncts(e: &ExprRef) -> Vec<ExprRef> {
    match &e.kind {
        ExprKind::Compound { op: LogicalOp::And, operands } => operands.iter().flat_map(conjuncts).collect(),
        _ => vec![e.clone()],
    }
}

// ------------------------------------------------------------------------------------------------ planning

struct AggCall {
    expr: ExprRef,
    plan: AggPlan,
}

struct WindowCall {
    expr: ExprRef,
    plan: AggPlan,
}

/// What `execute()` needs beyond the spec: the distinct aggregate and window calls and their slots.
struct Plan {
    aggs: Vec<AggCall>,
    windows: Vec<WindowCall>,
    slots: HashMap<usize, usize>,
    /// Some: aggregate (Some(empty): a single group, unless `group_slot` is set)
    grouping: Option<Vec<ExprRef>>,
    /// `group_by(base_table)`: the slot of the base table, whose rows define the groups
    group_slot: Option<usize>,
    /// the ordering the rows need before window functions are evaluated
    window_order: Vec<(ExprRef, Option<bool>)>,
    /// the ordering the input rows need before aggregation (aggregates with an ordering requirement)
    agg_order: Vec<(ExprRef, Option<bool>)>,
}

fn expr_key(e: &Expr) -> String {
    crate::expr::serde::to_json(e).map(|j| j.to_string()).unwrap_or_default()
}

fn collect_calls(e: &ExprRef, out: &mut Vec<ExprRef>, pred: &dyn Fn(&crate::expr::FunctionCall) -> bool) {
    if let ExprKind::FunctionCall(fc) = &e.kind {
        if pred(fc) {
            out.push(e.clone());
            return;
        }
    }
    for c in e.kind.children() {
        collect_calls(c, out, pred);
    }
}

fn is_std_agg(fc: &crate::expr::FunctionCall) -> bool {
    fc.is_agg() && !fc.is_window()
}

/// Upstream `Analyzer._determine_agg_status()`: (is aggregate output, is aggregate input).
fn agg_status(e: &ExprRef, grouping: &[String]) -> Result<(bool, bool)> {
    if grouping.contains(&expr_key(e)) {
        return Ok((true, true));
    }
    match &e.kind {
        ExprKind::FunctionCall(fc) if is_std_agg(fc) => {
            for c in e.kind.children() {
                let (_, is_input) = agg_status(c, grouping)?;
                if !is_input {
                    crate::bail!(InvalidExpression, "Invalid nested aggregates: {}", display(e));
                }
            }
            Ok((true, false))
        }
        ExprKind::Literal(_) | ExprKind::RelativeRoot { .. } => Ok((true, true)),
        ExprKind::ColumnRef(_) => Ok((false, true)),
        _ => {
            let children = e.kind.children();
            if children.is_empty() {
                return Ok((true, true));
            }
            let mut is_output = true;
            let mut is_input = true;
            for c in children {
                let (o, i) = agg_status(c, grouping)?;
                is_output &= o;
                is_input &= i;
            }
            if !is_output && !is_input {
                crate::bail!(InvalidExpression, "Invalid expression, mixes aggregate with non-aggregate: {}", display(e));
            }
            Ok((is_output, is_input))
        }
    }
}

/// Upstream `print_order_by_clause()`.
fn print_ordering(o: &[(ExprRef, Option<bool>)]) -> String {
    o.iter()
        .map(|(e, asc)| {
            let a = match asc {
                Some(true) => ", asc=True",
                Some(false) => ", asc=False",
                None => "",
            };
            format!("({}{a})", display(e))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Upstream `combine_order_by_clauses()` for two clauses; None if they are incompatible.
fn combine_orderings(a: &[(ExprRef, Option<bool>)], b: &[(ExprRef, Option<bool>)]) -> Option<Vec<(ExprRef, Option<bool>)>> {
    let mut out = Vec::new();
    for ((e1, a1), (e2, a2)) in a.iter().zip(b.iter()) {
        if expr_key(e1) != expr_key(e2) {
            return None;
        }
        if let (Some(x), Some(y)) = (a1, a2) {
            if x != y {
                return None;
            }
        }
        out.push((e1.clone(), a1.or(*a2)));
    }
    let n = a.len().min(b.len());
    out.extend(a[n..].iter().cloned());
    out.extend(b[n..].iter().cloned());
    Some(out)
}

/// Rejects grouping keys upstream cannot evaluate in SQL (upstream `_validate_grouping_exprs()`).
fn validate_grouping(group_by: &[ExprRef], unstored: &HashMap<(Uuid, ColId), ExprRef>) -> Result<()> {
    for e in group_by {
        if !is_sql_expr(e, unstored) {
            crate::bail!(
                InvalidExpression,
                "Invalid grouping expression, needs to be expressible in SQL: {}. Store the value in a computed column of a scalar type (cast with .astype() if needed) and use that column with distinct() or group_by().",
                display(e)
            );
        }
        if e.contains_agg() {
            crate::bail!(InvalidExpression, "Grouping expression contains aggregate function: {}", display(e));
        }
    }
    Ok(())
}

impl Plan {
    fn new(input: &QueryInput) -> Result<Plan> {
        let q = &input.q;
        let group_slot = match q.grouping_tbl {
            Some(t) => Some(input.slot(&t).ok_or_else(|| Error::internal("grouping table is not in the table path"))?),
            None => None,
        };
        let candidates: Vec<&ExprRef> = input.select.iter().map(|(e, _)| e).collect();
        let mut agg_exprs = Vec::new();
        let mut window_exprs = Vec::new();
        for e in candidates.iter().copied().chain(q.order_by.iter().map(|(e, _)| e)) {
            collect_calls(e, &mut agg_exprs, &is_std_agg);
            collect_calls(e, &mut window_exprs, &|fc| fc.is_window());
        }
        let mut slots = HashMap::new();
        let mut keys: HashMap<String, usize> = HashMap::new();
        let mut aggs = Vec::new();
        for e in agg_exprs {
            let k = expr_key(&e);
            let idx = match keys.get(&k) {
                Some(i) => *i,
                None => {
                    let fc = e.as_function_call().ok_or_else(|| Error::internal("aggregate call expected"))?;
                    aggs.push(AggCall { expr: e.clone(), plan: AggPlan::new(fc)? });
                    keys.insert(k, aggs.len() - 1);
                    aggs.len() - 1
                }
            };
            slots.insert(node_key(&e), idx);
        }
        let mut windows = Vec::new();
        let mut wkeys: HashMap<String, usize> = HashMap::new();
        for e in window_exprs {
            let k = expr_key(&e);
            let idx = match wkeys.get(&k) {
                Some(i) => *i,
                None => {
                    let fc = e.as_function_call().ok_or_else(|| Error::internal("window call expected"))?;
                    windows.push(WindowCall { expr: e.clone(), plan: AggPlan::new(fc)? });
                    wkeys.insert(k, windows.len() - 1);
                    windows.len() - 1
                }
            };
            slots.insert(node_key(&e), aggs.len() + idx);
        }

        let mut grouping = q.group_by.clone();
        if group_slot.is_some() {
            grouping = Some(Vec::new());
        }
        if !aggs.is_empty() || group_slot.is_some() {
            let g = grouping.get_or_insert_with(Vec::new).clone();
            let gkeys: Vec<String> = g.iter().map(|e| expr_key(e)).collect();
            for (e, _) in &input.select {
                let (is_output, _) = agg_status(e, &gkeys)?;
                if !is_output {
                    crate::bail!(InvalidExpression, "Invalid non-aggregate expression in aggregate query: {}", display(e));
                }
            }
            if let Some(w) = &q.where_ {
                if w.contains_agg() {
                    crate::bail!(InvalidExpression, "where() cannot contain aggregate functions: {}", display(w));
                }
            }
        }
        if let Some(g) = &grouping {
            validate_grouping(g, &input.unstored)?;
        }
        if let Some(s) = &q.sample {
            if let Some(w) = &q.where_ {
                if let Some(f) = conjuncts(w).into_iter().find(|c| !is_sql_expr(c, &input.unstored)) {
                    crate::bail!(UnsupportedOperation, "Filter {} not expressible in SQL", display(&f));
                }
            }
            let _ = s;
        }

        // ordering requirements: the query's order by, window functions, ordered aggregates
        let mut window_order: Vec<(ExprRef, Option<bool>)> = q.order_by.iter().map(|(e, a)| (e.clone(), Some(*a))).collect();
        let mut requirements: Vec<Vec<(ExprRef, Option<bool>)>> = Vec::new();
        for w in &windows {
            let fc = w.expr.as_function_call().ok_or_else(|| Error::internal("window call expected"))?;
            let mut o: Vec<(ExprRef, Option<bool>)> = fc.group_by.iter().map(|e| (e.clone(), None)).collect();
            o.extend(fc.order_by.iter().map(|e| (e.clone(), Some(true))));
            requirements.push(o);
        }
        let mut agg_order: Vec<(ExprRef, Option<bool>)> = Vec::new();
        for a in &aggs {
            let fc = a.expr.as_function_call().ok_or_else(|| Error::internal("aggregate call expected"))?;
            if !fc.order_by.is_empty() {
                let mut o: Vec<(ExprRef, Option<bool>)> =
                    grouping.iter().flatten().map(|e| (e.clone(), None)).collect();
                o.extend(fc.order_by.iter().map(|e| (e.clone(), Some(true))));
                // incompatible requirements are reported below, with the query's order by
                agg_order = combine_orderings(&agg_order, &o).unwrap_or(agg_order);
                requirements.push(o);
            }
        }
        for r in requirements {
            window_order = match combine_orderings(&window_order, &r) {
                Some(c) => c,
                None => crate::bail!(
                    UnsupportedOperation,
                    "Incompatible ordering requirements: {} vs {}",
                    print_ordering(&window_order),
                    print_ordering(&r)
                ),
            };
        }
        if !windows.is_empty() && grouping.is_some() {
            return Err(Error::not_implemented("window functions in an aggregation query"));
        }
        Ok(Plan { aggs, windows, slots, grouping, group_slot, window_order, agg_order })
    }
}

// ------------------------------------------------------------------------------------------------ scan / join

/// A hashable stand-in for a value under SQL equality (ints and integral floats are equal).
fn group_key(v: &Value) -> String {
    match v {
        Value::Null => "N".into(),
        Value::Bool(b) => format!("b{b}"),
        Value::Int(i) => format!("n{i}"),
        Value::Float(f) => {
            if f.fract() == 0.0 && f.abs() < 9.0e15 {
                format!("n{}", *f as i64)
            } else {
                format!("f{f:?}")
            }
        }
        Value::String(s) | Value::Media(s) => format!("s{s}"),
        Value::Timestamp(t) => format!("t{t}"),
        Value::Date(d) => format!("d{d}"),
        Value::Uuid(u) => format!("u{u}"),
        Value::Binary(b) => format!("x{b:?}"),
        Value::Json(j) => match j {
            crate::value::Json::Number(_) | crate::value::Json::String(_) | crate::value::Json::Bool(_) => group_key(&v.json_scalar()),
            _ => format!("j{j}"),
        },
        Value::Array(a) => format!("a{:?}", a),
        Value::PyObj(_) => "o".into(),
    }
}

/// A part of a grouping or join key: a value under SQL equality (ints and integral floats are equal, as are strings
/// and media paths with the same text), hashable; `group_key()` without the text.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum KeyPart {
    Null,
    Bool(bool),
    Num(i64),
    /// the bits of a float that is not integral (one bit pattern for every NaN)
    Float(u64),
    Str(String),
    Ts(i64),
    Date(i32),
    Uuid(u128),
    Bin(Vec<u8>),
    /// JSON lists and objects, arrays: by their text
    Text(String),
    Obj,
}

/// The key of a row: one part per key expression (most keys have one or two).
pub type GroupKey = SmallVec<[KeyPart; 2]>;

fn key_part(v: &Value) -> KeyPart {
    match v {
        Value::Null => KeyPart::Null,
        Value::Bool(b) => KeyPart::Bool(*b),
        Value::Int(i) => KeyPart::Num(*i),
        Value::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => KeyPart::Num(*f as i64),
        Value::Float(f) if f.is_nan() => KeyPart::Float(f64::NAN.to_bits()),
        Value::Float(f) => KeyPart::Float(f.to_bits()),
        Value::String(s) | Value::Media(s) => KeyPart::Str(s.clone()),
        Value::Timestamp(t) => KeyPart::Ts(*t),
        Value::Date(d) => KeyPart::Date(*d),
        Value::Uuid(u) => KeyPart::Uuid(u.as_u128()),
        Value::Binary(b) => KeyPart::Bin(b.clone()),
        Value::Json(j) => match j {
            crate::value::Json::Number(_) | crate::value::Json::String(_) | crate::value::Json::Bool(_) => key_part(&v.json_scalar()),
            _ => KeyPart::Text(format!("j{j}")),
        },
        Value::Array(a) => KeyPart::Text(format!("a{:?}", a)),
        Value::PyObj(_) => KeyPart::Obj,
    }
}

/// The hashable key of `vals` (see `KeyPart`).
pub fn group_key_of(vals: &[Value]) -> GroupKey {
    vals.iter().map(key_part).collect()
}

pub fn keys_of(vals: &[Value]) -> String {
    let mut s = String::new();
    for v in vals {
        s.push_str(&group_key(v));
        s.push('\u{1}');
    }
    s
}

/// The rows of the first from-table of a query, before the where clause, in key order.
enum Scan<'a> {
    /// the visible rows of a base table, by address: fetched (and their storage chunks decoded) on first use
    Leaf { data: &'a TableData, addrs: Vec<RowAddr> },
    /// materialized rows: a view's joined path rows, the rows a key index found, joined rows
    Rows(Rows<'a>),
}

impl<'a> Scan<'a> {
    /// The rows of the first from-table of a query without joins: those a key index finds for the where clause
    /// (`PathInput::index_rowids()`), else all visible rows (of a base table, only the first `prefix` ones).
    fn new(input: &'a QueryInput, prefix: Option<usize>) -> Result<Scan<'a>> {
        let path = &input.sources[0].path;
        if let Some(rowids) = path.index_rowids(input.q.where_.as_ref()) {
            let mut rows = Rows::new(path.elems.len());
            for r in path.rows_for_keys(rowids.iter().map(|r| (*r, &[] as &[i64]))) {
                rows.slots.extend(r.into_iter().map(Some));
            }
            return Ok(Scan::Rows(rows));
        }
        if let [e] = path.elems.as_slice() {
            return Ok(Scan::Leaf { data: &e.data, addrs: e.data.scan_order_upto(e.version, prefix.unwrap_or(usize::MAX)) });
        }
        let mut rows = Rows::new(path.elems.len());
        path.for_each_row(|r| rows.slots.extend(r.iter().map(|x| Some(*x))))?;
        Ok(Scan::Rows(rows))
    }

    fn len(&self) -> usize {
        match self {
            Scan::Leaf { addrs, .. } => addrs.len(),
            Scan::Rows(r) => r.len(),
        }
    }

    /// Calls `f` with row `i`.
    fn with_row<R>(&self, i: usize, f: impl FnOnce(&[Option<&'a StoredRow>]) -> R) -> R {
        match self {
            Scan::Leaf { data, addrs } => f(&[Some(data.row_by_addr(addrs[i]))]),
            Scan::Rows(r) => f(r.row(i)),
        }
    }

    fn reverse(&mut self) {
        match self {
            Scan::Leaf { addrs, .. } => addrs.reverse(),
            Scan::Rows(r) => r.reverse(),
        }
    }

    /// The rows `idx`, in that order (their storage chunks decoded in parallel first).
    fn select(&self, idx: &[usize]) -> Rows<'a> {
        match self {
            Scan::Leaf { data, addrs } => {
                let picked: Vec<RowAddr> = idx.iter().map(|i| addrs[*i]).collect();
                super::par::detached(|| data.decode_chunks_of(&picked));
                Rows { width: 1, slots: super::par::map_rows(picked.len(), |k| Some(data.row_by_addr(picked[k]))) }
            }
            Scan::Rows(r) => r.gather(idx.iter().copied()),
        }
    }

    /// The rows `idx`, in that order, as a scan (a base table's stay addresses).
    fn pick(self, idx: impl ExactSizeIterator<Item = usize>) -> Scan<'a> {
        match self {
            Scan::Leaf { data, addrs } => Scan::Leaf { data, addrs: idx.map(|i| addrs[i]).collect() },
            Scan::Rows(r) => Scan::Rows(r.gather(idx)),
        }
    }

    /// All rows.
    fn into_rows(self) -> Rows<'a> {
        match self {
            Scan::Leaf { data, addrs } => {
                super::par::detached(|| data.decode_chunks_of(&addrs));
                Rows { width: 1, slots: super::par::map_rows(addrs.len(), |k| Some(data.row_by_addr(addrs[k]))) }
            }
            Scan::Rows(r) => r,
        }
    }
}

/// For a join predicate, the equality conjuncts `lhs == rhs` whose sides reference only the left tables resp. only
/// the right table (hash join keys): (left exprs, right exprs).
fn equi_keys(pred: &ExprRef, input: &QueryInput, right_src: usize) -> (Vec<ExprRef>, Vec<ExprRef>) {
    let side = |e: &ExprRef| -> Option<bool> {
        let refs = e.column_refs();
        if refs.is_empty() {
            return None;
        }
        let srcs: Vec<usize> = refs.iter().filter_map(|c| input.source_of(c)).collect();
        if srcs.len() != refs.len() {
            return None;
        }
        if srcs.iter().all(|s| *s == right_src) {
            Some(true)
        } else if srcs.iter().all(|s| *s < right_src) {
            Some(false)
        } else {
            None
        }
    };
    let mut l = Vec::new();
    let mut r = Vec::new();
    for c in conjuncts(pred) {
        if let ExprKind::Comparison { op: CmpOp::Eq, lhs, rhs } = &c.kind {
            match (side(lhs), side(rhs)) {
                (Some(false), Some(true)) => {
                    l.push(lhs.clone());
                    r.push(rhs.clone());
                }
                (Some(true), Some(false)) => {
                    l.push(rhs.clone());
                    r.push(lhs.clone());
                }
                _ => {}
            }
        }
    }
    (l, r)
}

/// True if equal hash keys decide a join predicate: every conjunct is one of the equality conjuncts `lkeys[i] ==
/// rkeys[i]`, each comparing two stored columns of the same scalar type for which equal keys (`KeyPart`) mean SQL
/// equality (not floats, JSON or media, whose equality is not that of their keys); the predicate then needs no
/// evaluation for the candidates the keys find.
fn keys_decide(pred: &ExprRef, lkeys: &[ExprRef], rkeys: &[ExprRef], input: &QueryInput) -> bool {
    use crate::types::TypeKind;
    let plain = |e: &ExprRef| {
        e.as_column_ref().is_some_and(|c| matches!(input.reads.get(&(c.tbl_id, c.col_id, c.tbl_version)), Some(ColRead::Slot(_))))
    };
    let exact = |e: &ExprRef| {
        matches!(e.col_type.kind, TypeKind::Int | TypeKind::String | TypeKind::Bool | TypeKind::Timestamp | TypeKind::Date | TypeKind::Uuid)
    };
    conjuncts(pred).len() == lkeys.len()
        && lkeys.iter().zip(rkeys).all(|(l, r)| plain(l) && plain(r) && exact(l) && l.col_type.kind == r.col_type.kind)
}

/// The right rows of a hash join by key (keys of one integer, the most common, through their own map).
#[derive(Default)]
struct JoinIndex {
    by_key: FxHashMap<GroupKey, Vec<usize>>,
    by_int: FxHashMap<i64, Vec<usize>>,
}

impl JoinIndex {
    fn insert(&mut self, k: GroupKey, j: usize) {
        match k.as_slice() {
            [KeyPart::Num(i)] => self.by_int.entry(*i).or_default().push(j),
            _ => self.by_key.entry(k).or_default().push(j),
        }
    }

    fn get(&self, k: &GroupKey) -> &[usize] {
        let found = match k.as_slice() {
            [KeyPart::Num(i)] => self.by_int.get(i),
            _ => self.by_key.get(k),
        };
        found.map_or(&[], Vec::as_slice)
    }

    fn get_int(&self, i: i64) -> &[usize] {
        self.by_int.get(&i).map_or(&[], Vec::as_slice)
    }
}

/// Scans the from-tables and joins them (in from-clause order): rows of `input.width` slots.
fn scan_and_join<'a>(input: &'a QueryInput) -> Result<Rows<'a>> {
    let n = input.width;
    let empty = HashMap::new();
    type Row<'a> = SmallVec<[Option<&'a StoredRow>; 4]>;
    // the rows joined so far: the first from-table's (a base table's by address), then the joined rows
    let first = &input.sources[0].path;
    let mut left: Scan<'a> = match first.elems.as_slice() {
        [e] => Scan::Leaf { data: &e.data, addrs: e.data.scan_order(e.version) },
        _ => {
            let mut rows = Rows::with_capacity(n, first.elems[0].data.num_live());
            first.for_each_row(|r| {
                let start = rows.slots.len();
                rows.slots.resize(start + n, None);
                input.place(&mut rows.slots[start..], 0, r);
            })?;
            Scan::Rows(rows)
        }
    };
    // `jr` := left row `i` (all `n` slots)
    let fill_left = |left: &Scan<'a>, i: usize, jr: &mut Row<'a>| match left {
        Scan::Leaf { data, addrs } => {
            jr.fill(None);
            jr[0] = Some(data.row_by_addr(addrs[i]));
        }
        Scan::Rows(r) => jr.copy_from_slice(r.row(i)),
    };
    for (k, join) in input.q.joins.iter().enumerate() {
        let slot = k + 1;
        let right = input.sources[slot].path.rows()?;
        let mut out = Rows::new(n);
        let mut right_matched = vec![false; right.len()];
        let nleft = left.len();
        match (&join.on, join.how) {
            (None, _) | (_, JoinType::Cross) => {
                let mut jr: Row<'a> = smallvec![None; n];
                for i in 0..nleft {
                    for r in &right {
                        fill_left(&left, i, &mut jr);
                        input.place(&mut jr, slot, r);
                        out.push(&jr);
                    }
                }
            }
            (Some(pred), how) => {
                let (lkeys, rkeys) = equi_keys(pred, input, slot);
                let pure = input.expr_pure(pred, &None);
                let decided = !lkeys.is_empty() && keys_decide(pred, &lkeys, &rkeys, input);
                let pred_fast = input.compile(pred);
                let lfast: Vec<Fast> = lkeys.iter().map(|e| input.compile(e)).collect();
                let rfast: Vec<Fast> = rkeys.iter().map(|e| input.compile(e)).collect();
                // candidate right rows per left row: by hash on the equality keys, else all
                let mut index = JoinIndex::default();
                if !lkeys.is_empty() {
                    let right_keys = super::par::try_map_rows(right.len(), pure, |j| {
                        let mut jr: Row<'a> = smallvec![None; n];
                        input.place(&mut jr, slot, &right[j]);
                        let ctx = Ctx { input, row: &jr, pre: &[], slots: &empty, calls: None };
                        let mut key = GroupKey::new();
                        for f in &rfast {
                            let v = f.eval(&jr, &ctx)?;
                            if v.is_null() {
                                return Ok(None);
                            }
                            key.push(key_part(&v));
                        }
                        Ok(Some(key))
                    })?;
                    for (j, k) in right_keys.into_iter().enumerate() {
                        if let Some(k) = k {
                            index.insert(k, j);
                        }
                    }
                }
                let all: Vec<usize> = (0..right.len()).collect();
                // a key of one stored column of a base table on the left: read from the typed columns of its chunks
                let typed_key = match (&left, lfast.as_slice()) {
                    (Scan::Leaf { .. }, [Fast::Cell { slot: 0, col }]) => Some(*col),
                    _ => None,
                };
                // the joined rows of left row `i` (`key`: its key, if known), and the right rows it matched
                let join_row = |i: usize, key: Option<Option<i64>>, jr: &mut Row<'a>, out: &mut Rows<'a>, matched_right: &mut Vec<usize>| -> Result<()> {
                    fill_left(&left, i, jr);
                    let candidates: &[usize] = match key {
                        _ if lkeys.is_empty() => &all,
                        Some(Some(x)) => index.get_int(x),
                        Some(None) => &[],
                        None => {
                            let ctx = Ctx { input, row: jr, pre: &[], slots: &empty, calls: None };
                            let mut key = GroupKey::new();
                            let mut null = false;
                            for f in &lfast {
                                let v = f.eval(jr, &ctx)?;
                                null |= v.is_null();
                                key.push(key_part(&v));
                            }
                            if null { &[] } else { index.get(&key) }
                        }
                    };
                    let l: Row<'a> = jr.clone();
                    let mut matched = false;
                    for &j in candidates {
                        jr.copy_from_slice(&l);
                        input.place(jr, slot, &right[j]);
                        let passes = decided || {
                            let ctx = Ctx { input, row: jr, pre: &[], slots: &empty, calls: None };
                            pred_fast.eval_bool(jr, &ctx)? == Some(true)
                        };
                        if passes {
                            matched = true;
                            matched_right.push(j);
                            out.push(jr);
                        }
                    }
                    if !matched && matches!(how, JoinType::Left | JoinType::FullOuter) {
                        out.push(&l);
                    }
                    Ok(())
                };
                // the joined rows of a run of left rows, and the right rows they matched, up to the first failure
                let join_rows = |range: std::ops::Range<usize>| -> (Rows<'a>, Vec<usize>, Option<Error>) {
                    let mut out = Rows::with_capacity(n, range.len());
                    let mut matched_right = Vec::new();
                    let mut jr: Row<'a> = smallvec![None; n];
                    if let (Some(col), Scan::Leaf { data, addrs }) = (typed_key, &left) {
                        for (a, b) in super::fast::chunk_runs(addrs, range) {
                            let (c, off) = data.column_at(addrs[a], col);
                            for i in a..b {
                                let k = off + (i - a);
                                // an integer key, or NULL; other kinds are evaluated
                                let key = match c.as_ref() {
                                    ColChunk::Int(t) => Some((!t.is_null(k)).then(|| t.vals[k])),
                                    _ => None,
                                };
                                if let Err(err) = join_row(i, key, &mut jr, &mut out, &mut matched_right) {
                                    return (out, matched_right, Some(err));
                                }
                            }
                        }
                        return (out, matched_right, None);
                    }
                    for i in range {
                        if let Err(err) = join_row(i, None, &mut jr, &mut out, &mut matched_right) {
                            return (out, matched_right, Some(err));
                        }
                    }
                    (out, matched_right, None)
                };
                let parts = if pure { super::par::map_chunks(nleft, join_rows) } else { vec![join_rows(0..nleft)] };
                out.slots.reserve(parts.iter().map(|(p, _, _)| p.slots.len()).sum());
                for (part, matched, failed) in parts {
                    if let Some(err) = failed {
                        return Err(err);
                    }
                    out.append(part);
                    for j in matched {
                        right_matched[j] = true;
                    }
                }
                if how == JoinType::FullOuter {
                    let mut jr: Row<'a> = smallvec![None; n];
                    for (j, r) in right.iter().enumerate() {
                        if !right_matched[j] {
                            jr.fill(None);
                            input.place(&mut jr, slot, r);
                            out.push(&jr);
                        }
                    }
                }
            }
        }
        left = Scan::Rows(out);
    }
    Ok(left.into_rows())
}

/// Phase A of evaluating `exprs` over `n` rows of a query (`with_row(i, f)` calls `f` with row `i` and its precomputed
/// values): makes their user-function calls together (see `exec::prefetch`); a query stops at its first error.
fn prefetch_calls<'a>(
    input: &'a QueryInput,
    exprs: &[&ExprRef],
    n: usize,
    with_row: &(dyn Fn(usize, &mut dyn FnMut(&[Option<&StoredRow>], &[Value])) + Sync),
    slots: &'a HashMap<usize, usize>,
) -> Result<Option<CallResults>> {
    let derived = |c: &ColumnRef| input.unstored.get(&(c.tbl_id, c.col_id)).cloned();
    super::prefetch::prefetch(
        exprs,
        &derived,
        n,
        true,
        &|i, slot, f| with_row(i, &mut |row, pre| f(&Ctx { input, row, pre, slots, calls: Some(slot) })),
        &|e, done| input.expr_pure_with(e, done, 0),
    )
}

fn slot_of(calls: &Option<CallResults>, row: usize) -> Option<CallSlot<'_>> {
    calls.as_ref().map(|results| CallSlot { results, row })
}

/// The indexes of the rows `range` of `scan` that pass the filter `w`, in order, up to the first `want` of them; or
/// the error of the first row that fails before `want` rows passed. The filter's user-function calls are made for the
/// rows of `range` together, ahead (`exec::prefetch`); a pure filter is evaluated over chunks of rows in parallel.
fn filter_scan<'a>(input: &'a QueryInput, scan: &Scan<'a>, w: &ExprRef, range: std::ops::Range<usize>, want: usize) -> Result<Vec<usize>> {
    if want == 0 || range.is_empty() {
        return Ok(Vec::new());
    }
    let empty = HashMap::new();
    let start = range.start;
    let n = range.len();
    let calls = prefetch_calls(input, &[w], n, &|i, f| scan.with_row(start + i, |row| f(row, &[])), &empty)?;
    let fast = input.compile(w);
    // the rows of a run that pass, up to its first failure (or until `want` of them passed)
    let run = |r: std::ops::Range<usize>| -> (Vec<usize>, Option<Error>) {
        let mut out = Vec::new();
        for i in r {
            let res = scan.with_row(start + i, |row| {
                let ctx = Ctx { input, row, pre: &[], slots: &empty, calls: slot_of(&calls, i) };
                fast.eval_bool(row, &ctx)
            });
            match res {
                Ok(Some(true)) => {
                    out.push(start + i);
                    if out.len() >= want {
                        break;
                    }
                }
                Ok(_) => {}
                Err(err) => return (out, Some(super::compute::eval_error(w, err))),
            }
        }
        (out, None)
    };
    let vector = match (scan, &calls) {
        (Scan::Leaf { data, addrs }, None) => VPred::compile(&fast).map(|v| (v, *data, addrs)),
        _ => None,
    };
    let parts = match &vector {
        // a filter of a base table's cells: over runs of rows of a storage chunk at once, from its typed columns
        Some((vpred, data, addrs)) => {
            let mut cols = Vec::new();
            vpred.columns(&mut cols);
            let run_vec = |r: std::ops::Range<usize>| -> (Vec<usize>, Option<Error>) {
                let mut out = Vec::new();
                for (a, b, ok) in super::fast::filter_runs(data, &addrs[start..start + n], vpred, &cols, r) {
                    if ok.is_empty() {
                        // a column the filter cannot read in typed form: these rows one by one
                        let (part, failed) = run(a..b);
                        out.extend(part);
                        if failed.is_some() || out.len() >= want {
                            return (out, failed);
                        }
                        continue;
                    }
                    for (k, v) in ok.iter().enumerate() {
                        if *v == super::fast::V_TRUE {
                            out.push(start + a + k);
                            if out.len() >= want {
                                return (out, None);
                            }
                        }
                    }
                }
                (out, None)
            };
            super::par::map_chunks(n, run_vec)
        }
        None if input.expr_pure(w, &calls) => super::par::map_chunks(n, run),
        None => vec![run(0..n)],
    };
    let mut out = Vec::with_capacity(parts.iter().map(|(p, _)| p.len()).sum::<usize>().min(want));
    for (part, failed) in parts {
        let room = want - out.len();
        if part.len() >= room {
            out.extend(part.into_iter().take(room));
            return Ok(out);
        }
        out.extend(part);
        if let Some(err) = failed {
            return Err(err);
        }
    }
    Ok(out)
}

/// The rows after joins and the where clause.
fn filtered<'a>(input: &'a QueryInput) -> Result<Rows<'a>> {
    Ok(filtered_scan(input)?.into_rows())
}

/// `filtered()`, the rows of a base table by address.
fn filtered_scan<'a>(input: &'a QueryInput) -> Result<Scan<'a>> {
    let scan = if input.q.joins.is_empty() { Scan::new(input, None)? } else { Scan::Rows(scan_and_join(input)?) };
    match &input.q.where_ {
        None => Ok(scan),
        Some(w) => {
            let idx = filter_scan(input, &scan, w, 0..scan.len(), usize::MAX)?;
            Ok(match scan {
                Scan::Leaf { data, addrs } => Scan::Leaf { data, addrs: idx.into_iter().map(|i| addrs[i]).collect() },
                scan => Scan::Rows(scan.select(&idx)),
            })
        }
    }
}

/// The first `need` rows of `scan` that pass the where clause (all rows without one), in order: the filter is
/// evaluated over batches of rows of growing size until enough rows passed (as SQL's LIMIT stops a scan), so rows after
/// those are not evaluated and their user-function calls not made.
fn first_rows<'a>(input: &'a QueryInput, scan: Scan<'a>, need: usize) -> Result<Scan<'a>> {
    let n = scan.len();
    let Some(w) = &input.q.where_ else {
        return Ok(scan.pick(0..need.min(n)));
    };
    let mut out: Vec<usize> = Vec::new();
    let (mut start, mut batch) = (0, need.saturating_mul(2).max(1024));
    while start < n && out.len() < need {
        let end = start.saturating_add(batch).min(n);
        out.extend(filter_scan(input, &scan, w, start..end, need - out.len())?);
        start = end;
        batch = batch.saturating_mul(2).min(1 << 18);
    }
    Ok(scan.pick(out.into_iter()))
}

// ------------------------------------------------------------------------------------------------ sampling

/// The rank of a row in a sample (`sample::key_hash()` of the key of its first table's row).
fn row_hash(seed: u64, row: &[Option<&StoredRow>]) -> u64 {
    match row.first().copied().flatten() {
        Some(r) => super::sample::key_hash(seed, r.rowid, &r.pos),
        None => super::sample::splitmix64(seed),
    }
}

/// The seed of a sample: the given one, or a fresh random one per execution.
pub fn sample_seed(s: &SampleClause) -> u64 {
    match s.seed {
        Some(sd) => sd as u64,
        None => super::sample::random_u64(),
    }
}

/// Upstream `SqlSampleNode`: rows are ranked by their hash; a fraction keeps the rows whose hash falls below the
/// fraction of the hash space; `n` takes the lowest-ranked rows; stratified samples (`strata`: one key per row) rank
/// within strata (`n_per_stratum` per stratum, a fraction rounded up per stratum, `n` in rank order across strata).
/// Returns the indices of the selected rows, in rank order.
pub fn sample_select(hashes: &[u64], strata: Option<&[String]>, s: &SampleClause) -> Vec<usize> {
    let mut order: Vec<usize> = (0..hashes.len()).collect();
    order.sort_by_key(|i| hashes[*i]);
    let Some(strata) = strata else {
        return match (s.fraction, s.n) {
            (Some(f), _) => order.into_iter().filter(|i| super::sample::fraction_keeps(hashes[*i], f)).collect(),
            (None, Some(n)) => order.into_iter().take(n.max(0) as usize).collect(),
            _ => order,
        };
    };
    let mut ids: HashMap<&str, usize> = HashMap::new();
    let mut counts: Vec<usize> = Vec::new();
    // (stratum, rank in stratum, row)
    let mut ranked: Vec<(usize, usize, usize)> = Vec::with_capacity(order.len());
    for i in order {
        let next = ids.len();
        let sid = *ids.entry(strata[i].as_str()).or_insert(next);
        if sid == counts.len() {
            counts.push(0);
        }
        ranked.push((sid, counts[sid], i));
        counts[sid] += 1;
    }
    let mut out: Vec<(usize, usize)> = match (s.n_per_stratum, s.n, s.fraction) {
        (Some(per), _, _) => ranked.into_iter().filter(|(_, rank, _)| (*rank as i64) < per).map(|(_, rk, i)| (rk, i)).collect(),
        (None, Some(n), _) => {
            let mut v: Vec<(usize, usize)> = ranked.into_iter().map(|(_, rk, i)| (rk, i)).collect();
            v.sort_by(|a, b| a.0.cmp(&b.0).then(hashes[a.1].cmp(&hashes[b.1])));
            v.truncate(n.max(0) as usize);
            v
        }
        (None, None, Some(f)) => ranked
            .into_iter()
            .filter(|(sid, rank, _)| (*rank as f64) < (f * counts[*sid] as f64).ceil())
            .map(|(_, rk, i)| (rk, i))
            .collect(),
        _ => Vec::new(),
    };
    out.sort_by_key(|(_, i)| hashes[*i]);
    out.into_iter().map(|(_, i)| i).collect()
}

/// Samples the rows of a query (see `sample_select()`).
fn apply_sample<'a>(input: &'a QueryInput, rows: Rows<'a>, s: &SampleClause) -> Result<Rows<'a>> {
    let seed = sample_seed(s);
    let hashes: Vec<u64> = (0..rows.len()).map(|i| row_hash(seed, rows.row(i))).collect();
    let strata = if s.is_stratified() {
        let empty = HashMap::new();
        let mut keys = Vec::with_capacity(rows.len());
        for i in 0..rows.len() {
            let ctx = Ctx { input, row: rows.row(i), pre: &[], slots: &empty, calls: None };
            keys.push(keys_of(&s.stratify_by.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?));
        }
        Some(keys)
    } else {
        None
    };
    let selected = sample_select(&hashes, strata.as_deref(), s);
    Ok(rows.gather(selected.into_iter()))
}

// ------------------------------------------------------------------------------------------------ aggregation

fn call_args(e: &ExprRef, ctx: &dyn RowContext) -> Result<(Vec<Value>, Vec<Value>)> {
    let fc = e.as_function_call().ok_or_else(|| Error::internal("aggregate call expected"))?;
    let args = fc.args.iter().map(|a| eval(a, ctx)).collect::<Result<Vec<_>>>()?;
    let kwargs = fc.kwargs.iter().map(|(_, a)| eval(a, ctx)).collect::<Result<Vec<_>>>()?;
    Ok((args, kwargs))
}

fn agg_error(e: &ExprRef, err: Error) -> Error {
    super::compute::eval_error(e, err)
}

/// The groups of a run of rows during aggregation: (key, first row, states) in order of first occurrence, found by key
/// (keys of one integer, the most common, through their own map).
#[derive(Default)]
struct Groups {
    by_key: FxHashMap<GroupKey, usize>,
    by_int: FxHashMap<i64, usize>,
    list: Vec<(GroupKey, usize, Vec<Box<dyn AggState>>)>,
}

impl Groups {
    /// The group of key `k` (`row`: the row that has it), created with `new_states()` if new.
    fn get(&mut self, k: GroupKey, row: usize, new_states: &dyn Fn() -> Result<Vec<Box<dyn AggState>>>) -> Result<usize> {
        if let [KeyPart::Num(i)] = k.as_slice() {
            return self.get_int(*i, row, new_states);
        }
        if let Some(gi) = self.by_key.get(&k) {
            return Ok(*gi);
        }
        self.list.push((k.clone(), row, new_states()?));
        self.by_key.insert(k, self.list.len() - 1);
        Ok(self.list.len() - 1)
    }

    /// `get()` of the key of one integer `i`.
    fn get_int(&mut self, i: i64, row: usize, new_states: &dyn Fn() -> Result<Vec<Box<dyn AggState>>>) -> Result<usize> {
        if let Some(gi) = self.by_int.get(&i) {
            return Ok(*gi);
        }
        self.list.push((smallvec![KeyPart::Num(i)], row, new_states()?));
        self.by_int.insert(i, self.list.len() - 1);
        Ok(self.list.len() - 1)
    }
}

/// A value of a typed chunk column (`ColChunk`), as the cell holds it.
fn typed_value(c: &crate::store::table_data::ColChunk, k: usize) -> Value {
    match c {
        ColChunk::Int(t) if !t.is_null(k) => Value::Int(t.vals[k]),
        ColChunk::Float(t) if !t.is_null(k) => Value::Float(t.vals[k]),
        ColChunk::Timestamp(t) if !t.is_null(k) => Value::Timestamp(t.vals[k]),
        ColChunk::Str(t) if !t.is_null(k) => Value::String(t.get(k).to_string()),
        _ => Value::Null,
    }
}

/// `key_part()` of a value of a typed chunk column (a string copied once).
fn typed_key_part(c: &crate::store::table_data::ColChunk, k: usize) -> KeyPart {
    match c {
        ColChunk::Str(t) if !t.is_null(k) => KeyPart::Str(t.get(k).to_string()),
        c => key_part(&typed_value(c, k)),
    }
}

/// Groups `rows` by the grouping keys (groups in key order, ASC with NULLs last) and computes the aggregates.
fn aggregate<'a>(input: &'a QueryInput, plan: &Plan, rows: &Scan<'a>, grouping: &[ExprRef]) -> Result<OutRows<'a>> {
    let empty = HashMap::new();
    struct Group {
        key: Vec<Value>,
        /// the group's first row
        rep: usize,
        states: Vec<Box<dyn AggState>>,
    }
    let mut index: FxHashMap<GroupKey, usize> = FxHashMap::default();
    let mut groups: Vec<Group> = Vec::new();
    let new_states = || -> Result<Vec<Box<dyn AggState>>> {
        plan.aggs.iter().map(|a| a.plan.new_state().map_err(|e| agg_error(&a.expr, e))).collect()
    };
    let mut exprs: Vec<&ExprRef> = grouping.iter().collect();
    let mut arg_fasts: Vec<(usize, Vec<Fast>)> = Vec::with_capacity(plan.aggs.len());
    for a in &plan.aggs {
        let fc = a.expr.as_function_call().ok_or_else(|| Error::internal("aggregate call expected"))?;
        exprs.extend(fc.args.iter().chain(fc.kwargs.iter().map(|(_, e)| e)));
        let fasts = fc.args.iter().chain(fc.kwargs.iter().map(|(_, e)| e)).map(|e| input.compile(e)).collect();
        arg_fasts.push((fc.args.len(), fasts));
    }
    let key_fasts: Vec<Fast> = grouping.iter().map(|e| input.compile(e)).collect();
    let n = rows.len();
    let calls = prefetch_calls(input, &exprs, n, &|i, f| rows.with_row(i, |r| f(r, &[])), &empty)?;
    let pure = exprs.iter().all(|e| input.expr_pure(e, &calls));
    // the key of row `r`: (its hashable form, its values if `values`)
    let eval_key = |ctx: &Ctx<'_>, r: &[Option<&StoredRow>], values: bool| -> Result<(GroupKey, Vec<Value>)> {
        match plan.group_slot {
            // the key of the base row (rowid, then iterator positions)
            Some(slot) => {
                let vals: Vec<Value> = match r[slot] {
                    Some(b) => std::iter::once(b.rowid).chain(b.pos.iter().copied()).map(Value::Int).collect(),
                    None => Vec::new(),
                };
                Ok((group_key_of(&vals), vals))
            }
            None => {
                let mut key = GroupKey::new();
                let mut vals = Vec::new();
                for (f, e) in key_fasts.iter().zip(grouping) {
                    let v = f.eval(r, ctx).map_err(|err| agg_error(e, err))?;
                    key.push(key_part(&v));
                    if values {
                        vals.push(v.into_owned());
                    }
                }
                Ok((key, vals))
            }
        }
    };
    // without grouping keys, every row is in the one group
    let one_group = grouping.is_empty() && plan.group_slot.is_none();
    // Aggregates row `i` into `part`, row by row in order.
    let aggregate_row = |i: usize, part: &mut Groups, args: &mut SmallVec<[Value; 4]>| -> Result<()> {
        rows.with_row(i, |r| {
            let ctx = Ctx { input, row: r, pre: &[], slots: &empty, calls: slot_of(&calls, i) };
            let gi = if one_group && !part.list.is_empty() {
                0
            } else {
                match (key_fasts.as_slice(), plan.group_slot) {
                    // a key of one integer, without building the key
                    ([f], None) if !one_group => {
                        let v = f.eval(r, &ctx).map_err(|err| agg_error(&grouping[0], err))?;
                        match &*v {
                            Value::Int(x) => part.get_int(*x, i, &new_states)?,
                            other => part.get(smallvec![key_part(other)], i, &new_states)?,
                        }
                    }
                    _ => {
                        let (k, _) = eval_key(&ctx, r, false)?;
                        part.get(k, i, &new_states)?
                    }
                }
            };
            for ((a, (na, fasts)), state) in plan.aggs.iter().zip(&arg_fasts).zip(part.list[gi].2.iter_mut()) {
                if let ([f], 1) = (fasts.as_slice(), *na) {
                    // one argument: passed without copying it
                    let v = f.eval(r, &ctx).map_err(|err| agg_error(&a.expr, err))?;
                    a.plan.update_slices(state.as_mut(), std::slice::from_ref(&*v), &[]).map_err(|err| agg_error(&a.expr, err))?;
                    continue;
                }
                args.clear();
                for f in fasts {
                    args.push(f.eval(r, &ctx).map_err(|err| agg_error(&a.expr, err))?.into_owned());
                }
                a.plan.update_slices(state.as_mut(), &args[..*na], &args[*na..]).map_err(|err| agg_error(&a.expr, err))?;
            }
            Ok(())
        })
    };
    // Rows of a base table whose keys and arguments are all its numeric cells: aggregated from the typed columns of
    // the storage chunks (`ColChunk`), run by run, as `aggregate_row()` would.
    let cell = |f: &Fast| match f {
        Fast::Cell { slot: 0, col } => Some(*col),
        _ => None,
    };
    let typed: Option<(Vec<ColId>, Vec<ColId>)> = match rows {
        Scan::Leaf { .. } if calls.is_none() && plan.group_slot.is_none() => {
            let keys: Option<Vec<ColId>> = key_fasts.iter().map(cell).collect();
            let args: Option<Vec<ColId>> = arg_fasts
                .iter()
                .map(|(na, fasts)| match fasts.as_slice() {
                    [f] if *na == 1 => cell(f),
                    _ => None,
                })
                .collect();
            keys.zip(args)
        }
        _ => None,
    };
    // The groups of a run of rows, which are aggregated in order; the run stops at its first failure, which is
    // returned with it.
    let aggregate_rows = |range: std::ops::Range<usize>| -> (Groups, Option<Error>) {
        let mut part = Groups::default();
        let mut args: SmallVec<[Value; 4]> = SmallVec::new();
        let (Some((key_cols, arg_cols)), Scan::Leaf { data, addrs }) = (&typed, rows) else {
            for i in range {
                if let Err(err) = aggregate_row(i, &mut part, &mut args) {
                    return (part, Some(err));
                }
            }
            return (part, None);
        };
        let mut cols: Vec<ColId> = key_cols.clone();
        cols.extend(arg_cols.iter().copied());
        cols.dedup();
        for (a, b) in super::fast::chunk_runs(addrs, range) {
            let chunk: Vec<(ColId, std::sync::Arc<ColChunk>)> = cols.iter().map(|c| (*c, data.column_at(addrs[a], *c).0)).collect();
            let col = |id: &ColId| chunk.iter().find(|(c, _)| c == id).map(|(_, c)| c.as_ref()).expect("column of the run");
            let off = addrs[a].1 as usize % crate::store::segment::CHUNK_ROWS;
            let keys: Vec<&ColChunk> = key_cols.iter().map(col).collect();
            let vals: Vec<&ColChunk> = arg_cols.iter().map(col).collect();
            // (string arguments are passed as the cells are, not copied)
            if keys.iter().any(|c| matches!(c, ColChunk::Mixed)) || vals.iter().any(|c| matches!(c, ColChunk::Mixed | ColChunk::Str(_))) {
                for i in a..b {
                    if let Err(err) = aggregate_row(i, &mut part, &mut args) {
                        return (part, Some(err));
                    }
                }
                continue;
            }
            for i in a..b {
                let k = off + (i - a);
                let step = (|| -> Result<()> {
                    let gi = if one_group && !part.list.is_empty() {
                        0
                    } else {
                        match keys.as_slice() {
                            [ColChunk::Int(t)] if !t.is_null(k) => part.get_int(t.vals[k], i, &new_states)?,
                            _ => part.get(keys.iter().map(|c| typed_key_part(c, k)).collect(), i, &new_states)?,
                        }
                    };
                    for ((ag, c), state) in plan.aggs.iter().zip(&vals).zip(part.list[gi].2.iter_mut()) {
                        let v = typed_value(c, k);
                        ag.plan.update_slices(state.as_mut(), std::slice::from_ref(&v), &[]).map_err(|err| agg_error(&ag.expr, err))?;
                    }
                    Ok(())
                })();
                if let Err(err) = step {
                    return (part, Some(err));
                }
            }
        }
        (part, None)
    };
    // pure arguments of aggregates whose states merge: chunks of rows in parallel, merged in row order
    let chunked = pure && plan.aggs.iter().all(|a| a.plan.func.mergeable()) && super::par::worth_parallel(n);
    let parts = if chunked {
        super::par::map_chunks(n, aggregate_rows)
    } else if plan.aggs.iter().any(|a| a.plan.func.calls_python()) {
        // a Python aggregate's calls, one after the other: attached once for all of them
        vec![pyo3::Python::attach(|_| aggregate_rows(0..n))]
    } else {
        vec![aggregate_rows(0..n)]
    };
    if parts.len() > 1 && parts.iter().all(|(_, failed)| failed.is_none()) {
        // the partial states of each group, in chunk order, merged group by group in parallel
        let mut firsts: Vec<usize> = Vec::new();
        // per group: (part, position among the part's groups, states)
        let mut partials: Vec<Vec<(usize, usize, Vec<Box<dyn AggState>>)>> = Vec::new();
        for (pi, (part, _)) in parts.into_iter().enumerate() {
            for (pos, (k, first_row, states)) in part.list.into_iter().enumerate() {
                let gi = *index.entry(k).or_insert_with(|| {
                    firsts.push(first_row);
                    partials.push(Vec::new());
                    firsts.len() - 1
                });
                partials[gi].push((pi, pos, states));
            }
        }
        type Merged = std::result::Result<Vec<Box<dyn AggState>>, (usize, usize, Error)>;
        let merged: Vec<Merged> = super::par::detached(|| {
            use rayon::prelude::*;
            partials
                .into_par_iter()
                .map(|ps| -> Merged {
                    let mut it = ps.into_iter();
                    let Some((_, _, mut acc)) = it.next() else { return Ok(Vec::new()) };
                    for (pi, pos, mut later) in it {
                        for ((a, state), l) in plan.aggs.iter().zip(acc.iter_mut()).zip(later.iter_mut()) {
                            state.merge(l.as_mut()).map_err(|err| (pi, pos, agg_error(&a.expr, err)))?;
                        }
                    }
                    Ok(acc)
                })
                .collect()
        });
        // the first merge failure in the order of a sequential merge (part, then group)
        let first_failure = merged.iter().filter_map(|m| m.as_ref().err()).map(|(pi, pos, _)| (*pi, *pos)).min();
        for (first_row, states) in firsts.into_iter().zip(merged) {
            let states = match states {
                Ok(states) => states,
                Err((pi, pos, err)) if Some((pi, pos)) == first_failure => return Err(err),
                Err(_) => continue,
            };
            // the key's values (to order the groups), evaluated again for the group's first row
            let key = rows.with_row(first_row, |r| {
                let ctx = Ctx { input, row: r, pre: &[], slots: &empty, calls: slot_of(&calls, first_row) };
                eval_key(&ctx, r, true).map(|(_, key)| key)
            })?;
            groups.push(Group { key, rep: first_row, states });
        }
    } else {
        for (part, failed) in parts {
            for (k, first_row, mut states) in part.list {
                match index.get(&k) {
                    Some(gi) => {
                        for ((a, state), later) in plan.aggs.iter().zip(groups[*gi].states.iter_mut()).zip(states.iter_mut()) {
                            state.merge(later.as_mut()).map_err(|err| agg_error(&a.expr, err))?;
                        }
                    }
                    None => {
                        // the key's values (to order the groups), evaluated again for the group's first row
                        let key = rows.with_row(first_row, |r| {
                            let ctx = Ctx { input, row: r, pre: &[], slots: &empty, calls: slot_of(&calls, first_row) };
                            eval_key(&ctx, r, true).map(|(_, key)| key)
                        })?;
                        groups.push(Group { key, rep: first_row, states });
                        index.insert(k, groups.len() - 1);
                    }
                }
            }
            if let Some(err) = failed {
                return Err(err);
            }
        }
    }
    if groups.is_empty() && grouping.is_empty() && plan.group_slot.is_none() {
        // an aggregation without grouping returns one row, also for empty input (SQL semantics)
        let states = new_states()?;
        let mut reps = Rows::new(input.width);
        reps.slots.resize(input.width, None);
        let mut pre = Vec::with_capacity(plan.aggs.len());
        for (a, mut st) in plan.aggs.iter().zip(states) {
            pre.push(st.value().map_err(|err| agg_error(&a.expr, err))?);
        }
        return Ok(OutRows { rows: Scan::Rows(reps), pre: vec![pre] });
    }
    groups.sort_by(|a, b| {
        for (x, y) in a.key.iter().zip(b.key.iter()) {
            let o = cmp_values(x, y);
            if o.is_ne() {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    let reps: Vec<usize> = groups.iter().map(|g| g.rep).collect();
    let mut pres = Vec::with_capacity(groups.len());
    for mut g in groups {
        let mut pre = Vec::with_capacity(plan.aggs.len());
        for (a, st) in plan.aggs.iter().zip(g.states.iter_mut()) {
            pre.push(st.value().map_err(|err| agg_error(&a.expr, err))?);
        }
        pres.push(pre);
    }
    Ok(OutRows { rows: Scan::Rows(rows.select(&reps)), pre: pres })
}

/// Sorts `rows` by `ordering` (stable; ASC puts NULLs last, DESC first; a wildcard direction sorts ascending).
fn sort_rows<'a>(input: &'a QueryInput, plan: &Plan, rows: OutRows<'a>, ordering: &[(ExprRef, Option<bool>)]) -> Result<OutRows<'a>> {
    sort_rows_upto(input, plan, rows, ordering, None)
}

/// Compares sort keys by `ordering` (keys of the same length).
fn cmp_sort_keys<A: std::borrow::Borrow<Value>, B: std::borrow::Borrow<Value>>(
    ordering: &[(ExprRef, Option<bool>)],
    a: &[A],
    b: &[B],
) -> std::cmp::Ordering {
    for (i, (_, asc)) in ordering.iter().enumerate() {
        let ord = cmp_values(a[i].borrow(), b[i].borrow());
        let ord = if asc.unwrap_or(true) { ord } else { ord.reverse() };
        if ord.is_ne() {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

/// `sort_rows()`; with `keep`, only the first `keep` rows of the sorted order are returned (selected without sorting
/// the rest: each run of rows keeps its first `keep`, in order, then those are sorted).
fn sort_rows_upto<'a>(
    input: &'a QueryInput,
    plan: &Plan,
    rows: OutRows<'a>,
    ordering: &[(ExprRef, Option<bool>)],
    keep: Option<usize>,
) -> Result<OutRows<'a>> {
    if ordering.is_empty() {
        return Ok(rows);
    }
    let n = rows.len();
    let exprs: Vec<&ExprRef> = ordering.iter().map(|(e, _)| e).collect();
    let calls = prefetch_calls(input, &exprs, n, &|i, f| rows.rows.with_row(i, |r| f(r, rows.pre(i))), &plan.slots)?;
    let pure = exprs.iter().all(|e| input.expr_pure(e, &calls));
    let fasts: Vec<Fast> = exprs.iter().map(|e| input.compile(e)).collect();
    let row_keys = |i: usize| -> Result<SmallVec<[std::borrow::Cow<'_, Value>; 2]>> {
        rows.rows.with_row(i, |row| {
            let ctx = Ctx { input, row, pre: rows.pre(i), slots: &plan.slots, calls: slot_of(&calls, i) };
            fasts
                .iter()
                .zip(&exprs)
                .map(|(f, e)| f.eval(row, &ctx).map_err(|err| super::compute::eval_error(e, err)))
                .collect()
        })
    };
    if keep == Some(0) {
        return Ok(rows.window(0, Some(0)));
    }
    if let Some(order) = numeric_order(&rows, ordering, &fasts, calls.is_none() && pure, keep) {
        return Ok(rows.permute(&order));
    }
    if let Some(k) = keep.filter(|k| *k < n / 2) {
        // the first `k` rows of a run (by key, ties by position), in order
        let firsts = |range: std::ops::Range<usize>| -> Result<Vec<(SmallVec<[Value; 2]>, u32)>> {
            let mut best: Vec<(SmallVec<[Value; 2]>, u32)> = Vec::with_capacity(k + 1);
            for i in range {
                let ks = row_keys(i)?;
                // rows come in order, so a tie with the k-th row does not displace it
                if best.len() == k && cmp_sort_keys(ordering, &ks, &best[k - 1].0).is_ge() {
                    continue;
                }
                let at = best.partition_point(|(bk, _)| cmp_sort_keys(ordering, bk, &ks).is_le());
                best.insert(at, (ks.into_iter().map(std::borrow::Cow::into_owned).collect(), i as u32));
                best.truncate(k);
            }
            Ok(best)
        };
        let parts = if pure { super::par::map_chunks(n, firsts) } else { vec![firsts(0..n)] };
        let mut candidates = Vec::new();
        for part in parts {
            candidates.extend(part?);
        }
        candidates.sort_by(|(ka, a), (kb, b)| cmp_sort_keys(ordering, ka, kb).then(a.cmp(b)));
        candidates.truncate(k);
        let order: Vec<u32> = candidates.into_iter().map(|(_, i)| i).collect();
        return Ok(rows.permute(&order));
    }
    let keys = super::par::try_map_rows(n, pure, row_keys)?;
    // the order of row indices: by key, ties by position (a stable sort)
    let cmp = |a: &u32, b: &u32| cmp_sort_keys(ordering, &keys[*a as usize], &keys[*b as usize]).then(a.cmp(b));
    let mut order: Vec<u32> = (0..n as u32).collect();
    super::par::sort_by(&mut order, &cmp);
    if let Some(k) = keep {
        order.truncate(k);
    }
    drop(keys);
    Ok(rows.permute(&order))
}

/// A sort key that is a number or NULL, ordered as `cmp_values()` orders them.
#[derive(Clone, Copy)]
enum NumKey {
    Null,
    Int(i64),
    Float(f64),
}

fn cmp_num_keys(a: &NumKey, b: &NumKey) -> std::cmp::Ordering {
    use crate::value::cmp_f64;
    use std::cmp::Ordering;
    match (a, b) {
        (NumKey::Null, NumKey::Null) => Ordering::Equal,
        (NumKey::Null, _) => Ordering::Greater,
        (_, NumKey::Null) => Ordering::Less,
        (NumKey::Int(x), NumKey::Int(y)) => x.cmp(y),
        (NumKey::Float(x), NumKey::Float(y)) => cmp_f64(*x, *y),
        (NumKey::Int(x), NumKey::Float(y)) => cmp_f64(*x as f64, *y),
        (NumKey::Float(x), NumKey::Int(y)) => cmp_f64(*x, *y as f64),
    }
}

/// For an ordering by one numeric key that is a stored cell or a similarity lookup (and `pure`: nothing to prefetch),
/// the order `sort_rows_upto()` computes (the first `keep` rows), from keys computed without values: cells read as
/// numbers, similarities scored four at a time (`SimilarityQuery::scores()`). None if the ordering has another form or
/// a key is not a number or NULL.
fn numeric_order(rows: &OutRows<'_>, ordering: &[(ExprRef, Option<bool>)], fasts: &[Fast], pure: bool, keep: Option<usize>) -> Option<Vec<u32>> {
    static NULL: Value = Value::Null;
    let ([f], [(_, asc)], true) = (fasts, ordering, pure) else { return None };
    let asc = asc.unwrap_or(true);
    let n = rows.len();
    let num = |v: &Value| match v {
        Value::Null => Some(NumKey::Null),
        Value::Int(x) => Some(NumKey::Int(*x)),
        Value::Float(x) => Some(NumKey::Float(*x)),
        _ => None,
    };
    // the cell of `col` in the row in `slot` of row `i`
    let cell = |i: usize, slot: usize, col: ColId| -> &Value {
        match &rows.rows {
            Scan::Leaf { data, addrs } => data.row_by_addr(addrs[i]).cell(col),
            Scan::Rows(r) => r.row(i)[slot].map_or(&NULL, |r| r.cell(col)),
        }
    };
    let num_keys = |range: std::ops::Range<usize>| -> Option<Vec<NumKey>> {
        match (f, &rows.rows) {
            // a base table's cells: from the typed columns of its chunks, run by run
            (Fast::Cell { col, .. }, Scan::Leaf { data, addrs }) => {
                let mut out = Vec::with_capacity(range.len());
                for (a, b) in super::fast::chunk_runs(addrs, range) {
                    let (c, off) = data.column_at(addrs[a], *col);
                    for k in off..off + (b - a) {
                        out.push(match c.as_ref() {
                            ColChunk::Mixed => num(data.row_by_addr(addrs[a + k - off]).cell(*col))?,
                            c => match typed_value(c, k) {
                                Value::Int(x) => NumKey::Int(x),
                                Value::Float(x) => NumKey::Float(x),
                                Value::Null => NumKey::Null,
                                // timestamps order as `cmp_values()` orders them, not as numbers
                                _ => return None,
                            },
                        });
                    }
                }
                Some(out)
            }
            (Fast::Cell { slot, col }, _) => range.map(|i| num(cell(i, *slot, *col))).collect(),
            (Fast::Similarity { slot, col, query, .. }, _) => {
                if let Scan::Leaf { data, addrs } = &rows.rows {
                    data.decode_chunks_of(&addrs[range.clone()]);
                }
                let stored: Vec<&Value> = range.map(|i| cell(i, *slot, *col)).collect();
                Some(query.scores(&stored).into_iter().map(|s| s.map_or(NumKey::Null, NumKey::Float)).collect())
            }
            _ => None,
        }
    };
    let cmp = |a: &NumKey, b: &NumKey| if asc { cmp_num_keys(a, b) } else { cmp_num_keys(a, b).reverse() };
    match keep.filter(|k| *k < n / 2) {
        Some(k) => {
            // the first `k` rows of each run (by key, ties by position), then the first `k` of those
            let firsts = |range: std::ops::Range<usize>| -> Option<Vec<(NumKey, u32)>> {
                let start = range.start;
                let keys = num_keys(range)?;
                let mut best: Vec<(NumKey, u32)> = Vec::with_capacity(k + 1);
                for (j, key) in keys.into_iter().enumerate() {
                    if best.len() == k && cmp(&key, &best[k - 1].0).is_ge() {
                        continue;
                    }
                    let at = best.partition_point(|(bk, _)| cmp(bk, &key).is_le());
                    best.insert(at, (key, (start + j) as u32));
                    best.truncate(k);
                }
                Some(best)
            };
            let mut candidates = Vec::new();
            for part in super::par::map_chunks(n, firsts) {
                candidates.extend(part?);
            }
            candidates.sort_by(|(ka, a), (kb, b)| cmp(ka, kb).then(a.cmp(b)));
            candidates.truncate(k);
            Some(candidates.into_iter().map(|(_, i)| i).collect())
        }
        None => {
            let mut keys = Vec::with_capacity(n);
            for part in super::par::map_chunks(n, num_keys) {
                keys.extend(part?);
            }
            let mut order: Vec<u32> = (0..n as u32).collect();
            super::par::sort_by(&mut order, &|a: &u32, b: &u32| cmp(&keys[*a as usize], &keys[*b as usize]).then(a.cmp(b)));
            if let Some(k) = keep {
                order.truncate(k);
            }
            Some(order)
        }
    }
}

/// Evaluates the window function calls row by row in window order (upstream `FunctionCall.eval()` of a window
/// call): a new aggregator per partition (consecutive rows with equal `group_by` values), updated with each row and
/// read after each update.
fn eval_windows<'a>(input: &'a QueryInput, plan: &Plan, rows: &mut OutRows<'a>) -> Result<()> {
    let empty = HashMap::new();
    let base = plan.aggs.len();
    let n = rows.len();
    rows.pre.resize_with(n, Vec::new);
    for pre in rows.pre.iter_mut() {
        pre.resize(base + plan.windows.len(), Value::Null);
    }
    for (wi, w) in plan.windows.iter().enumerate() {
        let fc = w.expr.as_function_call().ok_or_else(|| Error::internal("window call expected"))?;
        let mut state: Option<Box<dyn AggState>> = None;
        let mut partition: Option<String> = None;
        for i in 0..n {
            let (part_vals, args, kwargs) = rows.rows.with_row(i, |row| -> Result<_> {
                let ctx = Ctx { input, row, pre: &[], slots: &empty, calls: None };
                let part_vals = fc.group_by.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>().map_err(|e| agg_error(&w.expr, e))?;
                let (args, kwargs) = call_args(&w.expr, &ctx).map_err(|e| agg_error(&w.expr, e))?;
                Ok((part_vals, args, kwargs))
            })?;
            let pkey = keys_of(&part_vals);
            if state.is_none() || partition.as_ref() != Some(&pkey) {
                state = Some(w.plan.new_state().map_err(|e| agg_error(&w.expr, e))?);
                partition = Some(pkey);
            }
            let st = state.as_mut().expect("state");
            let v = if w.plan.skips(&args, &kwargs) {
                Value::Null
            } else {
                w.plan.update(st.as_mut(), args, kwargs).map_err(|e| agg_error(&w.expr, e))?;
                st.value().map_err(|e| agg_error(&w.expr, e))?
            };
            rows.pre[i][base + wi] = v;
        }
    }
    Ok(())
}

/// Rows before order by / limit: filtered, sampled, aggregated, with window values. With `need` (a limit and nothing
/// that needs all rows: no ordering, aggregation, window functions, sample or join), only the first `need` rows.
fn result_rows<'a>(input: &'a QueryInput, plan: &Plan, rowid_desc: bool, need: Option<usize>) -> Result<OutRows<'a>> {
    let q = &input.q;
    let limited = need.filter(|_| {
        q.order_by.is_empty() && plan.grouping.is_none() && plan.windows.is_empty() && q.sample.is_none() && q.joins.is_empty()
    });
    if let (Some(g), None, true, true, false) = (&plan.grouping, &q.sample, plan.agg_order.is_empty(), plan.windows.is_empty(), rowid_desc) {
        // an aggregation reads the rows of a base table by address
        let scan = filtered_scan(input)?;
        return aggregate(input, plan, &scan, g);
    }
    // the rows that pass the filter (a base table's by address)
    let mut scan = match limited {
        Some(need) => {
            // without a filter, only the first `need` rows are read (unless the scan is reversed)
            let prefix = (q.where_.is_none() && !rowid_desc).then_some(need);
            let mut scan = Scan::new(input, prefix)?;
            if rowid_desc {
                scan.reverse();
            }
            first_rows(input, scan, need)?
        }
        None => {
            let mut scan = filtered_scan(input)?;
            if rowid_desc {
                scan.reverse();
            }
            scan
        }
    };
    if let Some(s) = &q.sample {
        scan = Scan::Rows(apply_sample(input, scan.into_rows(), s)?);
    }
    if !plan.agg_order.is_empty() {
        // ordered aggregates see their rows in order (groups keep the input order)
        scan = sort_rows(input, plan, OutRows { rows: scan, pre: Vec::new() }, &plan.agg_order)?.rows;
    }
    let mut out = match &plan.grouping {
        Some(g) => aggregate(input, plan, &scan, g)?,
        None => OutRows { rows: scan, pre: Vec::new() },
    };
    if !plan.windows.is_empty() {
        out = sort_rows(input, plan, out, &plan.window_order)?;
        eval_windows(input, plan, &mut out)?;
    }
    Ok(out)
}

/// Computes the query embedding of each similarity lookup in the query (once per expression, cached in the node)
/// before rows are evaluated on several threads; a failure is left to row evaluation to report.
fn init_similarities(input: &QueryInput) {
    let row: Vec<Option<&StoredRow>> = vec![None; input.width];
    let slots = HashMap::new();
    let ctx = Ctx { input, row: &row, pre: &[], slots: &slots, calls: None };
    let exprs = input.q.exprs();
    for e in exprs.iter().chain(input.select.iter().map(|(e, _)| e)) {
        e.visit(&mut |x| {
            if let ExprKind::Similarity(sim) = &x.kind {
                if sim.cache.get().is_none() {
                    if let Ok(v) = eval(&sim.item_embedding, &ctx) {
                        let _ = sim.cache.set(v);
                    }
                }
            }
        });
    }
}

/// Runs the query. `rowid_desc`: for `tail()`, scan in descending rowid order.
pub fn execute(input: &QueryInput, rowid_desc: bool) -> Result<ResultData> {
    let q = &input.q;
    let plan = Plan::new(input)?;
    init_similarities(input);
    let names: Vec<String> = input.select.iter().map(|(_, n)| n.clone()).collect();
    let types = input.select.iter().map(|(e, _)| e.col_type.clone()).collect();
    if q.limit == Some(0) {
        return Ok(ResultData { names, types, rows: Vec::new() });
    }
    let offset = q.offset.unwrap_or(0).max(0) as usize;
    let limit = q.limit.map(|n| n.max(0) as usize);
    // only the rows up to the limit are needed
    let need = limit.map(|n| offset.saturating_add(n));
    let mut rows = result_rows(input, &plan, rowid_desc, need)?;
    if !q.order_by.is_empty() {
        let ordering: Vec<(ExprRef, Option<bool>)> = q.order_by.iter().map(|(e, a)| (e.clone(), Some(*a))).collect();
        rows = sort_rows_upto(input, &plan, rows, &ordering, need)?;
    }
    let rows = rows.window(offset, limit);
    let exprs: Vec<&ExprRef> = input.select.iter().map(|(e, _)| e).collect();
    let calls = prefetch_calls(input, &exprs, rows.len(), &|i, f| rows.rows.with_row(i, |r| f(r, rows.pre(i))), &plan.slots)?;
    let pure = exprs.iter().all(|e| input.expr_pure(e, &calls));
    let fasts: Vec<Fast> = exprs.iter().map(|e| input.compile(e)).collect();
    if let Some(projected) = project_cells(&rows, &fasts, pure && calls.is_none()) {
        return Ok(ResultData { names, types, rows: projected });
    }
    if let Scan::Leaf { data, addrs } = &rows.rows {
        super::par::detached(|| data.decode_chunks_of(addrs));
    }
    let projected = super::par::try_map_rows(rows.len(), pure, |i| {
        rows.rows.with_row(i, |row| {
            let ctx = Ctx { input, row, pre: rows.pre(i), slots: &plan.slots, calls: slot_of(&calls, i) };
            let mut vals = Vec::with_capacity(fasts.len());
            for (f, e) in fasts.iter().zip(&exprs) {
                vals.push(f.eval(row, &ctx).map_err(|err| super::compute::eval_error(e, err))?.into_owned());
            }
            Ok(vals)
        })
    })?;
    Ok(ResultData { names, types, rows: projected })
}

/// The select list of a base table's rows when it is only stored cells: values taken from the typed columns of the
/// storage chunks where they have one (the rows' cells otherwise), run by run, without decoding rows the typed columns
/// cover. None for other select lists.
fn project_cells(rows: &OutRows<'_>, fasts: &[Fast], pure: bool) -> Option<Vec<Vec<Value>>> {
    let Scan::Leaf { data, addrs } = &rows.rows else { return None };
    let cols: Vec<ColId> = fasts
        .iter()
        .map(|f| match f {
            Fast::Cell { slot: 0, col } => Some(*col),
            _ => None,
        })
        .collect::<Option<_>>()?;
    if !pure || !rows.pre.is_empty() || cols.is_empty() {
        return None;
    }
    // the typed columns built in parallel, then the rows of the runs they do not cover decoded in parallel
    super::par::detached(|| data.build_columns_of(addrs, &cols));
    let runs = super::fast::chunk_runs(addrs, 0..addrs.len());
    let untyped: Vec<RowAddr> = runs
        .iter()
        .filter(|(a, _)| cols.iter().any(|c| matches!(data.column_at(addrs[*a], *c).0.as_ref(), ColChunk::Mixed)))
        .map(|(a, _)| addrs[*a])
        .collect();
    super::par::detached(|| data.decode_chunks_of(&untyped));
    let parts = super::par::map_chunks(addrs.len(), |range| {
        let mut out = Vec::with_capacity(range.len());
        for (a, b) in super::fast::chunk_runs(addrs, range) {
            let typed: Vec<(std::sync::Arc<ColChunk>, usize)> = cols.iter().map(|c| data.column_at(addrs[a], *c)).collect();
            for i in a..b {
                out.push(
                    typed
                        .iter()
                        .zip(&cols)
                        .map(|((c, off), col)| match c.as_ref() {
                            ColChunk::Mixed => data.row_by_addr(addrs[i]).cell(*col).clone(),
                            c => typed_value(c, off + (i - a)),
                        })
                        .collect(),
                );
            }
        }
        out
    });
    Some(parts.into_iter().flatten().collect())
}

/// `Query.count()`
pub fn count(input: &QueryInput) -> Result<usize> {
    let q = &input.q;
    if q.limit.is_some() || q.offset.is_some() {
        crate::bail!(
            UnsupportedOperation,
            "count() cannot be used with limit() or offset(). Use `select(pxtf.count())` instead."
        );
    }
    if let Some(g) = &q.group_by {
        validate_grouping(g, &input.unstored)?;
    }
    if let Some(w) = &q.where_ {
        if conjuncts(w).iter().any(|c| !is_sql_expr(c, &input.unstored)) {
            if q.sample.is_some() {
                let f = conjuncts(w).into_iter().find(|c| !is_sql_expr(c, &input.unstored)).expect("non-SQL conjunct");
                crate::bail!(UnsupportedOperation, "Filter {} not expressible in SQL", display(&f));
            }
            // upstream plans count(1) over the rows: with a Python filter, the root is a Python aggregation
            crate::bail!(UnsupportedOperation, "count() cannot be used: query plan contains a non-SQL node (AggregationNode)");
        }
    }
    if q.sample.is_none() && q.group_by.is_none() && q.grouping_tbl.is_none() && q.joins.is_empty() {
        if q.where_.is_none() {
            if let [src] = input.sources.as_slice() {
                if let [e] = src.path.elems.as_slice() {
                    // the visible rows of one table: counted by the store
                    return Ok(e.data.count_at(e.version));
                }
            }
        }
        let scan = Scan::new(input, None)?;
        return Ok(match &q.where_ {
            Some(w) => filter_scan(input, &scan, w, 0..scan.len(), usize::MAX)?.len(),
            None => scan.len(),
        });
    }
    let mut rows = filtered(input)?;
    if let Some(s) = &q.sample {
        rows = apply_sample(input, rows, s)?;
    }
    if let Some(t) = q.grouping_tbl {
        let slot = input.slot(&t).ok_or_else(|| Error::internal("grouping table is not in the table path"))?;
        let keys: std::collections::HashSet<(i64, &[i64])> = (0..rows.len()).filter_map(|i| rows.row(i)[slot]).map(|b| b.key()).collect();
        return Ok(keys.len());
    }
    match &q.group_by {
        Some(g) => {
            let empty = HashMap::new();
            let mut keys = std::collections::HashSet::new();
            for i in 0..rows.len() {
                let ctx = Ctx { input, row: rows.row(i), pre: &[], slots: &empty, calls: None };
                let vals = g.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
                keys.insert(keys_of(&vals));
            }
            Ok(keys.len())
        }
        None => Ok(rows.len()),
    }
}

// ------------------------------------------------------------------------------------- windows in computed columns

/// Evaluates against one joined row of a table path, with precomputed window values.
struct WindowCtx<'a> {
    path: PathCtx<'a>,
    pre: &'a [Value],
    slots: &'a HashMap<usize, usize>,
}

impl RowContext for WindowCtx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        self.path.column_value(col)
    }
    fn column_error(&self, col: &ColumnRef) -> Result<Option<crate::store::rows::CellError>> {
        self.path.column_error(col)
    }
    fn precomputed(&self, e: &Expr) -> Option<Value> {
        self.slots.get(&node_key(e)).map(|i| self.pre[*i].clone())
    }
    fn python_semantics(&self) -> bool {
        self.path.python_semantics()
    }
    fn row_key(&self, tbl_id: Uuid) -> Result<Option<(i64, Vec<i64>)>> {
        self.path.row_key(tbl_id)
    }
    fn call_result(&self, e: &Expr) -> Option<Result<Value>> {
        self.path.call_result(e)
    }
}

/// The number of components of the row keys of table `tbl_id`: the rowid, then one iterator position per component
/// view in its path.
pub fn num_rowid_columns(cat: &Catalog, tbl_id: Uuid) -> Result<usize> {
    let mut n = 1;
    for e in cat.tbl_path(tbl_id, None)? {
        if cat.table(e.tbl_id)?.is_component_view() {
            n += 1;
        }
    }
    Ok(n)
}

/// True if `e` contains a window function call.
pub fn has_window_calls(e: &ExprRef) -> bool {
    let mut out = Vec::new();
    collect_calls(e, &mut out, &|fc| fc.is_window());
    !out.is_empty()
}

/// Evaluates a computed column's value expression that contains window function calls over `rows` (upstream
/// computes such columns with a query plan that orders the rows by the window clauses). Returns one result per
/// row, in the order of `rows` (joined rows of the table path `infos`; `python`: evaluate with Python semantics).
pub fn eval_rows_with_windows(
    infos: &[&ElemInfo],
    rows: &[Vec<&StoredRow>],
    expr: &ExprRef,
    python: bool,
) -> Result<Vec<Result<Value>>> {

    let mut calls = Vec::new();
    collect_calls(expr, &mut calls, &|fc| fc.is_window());
    let mut slots = HashMap::new();
    let mut keys: HashMap<String, usize> = HashMap::new();
    let mut distinct: Vec<(ExprRef, AggPlan)> = Vec::new();
    for e in calls {
        let k = expr_key(&e);
        let idx = match keys.get(&k) {
            Some(i) => *i,
            None => {
                let fc = e.as_function_call().ok_or_else(|| Error::internal("window call expected"))?;
                distinct.push((e.clone(), AggPlan::new(fc)?));
                keys.insert(k, distinct.len() - 1);
                distinct.len() - 1
            }
        };
        slots.insert(node_key(&e), idx);
    }
    let mut ordering: Vec<(ExprRef, Option<bool>)> = Vec::new();
    for (e, _) in &distinct {
        let fc = e.as_function_call().ok_or_else(|| Error::internal("window call expected"))?;
        let mut o: Vec<(ExprRef, Option<bool>)> = fc.group_by.iter().map(|g| (g.clone(), None)).collect();
        o.extend(fc.order_by.iter().map(|x| (x.clone(), Some(true))));
        ordering = match combine_orderings(&ordering, &o) {
            Some(c) => c,
            None => crate::bail!(
                UnsupportedOperation,
                "Incompatible ordering requirements: {} vs {}",
                print_ordering(&ordering),
                print_ordering(&o)
            ),
        };
    }
    let no_slots = HashMap::new();
    let mut order: Vec<(Vec<Value>, usize)> = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let ctx = WindowCtx { path: PathCtx { elems: infos, rows: r, python, calls: None }, pre: &[], slots: &no_slots };
        order.push((ordering.iter().map(|(e, _)| eval(e, &ctx)).collect::<Result<Vec<_>>>()?, i));
    }
    order.sort_by(|(a, _), (b, _)| {
        for (k, (_, asc)) in ordering.iter().enumerate() {
            let o = cmp_values(&a[k], &b[k]);
            let o = if asc.unwrap_or(true) { o } else { o.reverse() };
            if o.is_ne() {
                return o;
            }
        }
        std::cmp::Ordering::Equal
    });
    let mut pre: Vec<Vec<Value>> = vec![vec![Value::Null; distinct.len()]; rows.len()];
    for (wi, (e, plan)) in distinct.iter().enumerate() {
        let fc = e.as_function_call().ok_or_else(|| Error::internal("window call expected"))?;
        let mut state: Option<Box<dyn AggState>> = None;
        let mut partition: Option<String> = None;
        for (_, i) in &order {
            let ctx = WindowCtx { path: PathCtx { elems: infos, rows: &rows[*i], python, calls: None }, pre: &[], slots: &no_slots };
            let pkey = keys_of(&fc.group_by.iter().map(|g| eval(g, &ctx)).collect::<Result<Vec<_>>>()?);
            if state.is_none() || partition.as_ref() != Some(&pkey) {
                state = Some(plan.new_state()?);
                partition = Some(pkey);
            }
            let (args, kwargs) = call_args(e, &ctx)?;
            let st = state.as_mut().expect("state");
            // a skipped row (NULL argument) still gets the running value
            plan.update(st.as_mut(), args, kwargs)?;
            pre[*i][wi] = st.value()?;
        }
    }
    Ok(rows
        .iter()
        .enumerate()
        .map(|(i, r)| eval(expr, &WindowCtx { path: PathCtx { elems: infos, rows: r, python, calls: None }, pre: &pre[i], slots: &slots }))
        .collect())
}
