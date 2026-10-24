//! Query execution: scan -> join -> filter -> sample -> aggregate -> window functions -> sort -> limit -> project.
//!
//! `QueryInput::snapshot()` captures, under the catalog lock, the table path (`scan::PathInput`: the table and the
//! bases its rows join with) of every from-table. `execute()` and `count()` run without the lock.
//!
//! Rows flow as `JRow`s: one optional stored row per slot, where each from-table occupies one slot per table of its
//! path (leaf first); None is the missing side of an outer join.
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
use crate::store::rows::StoredRow;
use crate::value::{Value, cmp_values};

/// One from-table of a query: its table path, captured under the catalog lock.
pub struct Source {
    pub path: PathInput,
    /// the slot of the path's first (leaf) table in a `JRow`
    pub offset: usize,
}

/// Everything a query reads, captured under the catalog lock.
pub struct QueryInput {
    pub q: QuerySpec,
    pub sources: Vec<Source>,
    pub select: Vec<(ExprRef, String)>,
    /// value expressions of unstored computed columns, evaluated on read
    pub unstored: HashMap<(Uuid, ColId), ExprRef>,
    /// number of slots of a `JRow`
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
        self.expr_pure_at(e, calls, 0)
    }

    fn expr_pure_at(&self, e: &Expr, calls: &Option<CallResults>, depth: usize) -> bool {
        let col_pure = |c: &ColumnRef| {
            let read = self.reads.get(&(c.tbl_id, c.col_id, c.tbl_version)).copied().unwrap_or_else(|| self.col_read(c));
            match (read, self.unstored.get(&(c.tbl_id, c.col_id))) {
                (ColRead::Slot(_), _) => true,
                // an unstored computed column is pure if its expression is (iterator outputs and media validated on
                // read are not)
                (ColRead::Path, Some(x)) => depth < 64 && self.expr_pure_at(x, calls, depth + 1),
                (ColRead::Path, None) => false,
            }
        };
        super::par::is_pure(e, &col_pure, &|n| calls.as_ref().is_some_and(|r| r.contains(n)))
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

    /// A row with only the slots of source `i` set.
    fn place<'a>(&self, jr: &mut JRow<'a>, i: usize, rows: &[&'a StoredRow]) {
        let off = self.sources[i].offset;
        for (k, r) in rows.iter().enumerate() {
            jr[off + k] = Some(*r);
        }
    }
}

/// One slot per table of each from-table's path (most queries have one or two).
type JRow<'a> = SmallVec<[Option<&'a StoredRow>; 2]>;

/// An output row: the (representative) input row and the precomputed aggregate/window values.
struct OutRow<'a> {
    row: JRow<'a>,
    pre: Vec<Value>,
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

/// The visible rows of source `i` (each joined with its path's base rows), in key order. Without joins, only the
/// rows the where clause can select when it fixes the table's primary key (`PathInput::rows_where()`).
fn source_rows<'a>(input: &'a QueryInput, i: usize) -> Result<Vec<Vec<&'a StoredRow>>> {
    if i == 0 && input.q.joins.is_empty() {
        return input.sources[0].path.rows_where(input.q.where_.as_ref());
    }
    input.sources[i].path.rows()
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

/// Scans the from-tables and joins them (in from-clause order).
fn scan_and_join<'a>(input: &'a QueryInput) -> Result<Vec<JRow<'a>>> {
    let n = input.width;
    let empty = HashMap::new();
    // the joined rows of the first from-table, as consecutive runs of `width` rows (leaf first), then one `JRow` each
    // (built in parallel: most of the cost is writing fresh memory)
    let first = &input.sources[0].path;
    let width = first.elems.len();
    let capacity = if input.q.where_.is_none() { first.elems[0].data.num_live() * width } else { 0 };
    let mut flat: Vec<&'a StoredRow> = Vec::with_capacity(capacity);
    let mut add = |r: &[&'a StoredRow]| flat.extend_from_slice(r);
    if input.q.joins.is_empty() {
        first.for_each_row_where(input.q.where_.as_ref(), &mut add)?;
    } else {
        first.for_each_row(&mut add)?;
    }
    let mut rows: Vec<JRow<'a>> = super::par::map_rows(flat.len() / width, |i| {
        let mut jr: JRow<'a> = smallvec![None; n];
        input.place(&mut jr, 0, &flat[i * width..(i + 1) * width]);
        jr
    });
    drop(flat);
    for (k, join) in input.q.joins.iter().enumerate() {
        let slot = k + 1;
        let right = source_rows(input, slot)?;
        let mut out: Vec<JRow<'a>> = Vec::new();
        let mut right_matched = vec![false; right.len()];
        let combined = |left: &JRow<'a>, r: &[&'a StoredRow]| -> JRow<'a> {
            let mut jr = left.clone();
            input.place(&mut jr, slot, r);
            jr
        };
        match (&join.on, join.how) {
            (None, _) | (_, JoinType::Cross) => {
                for l in &rows {
                    for r in &right {
                        out.push(combined(l, r));
                    }
                }
            }
            (Some(pred), how) => {
                let (lkeys, rkeys) = equi_keys(pred, input, slot);
                let pure = input.expr_pure(pred, &None);
                // candidate right rows per left row: by hash on the equality keys, else all
                let mut index: FxHashMap<GroupKey, Vec<usize>> = FxHashMap::default();
                if !lkeys.is_empty() {
                    let right_keys = super::par::try_map_rows(right.len(), pure, |j| {
                        let mut jr: JRow<'a> = smallvec![None; n];
                        input.place(&mut jr, slot, &right[j]);
                        let ctx = Ctx { input, row: &jr, pre: &[], slots: &empty, calls: None };
                        let vals = rkeys.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
                        Ok((!vals.iter().any(Value::is_null)).then(|| group_key_of(&vals)))
                    });
                    for (j, k) in right_keys.into_iter().enumerate() {
                        if let Some(k) = k? {
                            index.entry(k).or_default().push(j);
                        }
                    }
                }
                let all: Vec<usize> = (0..right.len()).collect();
                // the joined rows of a run of left rows, and the right rows they matched, up to the first failure
                let join_rows = |range: std::ops::Range<usize>| -> (Vec<JRow<'a>>, Vec<usize>, Option<Error>) {
                    let mut out = Vec::new();
                    let mut matched_right = Vec::new();
                    for i in range {
                        let l = &rows[i];
                        let step = (|| -> Result<()> {
                            let candidates: &[usize] = if lkeys.is_empty() {
                                &all
                            } else {
                                let ctx = Ctx { input, row: l, pre: &[], slots: &empty, calls: None };
                                let vals = lkeys.iter().map(|e| eval(e, &ctx)).collect::<Result<SmallVec<[Value; 2]>>>()?;
                                if vals.iter().any(Value::is_null) {
                                    &[]
                                } else {
                                    index.get(&group_key_of(&vals)).map(Vec::as_slice).unwrap_or(&[])
                                }
                            };
                            let mut matched = false;
                            for &j in candidates {
                                let jr = combined(l, &right[j]);
                                let ctx = Ctx { input, row: &jr, pre: &[], slots: &empty, calls: None };
                                if matches!(eval(pred, &ctx)?, Value::Bool(true)) {
                                    matched = true;
                                    matched_right.push(j);
                                    out.push(jr);
                                }
                            }
                            if !matched && matches!(how, JoinType::Left | JoinType::FullOuter) {
                                out.push(l.clone());
                            }
                            Ok(())
                        })();
                        if let Err(err) = step {
                            return (out, matched_right, Some(err));
                        }
                    }
                    (out, matched_right, None)
                };
                let parts = if pure { super::par::map_chunks(rows.len(), join_rows) } else { vec![join_rows(0..rows.len())] };
                out.reserve(parts.iter().map(|(p, _, _)| p.len()).sum());
                for (part, matched, failed) in parts {
                    if let Some(err) = failed {
                        return Err(err);
                    }
                    out.extend(part);
                    for j in matched {
                        right_matched[j] = true;
                    }
                }
                if how == JoinType::FullOuter {
                    for (j, r) in right.iter().enumerate() {
                        if !right_matched[j] {
                            let mut jr: JRow<'a> = smallvec![None; n];
                            input.place(&mut jr, slot, r);
                            out.push(jr);
                        }
                    }
                }
            }
        }
        rows = out;
    }
    Ok(rows)
}

/// The rows after joins and the where clause.
/// Phase A of evaluating `exprs` over `n` rows of a query (`row_of(i)`: the row and its precomputed values): makes
/// their user-function calls together (see `exec::prefetch`); a query stops at its first error.
fn prefetch_calls<'a>(
    input: &'a QueryInput,
    exprs: &[&ExprRef],
    n: usize,
    row_of: &dyn Fn(usize) -> (&'a [Option<&'a StoredRow>], &'a [Value]),
    slots: &'a HashMap<usize, usize>,
) -> Result<Option<CallResults>> {
    let derived = |c: &ColumnRef| input.unstored.get(&(c.tbl_id, c.col_id)).cloned();
    super::prefetch::prefetch(exprs, &derived, n, true, &|i, slot, f| {
        let (row, pre) = row_of(i);
        f(&Ctx { input, row, pre, slots, calls: Some(slot) })
    })
}

fn slot_of(calls: &Option<CallResults>, row: usize) -> Option<CallSlot<'_>> {
    calls.as_ref().map(|results| CallSlot { results, row })
}

fn filtered<'a>(input: &'a QueryInput) -> Result<Vec<JRow<'a>>> {
    let rows = scan_and_join(input)?;
    let Some(w) = &input.q.where_ else { return Ok(rows) };
    let empty = HashMap::new();
    let calls = prefetch_calls(input, &[w], rows.len(), &|i| (&rows[i], &[]), &empty)?;
    // the rows of a run that pass, up to its first failure
    let filter_rows = |range: std::ops::Range<usize>| -> (Vec<JRow<'a>>, Option<Error>) {
        let mut out = Vec::new();
        for i in range {
            let ctx = Ctx { input, row: &rows[i], pre: &[], slots: &empty, calls: slot_of(&calls, i) };
            match eval(w, &ctx) {
                Ok(Value::Bool(true)) => out.push(rows[i].clone()),
                Ok(_) => {}
                Err(err) => return (out, Some(super::compute::eval_error(w, err))),
            }
        }
        (out, None)
    };
    let parts = if input.expr_pure(w, &calls) {
        super::par::map_chunks(rows.len(), filter_rows)
    } else {
        vec![filter_rows(0..rows.len())]
    };
    let mut out = Vec::with_capacity(parts.iter().map(|(p, _)| p.len()).sum());
    for (part, failed) in parts {
        if let Some(err) = failed {
            return Err(err);
        }
        out.extend(part);
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------ sampling

/// The rank of a row in a sample (`sample::key_hash()` of the key of its first table's row).
fn row_hash(seed: u64, row: &JRow<'_>) -> u64 {
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
fn apply_sample<'a>(input: &'a QueryInput, rows: Vec<JRow<'a>>, s: &SampleClause) -> Result<Vec<JRow<'a>>> {
    let seed = sample_seed(s);
    let hashes: Vec<u64> = rows.iter().map(|r| row_hash(seed, r)).collect();
    let strata = if s.is_stratified() {
        let empty = HashMap::new();
        let mut keys = Vec::with_capacity(rows.len());
        for r in &rows {
            let ctx = Ctx { input, row: r, pre: &[], slots: &empty, calls: None };
            keys.push(keys_of(&s.stratify_by.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?));
        }
        Some(keys)
    } else {
        None
    };
    let selected = sample_select(&hashes, strata.as_deref(), s);
    let mut rows: Vec<Option<JRow<'a>>> = rows.into_iter().map(Some).collect();
    Ok(selected.into_iter().filter_map(|i| rows[i].take()).collect())
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

/// Groups `rows` by the grouping keys (groups in key order, ASC with NULLs last) and computes the aggregates.
fn aggregate<'a>(input: &'a QueryInput, plan: &Plan, rows: Vec<JRow<'a>>, grouping: &[ExprRef]) -> Result<Vec<OutRow<'a>>> {
    let empty = HashMap::new();
    struct Group<'a> {
        key: Vec<Value>,
        rep: JRow<'a>,
        states: Vec<Box<dyn AggState>>,
    }
    let mut index: FxHashMap<GroupKey, usize> = FxHashMap::default();
    let mut groups: Vec<Group<'a>> = Vec::new();
    let new_states = || -> Result<Vec<Box<dyn AggState>>> {
        plan.aggs.iter().map(|a| a.plan.new_state().map_err(|e| agg_error(&a.expr, e))).collect()
    };
    let mut exprs: Vec<&ExprRef> = grouping.iter().collect();
    for a in &plan.aggs {
        if let Some(fc) = a.expr.as_function_call() {
            exprs.extend(fc.args.iter().chain(fc.kwargs.iter().map(|(_, e)| e)));
        }
    }
    let calls = prefetch_calls(input, &exprs, rows.len(), &|i| (&rows[i], &[]), &empty)?;
    let pure = exprs.iter().all(|e| input.expr_pure(e, &calls));
    let eval_key = |ctx: &Ctx<'_>, r: &JRow<'_>| -> Result<Vec<Value>> {
        match plan.group_slot {
            // the key of the base row (rowid, then iterator positions)
            Some(slot) => Ok(match r[slot] {
                Some(b) => std::iter::once(b.rowid).chain(b.pos.iter().copied()).map(Value::Int).collect(),
                None => Vec::new(),
            }),
            None => grouping.iter().map(|e| eval(e, ctx).map_err(|err| agg_error(e, err))).collect(),
        }
    };
    // The groups of a run of rows: (key, first row, states), in order of first occurrence. Rows are aggregated in
    // order and the run stops at its first failure, which is returned with it.
    type Partial = (FxHashMap<GroupKey, usize>, Vec<(GroupKey, usize, Vec<Box<dyn AggState>>)>);
    let aggregate_rows = |range: std::ops::Range<usize>| -> (Partial, Option<Error>) {
        let mut part: Partial = (FxHashMap::default(), Vec::new());
        let mut args: SmallVec<[Value; 4]> = SmallVec::new();
        for i in range {
            let r = &rows[i];
            let ctx = Ctx { input, row: r, pre: &[], slots: &empty, calls: slot_of(&calls, i) };
            let step = (|| -> Result<()> {
                let k = group_key_of(&eval_key(&ctx, r)?);
                let gi = match part.0.get(&k) {
                    Some(gi) => *gi,
                    None => {
                        part.1.push((k.clone(), i, new_states()?));
                        part.0.insert(k, part.1.len() - 1);
                        part.1.len() - 1
                    }
                };
                for (a, state) in plan.aggs.iter().zip(part.1[gi].2.iter_mut()) {
                    let fc = a.expr.as_function_call().ok_or_else(|| Error::internal("aggregate call expected"))?;
                    args.clear();
                    for e in fc.args.iter().chain(fc.kwargs.iter().map(|(_, e)| e)) {
                        args.push(eval(e, &ctx).map_err(|err| agg_error(&a.expr, err))?);
                    }
                    let na = fc.args.len();
                    a.plan.update_slices(state.as_mut(), &args[..na], &args[na..]).map_err(|err| agg_error(&a.expr, err))?;
                }
                Ok(())
            })();
            if let Err(err) = step {
                return (part, Some(err));
            }
        }
        (part, None)
    };
    // pure arguments of aggregates whose states merge: chunks of rows in parallel, merged in row order
    let chunked = pure && plan.aggs.iter().all(|a| a.plan.func.mergeable()) && super::par::worth_parallel(rows.len());
    let parts = if chunked { super::par::map_chunks(rows.len(), aggregate_rows) } else { vec![aggregate_rows(0..rows.len())] };
    for ((_, part_groups), failed) in parts {
        for (k, first_row, mut states) in part_groups {
            match index.get(&k) {
                Some(gi) => {
                    for ((a, state), later) in plan.aggs.iter().zip(groups[*gi].states.iter_mut()).zip(states.iter_mut()) {
                        state.merge(later.as_mut()).map_err(|err| agg_error(&a.expr, err))?;
                    }
                }
                None => {
                    let r = &rows[first_row];
                    // the key's values (to order the groups), evaluated again for the group's first row
                    let key = eval_key(&Ctx { input, row: r, pre: &[], slots: &empty, calls: slot_of(&calls, first_row) }, r)?;
                    groups.push(Group { key, rep: r.clone(), states });
                    index.insert(k, groups.len() - 1);
                }
            }
        }
        if let Some(err) = failed {
            return Err(err);
        }
    }
    if groups.is_empty() && grouping.is_empty() && plan.group_slot.is_none() {
        // an aggregation without grouping returns one row, also for empty input (SQL semantics)
        groups.push(Group { key: Vec::new(), rep: smallvec![None; input.width], states: new_states()? });
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
    let mut out = Vec::with_capacity(groups.len());
    for mut g in groups {
        let mut pre = Vec::with_capacity(plan.aggs.len());
        for (a, st) in plan.aggs.iter().zip(g.states.iter_mut()) {
            pre.push(st.value().map_err(|err| agg_error(&a.expr, err))?);
        }
        out.push(OutRow { row: g.rep, pre });
    }
    Ok(out)
}

/// Sorts `rows` by `ordering` (stable; ASC puts NULLs last, DESC first; a wildcard direction sorts ascending).
fn sort_rows<'a>(input: &'a QueryInput, plan: &Plan, rows: Vec<OutRow<'a>>, ordering: &[(ExprRef, Option<bool>)]) -> Result<Vec<OutRow<'a>>> {
    sort_rows_upto(input, plan, rows, ordering, None)
}

/// `sort_rows()`; with `keep`, only the first `keep` rows of the sorted order are returned (they are selected without
/// sorting the rest).
fn sort_rows_upto<'a>(
    input: &'a QueryInput,
    plan: &Plan,
    rows: Vec<OutRow<'a>>,
    ordering: &[(ExprRef, Option<bool>)],
    keep: Option<usize>,
) -> Result<Vec<OutRow<'a>>> {
    if ordering.is_empty() {
        return Ok(rows);
    }
    let exprs: Vec<&ExprRef> = ordering.iter().map(|(e, _)| e).collect();
    let calls = prefetch_calls(input, &exprs, rows.len(), &|i| (&rows[i].row, &rows[i].pre), &plan.slots)?;
    let pure = exprs.iter().all(|e| input.expr_pure(e, &calls));
    let row_keys = |i: usize| -> Result<SmallVec<[Value; 2]>> {
        let r = &rows[i];
        let ctx = Ctx { input, row: &r.row, pre: &r.pre, slots: &plan.slots, calls: slot_of(&calls, i) };
        ordering.iter().map(|(e, _)| eval(e, &ctx).map_err(|err| super::compute::eval_error(e, err))).collect()
    };
    // by key, ties by position (a stable sort)
    let cmp_keyed = |(ka, a): &(SmallVec<[Value; 2]>, u32), (kb, b): &(SmallVec<[Value; 2]>, u32)| -> std::cmp::Ordering {
        for (i, (_, asc)) in ordering.iter().enumerate() {
            let ord = cmp_values(&ka[i], &kb[i]);
            let ord = if asc.unwrap_or(true) { ord } else { ord.reverse() };
            if ord.is_ne() {
                return ord;
            }
        }
        a.cmp(b)
    };
    if let Some(k) = keep.filter(|k| pure && *k < super::par::CHUNK_ROWS / 4 && super::par::worth_parallel(rows.len())) {
        // the first `k` rows of each chunk, then the first `k` of those
        let firsts = |range: std::ops::Range<usize>| -> Result<Vec<(SmallVec<[Value; 2]>, u32)>> {
            let mut keyed = Vec::with_capacity(range.len());
            for i in range {
                keyed.push((row_keys(i)?, i as u32));
            }
            if keyed.len() > k {
                if k > 0 {
                    keyed.select_nth_unstable_by(k - 1, &cmp_keyed);
                }
                keyed.truncate(k);
            }
            Ok(keyed)
        };
        let mut candidates = Vec::new();
        for part in super::par::map_chunks(rows.len(), firsts) {
            candidates.extend(part?);
        }
        candidates.sort_unstable_by(&cmp_keyed);
        candidates.truncate(k);
        let mut rows: Vec<Option<OutRow<'a>>> = rows.into_iter().map(Some).collect();
        return Ok(candidates.into_iter().filter_map(|(_, i)| rows[i as usize].take()).collect());
    }
    let keys = super::par::try_map_rows(rows.len(), pure, |i| {
        let r = &rows[i];
        let ctx = Ctx { input, row: &r.row, pre: &r.pre, slots: &plan.slots, calls: slot_of(&calls, i) };
        ordering
            .iter()
            .map(|(e, _)| eval(e, &ctx).map_err(|err| super::compute::eval_error(e, err)))
            .collect::<Result<SmallVec<[Value; 2]>>>()
    });
    let keys = keys.into_iter().collect::<Result<Vec<_>>>()?;
    // the order of row indices: by key, ties by position (a stable sort)
    let cmp = |a: &u32, b: &u32| -> std::cmp::Ordering {
        let (ka, kb) = (&keys[*a as usize], &keys[*b as usize]);
        for (i, (_, asc)) in ordering.iter().enumerate() {
            let ord = cmp_values(&ka[i], &kb[i]);
            let ord = if asc.unwrap_or(true) { ord } else { ord.reverse() };
            if ord.is_ne() {
                return ord;
            }
        }
        a.cmp(b)
    };
    let mut order: Vec<u32> = (0..rows.len() as u32).collect();
    match keep {
        Some(k) if k < order.len() / 2 => {
            if k > 0 {
                order.select_nth_unstable_by(k - 1, &cmp);
            }
            order.truncate(k);
            order.sort_unstable_by(&cmp);
        }
        _ => super::par::sort_by(&mut order, &cmp),
    }
    let mut rows: Vec<Option<OutRow<'a>>> = rows.into_iter().map(Some).collect();
    Ok(order.into_iter().filter_map(|i| rows[i as usize].take()).collect())
}

/// Evaluates the window function calls row by row in window order (upstream `FunctionCall.eval()` of a window
/// call): a new aggregator per partition (consecutive rows with equal `group_by` values), updated with each row and
/// read after each update.
fn eval_windows<'a>(input: &'a QueryInput, plan: &Plan, rows: &mut [OutRow<'a>]) -> Result<()> {
    let empty = HashMap::new();
    let base = plan.aggs.len();
    for r in rows.iter_mut() {
        r.pre.resize(base + plan.windows.len(), Value::Null);
    }
    for (wi, w) in plan.windows.iter().enumerate() {
        let fc = w.expr.as_function_call().ok_or_else(|| Error::internal("window call expected"))?;
        let mut state: Option<Box<dyn AggState>> = None;
        let mut partition: Option<String> = None;
        for r in rows.iter_mut() {
            let ctx = Ctx { input, row: &r.row, pre: &[], slots: &empty, calls: None };
            let part_vals = fc.group_by.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>().map_err(|e| agg_error(&w.expr, e))?;
            let pkey = keys_of(&part_vals);
            if state.is_none() || partition.as_ref() != Some(&pkey) {
                state = Some(w.plan.new_state().map_err(|e| agg_error(&w.expr, e))?);
                partition = Some(pkey);
            }
            let st = state.as_mut().expect("state");
            let (args, kwargs) = call_args(&w.expr, &ctx).map_err(|e| agg_error(&w.expr, e))?;
            if w.plan.skips(&args, &kwargs) {
                r.pre[base + wi] = Value::Null;
                continue;
            }
            w.plan.update(st.as_mut(), args, kwargs).map_err(|e| agg_error(&w.expr, e))?;
            r.pre[base + wi] = st.value().map_err(|e| agg_error(&w.expr, e))?;
        }
    }
    Ok(())
}

/// Rows before order by / limit: filtered, sampled, aggregated, with window values.
fn result_rows<'a>(input: &'a QueryInput, plan: &Plan, rowid_desc: bool) -> Result<Vec<OutRow<'a>>> {
    let mut rows = filtered(input)?;
    if rowid_desc {
        rows.reverse();
    }
    if let Some(s) = &input.q.sample {
        rows = apply_sample(input, rows, s)?;
    }
    if !plan.agg_order.is_empty() {
        // ordered aggregates see their rows in order (groups keep the input order)
        let wrapped = rows.into_iter().map(|row| OutRow { row, pre: Vec::new() }).collect();
        rows = sort_rows(input, plan, wrapped, &plan.agg_order)?.into_iter().map(|r| r.row).collect();
    }
    let mut out: Vec<OutRow<'a>> = match &plan.grouping {
        Some(g) => aggregate(input, plan, rows, g)?,
        None => rows.into_iter().map(|row| OutRow { row, pre: Vec::new() }).collect(),
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
    let row: JRow<'_> = smallvec![None; input.width];
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
    let mut rows = result_rows(input, &plan, rowid_desc)?;
    if !q.order_by.is_empty() {
        let ordering: Vec<(ExprRef, Option<bool>)> = q.order_by.iter().map(|(e, a)| (e.clone(), Some(*a))).collect();
        // only the rows up to the limit are needed
        let keep = q.limit.map(|n| q.offset.unwrap_or(0).max(0) as usize + n.max(0) as usize);
        rows = sort_rows_upto(input, &plan, rows, &ordering, keep)?;
    }
    let offset = q.offset.unwrap_or(0).max(0) as usize;
    let rows: Vec<OutRow<'_>> = match q.limit {
        Some(n) => rows.into_iter().skip(offset).take(n.max(0) as usize).collect(),
        None => rows.into_iter().skip(offset).collect(),
    };
    let exprs: Vec<&ExprRef> = input.select.iter().map(|(e, _)| e).collect();
    let calls = prefetch_calls(input, &exprs, rows.len(), &|i| (&rows[i].row, &rows[i].pre), &plan.slots)?;
    let pure = exprs.iter().all(|e| input.expr_pure(e, &calls));
    let projected = super::par::try_map_rows(rows.len(), pure, |i| {
        let r = &rows[i];
        let ctx = Ctx { input, row: &r.row, pre: &r.pre, slots: &plan.slots, calls: slot_of(&calls, i) };
        let mut vals = Vec::with_capacity(input.select.len());
        for (e, _) in &input.select {
            vals.push(eval(e, &ctx).map_err(|err| super::compute::eval_error(e, err))?);
        }
        Ok(vals)
    });
    let out_rows = projected.into_iter().collect::<Result<Vec<_>>>()?;
    Ok(ResultData { names, types, rows: out_rows })
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
            crate::bail!(UnsupportedOperation, "count() cannot be used: query plan contains a non-SQL node (FilterNode)");
        }
    }
    if q.where_.is_none() && q.sample.is_none() && q.group_by.is_none() && q.grouping_tbl.is_none() && q.joins.is_empty() {
        if let [src] = input.sources.as_slice() {
            if let [e] = src.path.elems.as_slice() {
                // the visible rows of one table: counted by the store
                return Ok(e.data.count_at(e.version));
            }
        }
    }
    let mut rows = filtered(input)?;
    if let Some(s) = &q.sample {
        rows = apply_sample(input, rows, s)?;
    }
    if let Some(t) = q.grouping_tbl {
        let slot = input.slot(&t).ok_or_else(|| Error::internal("grouping table is not in the table path"))?;
        let keys: std::collections::HashSet<(i64, &[i64])> = rows.iter().filter_map(|r| r[slot]).map(|b| b.key()).collect();
        return Ok(keys.len());
    }
    match &q.group_by {
        Some(g) => {
            let empty = HashMap::new();
            let mut keys = std::collections::HashSet::new();
            for r in &rows {
                let ctx = Ctx { input, row: r, pre: &[], slots: &empty, calls: None };
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
