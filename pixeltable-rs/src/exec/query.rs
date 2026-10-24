//! Query execution: scan -> join -> filter -> sample -> aggregate -> window functions -> sort -> limit -> project.
//!
//! `QueryInput::snapshot()` captures, under the catalog lock, the rows and metadata of every from-table (`Source`:
//! the scanned base table, the view predicates, the unstored computed columns). `execute()` and `count()` run
//! without the lock.
//!
//! Rows flow as `JRow`s (one optional stored row per from-table; None is the missing side of an outer join).
//! Aggregate and window function results are computed outside of expression evaluation and handed to it through
//! `RowContext::precomputed()`: `Plan::slots` maps each aggregate/window call node (by address) to an index into
//! the per-output-row `pre` values.
//!
//! Semantics that upstream gets from evaluating part of a query in SQL are reproduced where tests observe them:
//! grouping keys must be "SQL-expressible" (`is_sql_expr()`), `count()` rejects non-SQL filters, and division by
//! zero in a SQL-expressible arithmetic expression yields NULL (unless `RowContext::python_semantics()`).

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use super::{JoinType, QuerySpec, ResultData, SampleClause};
use crate::catalog::Catalog;
use crate::catalog::table::ColId;
use crate::catalog::views::ScanSource;
use crate::error::{Error, Result};
use crate::expr::display::display;
use crate::expr::eval::{RowContext, eval};
use crate::expr::{CmpOp, ColumnRef, Expr, ExprKind, ExprRef, LogicalOp};
use crate::functions::agg::{AggPlan, AggState};
use crate::store::rows::{StoredRow, TableData};
use crate::value::{Value, cmp_values};

/// One from-table of a query, captured under the catalog lock.
pub struct Source {
    /// tables whose columns this source provides (the from-table and its bases)
    pub chain: Vec<Uuid>,
    pub scan: ScanSource,
    /// version of the scanned table
    pub version: i64,
    pub data: Arc<TableData>,
    /// view predicates
    pub predicates: Vec<ExprRef>,
    /// media columns of the scanned table that are validated on read
    pub read_validation: Arc<crate::media::ReadValidation>,
}

/// Everything a query reads, captured under the catalog lock.
pub struct QueryInput {
    pub q: QuerySpec,
    pub sources: Vec<Source>,
    pub select: Vec<(ExprRef, String)>,
    /// value expressions of unstored computed columns, evaluated on read
    pub unstored: HashMap<(Uuid, ColId), ExprRef>,
    /// table id -> index of the source that provides its columns
    slot_of: HashMap<Uuid, usize>,
}

impl QueryInput {
    pub fn snapshot(cat: &mut Catalog, q: &QuerySpec) -> Result<QueryInput> {
        q.check_columns_exist(cat)?;
        let select = q.effective_select(cat)?;
        let mut sources = Vec::new();
        let mut unstored = HashMap::new();
        let mut slot_of = HashMap::new();
        for (i, (tbl_id, version)) in q.from_tbls().into_iter().enumerate() {
            let scan = cat.scan_source(tbl_id, version)?;
            let base = cat.table(scan.base_tbl_id)?.clone();
            let v = scan.base_version.unwrap_or(base.current_version);
            let predicates = scan.predicates.iter().map(crate::expr::serde::from_json).collect::<Result<Vec<_>>>()?;
            for c in &base.columns {
                if !c.stored {
                    if let Some(e) = &c.value_expr {
                        unstored.insert((scan.base_tbl_id, c.id), crate::expr::serde::from_json(e)?);
                    }
                }
            }
            let chain = cat.base_chain(tbl_id);
            for t in &chain {
                slot_of.entry(*t).or_insert(i);
            }
            let read_validation = Arc::new(crate::media::ReadValidation::for_table(&base)?);
            let data = cat.data(scan.base_tbl_id)?;
            sources.push(Source { chain, scan, version: v, data, predicates, read_validation });
        }
        Ok(QueryInput { q: q.clone(), sources, select, unstored, slot_of })
    }

    fn slot(&self, tbl_id: &Uuid) -> Option<usize> {
        self.slot_of.get(tbl_id).copied()
    }
}

type JRow<'a> = Vec<Option<&'a StoredRow>>;

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
}

fn node_key(e: &Expr) -> usize {
    e as *const Expr as usize
}

impl Ctx<'_> {
    /// The source slot and stored row (None: the missing side of an outer join) that provide `col`.
    fn row_of(&self, col: &ColumnRef) -> Result<(usize, Option<&StoredRow>)> {
        let Some(slot) = self.input.slot(&col.tbl_id) else {
            return Err(Error::internal(format!("column {} is not provided by the query's tables", col.name)));
        };
        Ok((slot, self.row.get(slot).copied().flatten()))
    }

    /// For a read-validated media column with a file: the validation error, if any.
    fn read_validation_error(&self, slot: usize, r: &StoredRow, col: &ColumnRef) -> Option<Option<crate::store::rows::CellError>> {
        let rv = &self.input.sources[slot].read_validation;
        let kind = rv.kind(col.col_id)?;
        match r.cell(col.col_id) {
            Value::Media(s) => Some(rv.error(s, kind)),
            _ => Some(None),
        }
    }
}

impl RowContext for Ctx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        let (slot, row) = self.row_of(col)?;
        let Some(r) = row else { return Ok(Value::Null) };
        if let Some(e) = self.input.unstored.get(&(col.tbl_id, col.col_id)) {
            return eval(e, self);
        }
        if let Some(Some(_)) = self.read_validation_error(slot, r, col) {
            // an invalid file reads as NULL; its error is the cell's error
            return Ok(Value::Null);
        }
        Ok(r.cell(col.col_id).clone())
    }

    fn column_error(&self, col: &ColumnRef) -> Result<Option<crate::store::rows::CellError>> {
        let (slot, row) = self.row_of(col)?;
        let Some(r) = row else { return Ok(None) };
        if let Some(e) = self.read_validation_error(slot, r, col) {
            return Ok(e);
        }
        Ok(r.error(col.col_id).cloned())
    }

    fn precomputed(&self, e: &Expr) -> Option<Value> {
        if self.slots.is_empty() {
            return None;
        }
        self.slots.get(&node_key(e)).map(|i| self.pre[*i].clone())
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
        ExprKind::Literal(_) => true,
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
    /// Some: aggregate (Some(empty): a single group)
    grouping: Option<Vec<ExprRef>>,
    /// the ordering the rows need before window functions are evaluated
    window_order: Vec<(ExprRef, Option<bool>)>,
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
        if q.grouping_tbl.is_some() {
            return Err(Error::not_implemented("group_by(<base table>)"));
        }
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
        if !aggs.is_empty() {
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
        for a in &aggs {
            let fc = a.expr.as_function_call().ok_or_else(|| Error::internal("aggregate call expected"))?;
            if !fc.order_by.is_empty() {
                let mut o: Vec<(ExprRef, Option<bool>)> =
                    grouping.iter().flatten().map(|e| (e.clone(), None)).collect();
                o.extend(fc.order_by.iter().map(|e| (e.clone(), Some(true))));
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
        Ok(Plan { aggs, windows, slots, grouping, window_order })
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

fn keys_of(vals: &[Value]) -> String {
    let mut s = String::new();
    for v in vals {
        s.push_str(&group_key(v));
        s.push('\u{1}');
    }
    s
}

/// The visible rows of source `i` (version + view predicates), in rowid order.
fn source_rows<'a>(input: &'a QueryInput, i: usize) -> Result<Vec<&'a StoredRow>> {
    let src = &input.sources[i];
    let mut rows: Vec<&StoredRow> = src.data.visible_at(src.version).collect();
    rows.sort_by_key(|r| r.rowid);
    if src.predicates.is_empty() {
        return Ok(rows);
    }
    let n = input.sources.len();
    let empty = HashMap::new();
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let mut jrow: JRow<'_> = vec![None; n];
        jrow[i] = Some(r);
        let ctx = Ctx { input, row: &jrow, pre: &[], slots: &empty };
        let mut keep = true;
        for p in &src.predicates {
            if !matches!(eval(p, &ctx)?, Value::Bool(true)) {
                keep = false;
                break;
            }
        }
        if keep {
            out.push(r);
        }
    }
    Ok(out)
}

/// For a join predicate, the equality conjuncts `lhs == rhs` whose sides reference only the left tables resp. only
/// the right table (hash join keys): (left exprs, right exprs).
fn equi_keys(pred: &ExprRef, input: &QueryInput, right_slot: usize) -> (Vec<ExprRef>, Vec<ExprRef>) {
    let side = |e: &ExprRef| -> Option<bool> {
        let refs = e.column_refs();
        if refs.is_empty() {
            return None;
        }
        let slots: Vec<usize> = refs.iter().filter_map(|c| input.slot(&c.tbl_id)).collect();
        if slots.len() != refs.len() {
            return None;
        }
        if slots.iter().all(|s| *s == right_slot) {
            Some(true)
        } else if slots.iter().all(|s| *s < right_slot) {
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
    let n = input.sources.len();
    let empty = HashMap::new();
    let mut rows: Vec<JRow<'a>> = source_rows(input, 0)?
        .into_iter()
        .map(|r| {
            let mut jr = vec![None; n];
            jr[0] = Some(r);
            jr
        })
        .collect();
    for (k, join) in input.q.joins.iter().enumerate() {
        let slot = k + 1;
        let right = source_rows(input, slot)?;
        let mut out: Vec<JRow<'a>> = Vec::new();
        let mut right_matched = vec![false; right.len()];
        let combined = |left: &JRow<'a>, r: &'a StoredRow| -> JRow<'a> {
            let mut jr = left.clone();
            jr[slot] = Some(r);
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
                // candidate right rows per left row: by hash on the equality keys, else all
                let mut index: HashMap<String, Vec<usize>> = HashMap::new();
                if !lkeys.is_empty() {
                    for (j, r) in right.iter().enumerate() {
                        let mut jr = vec![None; n];
                        jr[slot] = Some(*r);
                        let ctx = Ctx { input, row: &jr, pre: &[], slots: &empty };
                        let vals = rkeys.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
                        if vals.iter().any(Value::is_null) {
                            continue;
                        }
                        index.entry(keys_of(&vals)).or_default().push(j);
                    }
                }
                let all: Vec<usize> = (0..right.len()).collect();
                for l in &rows {
                    let candidates: &[usize] = if lkeys.is_empty() {
                        &all
                    } else {
                        let ctx = Ctx { input, row: l, pre: &[], slots: &empty };
                        let vals = lkeys.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
                        if vals.iter().any(Value::is_null) {
                            &[]
                        } else {
                            index.get(&keys_of(&vals)).map(Vec::as_slice).unwrap_or(&[])
                        }
                    };
                    let mut matched = false;
                    for &j in candidates {
                        let jr = combined(l, right[j]);
                        let ctx = Ctx { input, row: &jr, pre: &[], slots: &empty };
                        if matches!(eval(pred, &ctx)?, Value::Bool(true)) {
                            matched = true;
                            right_matched[j] = true;
                            out.push(jr);
                        }
                    }
                    if !matched && matches!(how, JoinType::Left | JoinType::FullOuter) {
                        out.push(l.clone());
                    }
                }
                if how == JoinType::FullOuter {
                    for (j, r) in right.iter().enumerate() {
                        if !right_matched[j] {
                            let mut jr = vec![None; n];
                            jr[slot] = Some(*r);
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
fn filtered<'a>(input: &'a QueryInput) -> Result<Vec<JRow<'a>>> {
    let rows = scan_and_join(input)?;
    let Some(w) = &input.q.where_ else { return Ok(rows) };
    let empty = HashMap::new();
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let ctx = Ctx { input, row: &r, pre: &[], slots: &empty };
        match eval(w, &ctx) {
            Ok(Value::Bool(true)) => out.push(r),
            Ok(_) => {}
            Err(err) => return Err(super::compute::eval_error(w, err)),
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------ sampling

fn row_hash(seed: u64, row: &JRow<'_>) -> u64 {
    let rowid = row.first().copied().flatten().map_or(0, |r| r.rowid);
    super::sample::splitmix64(seed ^ super::sample::splitmix64(rowid as u64))
}

/// Upstream `SqlSampleNode`: rows are ranked by a hash of (seed, rowid); a fraction keeps the rows whose hash falls
/// below the fraction of the hash space; `n` takes the lowest-ranked rows; stratified samples rank within strata.
fn apply_sample<'a>(input: &'a QueryInput, rows: Vec<JRow<'a>>, s: &SampleClause) -> Result<Vec<JRow<'a>>> {
    let seed = match s.seed {
        Some(sd) => sd as u64,
        None => super::sample::random_u64(),
    };
    let mut keyed: Vec<(u64, JRow<'a>)> = rows.into_iter().map(|r| (row_hash(seed, &r), r)).collect();
    keyed.sort_by_key(|(k, _)| *k);
    let threshold = |f: f64| -> u64 { if f >= 1.0 { u64::MAX } else { (f * u64::MAX as f64) as u64 } };
    if !s.is_stratified() {
        return Ok(match (s.fraction, s.n) {
            (Some(f), _) => {
                let t = threshold(f);
                keyed.into_iter().filter(|(k, _)| *k < t || (f >= 1.0)).map(|(_, r)| r).collect()
            }
            (None, Some(n)) => keyed.into_iter().take(n.max(0) as usize).map(|(_, r)| r).collect(),
            _ => keyed.into_iter().map(|(_, r)| r).collect(),
        });
    }
    // stratified: rank within strata (in hash order)
    let empty = HashMap::new();
    let mut strata: HashMap<String, usize> = HashMap::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut ranked: Vec<(usize, usize, u64, JRow<'a>)> = Vec::with_capacity(keyed.len());
    for (k, r) in keyed {
        let ctx = Ctx { input, row: &r, pre: &[], slots: &empty };
        let vals = s.stratify_by.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
        let skey = keys_of(&vals);
        let next = strata.len();
        let sid = *strata.entry(skey).or_insert(next);
        if sid == counts.len() {
            counts.push(0);
        }
        let rank = counts[sid];
        counts[sid] += 1;
        ranked.push((sid, rank, k, r));
    }
    let out: Vec<(usize, u64, JRow<'a>)> = match (s.n_per_stratum, s.n, s.fraction) {
        (Some(per), _, _) => ranked.into_iter().filter(|(_, rank, _, _)| (*rank as i64) < per).map(|(_, rk, k, r)| (rk, k, r)).collect(),
        (None, Some(n), _) => {
            let mut v: Vec<(usize, u64, JRow<'a>)> = ranked.into_iter().map(|(_, rk, k, r)| (rk, k, r)).collect();
            v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            v.truncate(n.max(0) as usize);
            v
        }
        (None, None, Some(f)) => ranked
            .into_iter()
            .filter(|(sid, rank, _, _)| (*rank as f64) < (f * counts[*sid] as f64).ceil())
            .map(|(_, rk, k, r)| (rk, k, r))
            .collect(),
        _ => Vec::new(),
    };
    let mut out = out;
    out.sort_by_key(|(_, k, _)| *k);
    Ok(out.into_iter().map(|(_, _, r)| r).collect())
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
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut groups: Vec<Group<'a>> = Vec::new();
    let new_states = || -> Result<Vec<Box<dyn AggState>>> {
        plan.aggs.iter().map(|a| a.plan.new_state().map_err(|e| agg_error(&a.expr, e))).collect()
    };
    for r in rows {
        let ctx = Ctx { input, row: &r, pre: &[], slots: &empty };
        let key = grouping.iter().map(|e| eval(e, &ctx).map_err(|err| agg_error(e, err))).collect::<Result<Vec<_>>>()?;
        let k = keys_of(&key);
        let gi = match index.get(&k) {
            Some(i) => *i,
            None => {
                groups.push(Group { key, rep: r.clone(), states: new_states()? });
                index.insert(k, groups.len() - 1);
                groups.len() - 1
            }
        };
        for (a, state) in plan.aggs.iter().zip(groups[gi].states.iter_mut()) {
            let (args, kwargs) = call_args(&a.expr, &ctx).map_err(|err| agg_error(&a.expr, err))?;
            a.plan.update(state.as_mut(), args, kwargs).map_err(|err| agg_error(&a.expr, err))?;
        }
    }
    if groups.is_empty() && grouping.is_empty() {
        // an aggregation without grouping returns one row, also for empty input (SQL semantics)
        groups.push(Group { key: Vec::new(), rep: vec![None; input.sources.len()], states: new_states()? });
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
    if ordering.is_empty() {
        return Ok(rows);
    }
    let mut keyed: Vec<(Vec<Value>, OutRow<'a>)> = Vec::with_capacity(rows.len());
    for r in rows {
        let ctx = Ctx { input, row: &r.row, pre: &r.pre, slots: &plan.slots };
        let keys = ordering.iter().map(|(e, _)| eval(e, &ctx).map_err(|err| super::compute::eval_error(e, err))).collect::<Result<Vec<_>>>()?;
        keyed.push((keys, r));
    }
    keyed.sort_by(|(a, _), (b, _)| {
        for (i, (_, asc)) in ordering.iter().enumerate() {
            let ord = cmp_values(&a[i], &b[i]);
            let ord = if asc.unwrap_or(true) { ord } else { ord.reverse() };
            if ord.is_ne() {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });
    Ok(keyed.into_iter().map(|(_, r)| r).collect())
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
            let ctx = Ctx { input, row: &r.row, pre: &[], slots: &empty };
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

/// Runs the query. `rowid_desc`: for `tail()`, scan in descending rowid order.
pub fn execute(input: &QueryInput, rowid_desc: bool) -> Result<ResultData> {
    let q = &input.q;
    let plan = Plan::new(input)?;
    let names: Vec<String> = input.select.iter().map(|(_, n)| n.clone()).collect();
    let types = input.select.iter().map(|(e, _)| e.col_type.clone()).collect();
    if q.limit == Some(0) {
        return Ok(ResultData { names, types, rows: Vec::new() });
    }
    let mut rows = result_rows(input, &plan, rowid_desc)?;
    if !q.order_by.is_empty() {
        let ordering: Vec<(ExprRef, Option<bool>)> = q.order_by.iter().map(|(e, a)| (e.clone(), Some(*a))).collect();
        rows = sort_rows(input, &plan, rows, &ordering)?;
    }
    let offset = q.offset.unwrap_or(0).max(0) as usize;
    let rows: Vec<OutRow<'_>> = match q.limit {
        Some(n) => rows.into_iter().skip(offset).take(n.max(0) as usize).collect(),
        None => rows.into_iter().skip(offset).collect(),
    };
    let mut out_rows = Vec::with_capacity(rows.len());
    for r in &rows {
        let ctx = Ctx { input, row: &r.row, pre: &r.pre, slots: &plan.slots };
        let mut vals = Vec::with_capacity(input.select.len());
        for (e, _) in &input.select {
            vals.push(eval(e, &ctx).map_err(|err| super::compute::eval_error(e, err))?);
        }
        out_rows.push(vals);
    }
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
    let mut rows = filtered(input)?;
    if let Some(s) = &q.sample {
        rows = apply_sample(input, rows, s)?;
    }
    match &q.group_by {
        Some(g) => {
            let empty = HashMap::new();
            let mut keys = std::collections::HashSet::new();
            for r in &rows {
                let ctx = Ctx { input, row: r, pre: &[], slots: &empty };
                let vals = g.iter().map(|e| eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
                keys.insert(keys_of(&vals));
            }
            Ok(keys.len())
        }
        None => Ok(rows.len()),
    }
}

// ------------------------------------------------------------------------------------- windows in computed columns

/// Evaluates against the cells of one stored row, with precomputed window values.
struct CellsCtx<'a> {
    row: &'a StoredRow,
    pre: &'a [Value],
    slots: &'a HashMap<usize, usize>,
}

impl RowContext for CellsCtx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        Ok(self.row.cell(col.col_id).clone())
    }
    fn precomputed(&self, e: &Expr) -> Option<Value> {
        self.slots.get(&node_key(e)).map(|i| self.pre[*i].clone())
    }
}

/// True if `e` contains a window function call.
pub fn has_window_calls(e: &ExprRef) -> bool {
    let mut out = Vec::new();
    collect_calls(e, &mut out, &|fc| fc.is_window());
    !out.is_empty()
}

/// Evaluates a computed column's value expression that contains window function calls over `rows` (upstream
/// computes such columns with a query plan that orders the rows by the window clauses). Returns one result per
/// row, in the order of `rows`.
pub fn eval_rows_with_windows(rows: &[&StoredRow], expr: &ExprRef) -> Result<Vec<Result<Value>>> {
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
        let ctx = CellsCtx { row: r, pre: &[], slots: &no_slots };
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
            let ctx = CellsCtx { row: rows[*i], pre: &[], slots: &no_slots };
            let pkey = keys_of(&fc.group_by.iter().map(|g| eval(g, &ctx)).collect::<Result<Vec<_>>>()?);
            if state.is_none() || partition.as_ref() != Some(&pkey) {
                state = Some(plan.new_state()?);
                partition = Some(pkey);
            }
            let (args, kwargs) = call_args(e, &ctx)?;
            if plan.skips(&args, &kwargs) {
                continue;
            }
            let st = state.as_mut().expect("state");
            plan.update(st.as_mut(), args, kwargs)?;
            pre[*i][wi] = st.value()?;
        }
    }
    Ok(rows
        .iter()
        .enumerate()
        .map(|(i, r)| eval(expr, &CellsCtx { row: r, pre: &pre[i], slots: &slots }))
        .collect())
}
