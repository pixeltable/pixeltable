//! Expressions compiled against a row layout, for evaluating many rows quickly.
//!
//! `expr::eval()` walks the expression tree through a `RowContext` trait object and returns owned values, so every
//! column read is a lookup plus a copy of the cell (an allocation for a string). A `Fast` expression resolves its
//! column reads to (slot, column) pairs once and evaluates the common node kinds (column references, literals,
//! comparisons, `&`/`|`/`~`, `is None`, `isin()` with literals, arithmetic) on borrowed values; every other node is
//! evaluated by `expr::eval()` against the row's context, so results, errors and NULL semantics are exactly
//! `expr::eval()`'s. Only for evaluation without precomputed values (aggregates are not evaluated per row).

use std::borrow::Cow;

use crate::catalog::table::ColId;
use crate::error::Result;
use crate::expr::eval::{RowContext, arithmetic, cmp_operand, compare_tri, eval, locate};
use crate::expr::{ArithOp, CmpOp, ColumnRef, ExprKind, ExprRef, LogicalOp};
use crate::store::rows::StoredRow;
use crate::store::table_data::{ColChunk, RowAddr, TableData, TypedCol};
use crate::value::{Json, Value, values_equal};

/// A compiled expression (see the module documentation).
pub enum Fast {
    /// the cell of a stored column of the row in a slot (NULL if the slot is empty: the missing side of an outer join)
    Cell { slot: usize, col: ColId },
    Lit(Value),
    Cmp { op: CmpOp, lhs: Box<Fast>, rhs: Box<Fast> },
    And(Vec<Fast>),
    Or(Vec<Fast>),
    Not(Box<Fast>),
    IsNull(Box<Fast>),
    InList { lhs: Box<Fast>, values: Vec<Value> },
    /// (`node`: the expression, which a failure names, as `expr::eval()` locates it)
    Arith { node: ExprRef, lhs: Box<Fast>, rhs: Box<Fast>, lhs_json: bool, rhs_json: bool, sql: bool },
    /// a similarity lookup whose query embedding is computed: the stored embedding is the cell of the index's value
    /// column in a slot
    Similarity { node: ExprRef, slot: usize, col: ColId, query: crate::expr::eval::SimilarityQuery },
    /// evaluated by `expr::eval()`
    Other(ExprRef),
}

impl Fast {
    /// Compiles `e`; `slot_of(col)` gives the slot whose row's cell holds the column's value (None: the column is read
    /// through the row context, e.g. an unstored computed column).
    pub fn compile(e: &ExprRef, slot_of: &dyn Fn(&ColumnRef) -> Option<usize>) -> Fast {
        let sub = |x: &ExprRef| Box::new(Fast::compile(x, slot_of));
        match &e.kind {
            ExprKind::ColumnRef(c) => match slot_of(c) {
                Some(slot) => Fast::Cell { slot, col: c.col_id },
                None => Fast::Other(e.clone()),
            },
            ExprKind::Literal(v) => Fast::Lit(v.clone()),
            ExprKind::Comparison { op, lhs, rhs } => Fast::Cmp { op: *op, lhs: sub(lhs), rhs: sub(rhs) },
            ExprKind::Compound { op: LogicalOp::And, operands } => {
                Fast::And(operands.iter().map(|o| Fast::compile(o, slot_of)).collect())
            }
            ExprKind::Compound { op: LogicalOp::Or, operands } => {
                Fast::Or(operands.iter().map(|o| Fast::compile(o, slot_of)).collect())
            }
            ExprKind::Compound { op: LogicalOp::Not, operands } if operands.len() == 1 => Fast::Not(sub(&operands[0])),
            ExprKind::IsNull(x) => Fast::IsNull(sub(x)),
            ExprKind::InList { lhs, values, .. } => Fast::InList { lhs: sub(lhs), values: values.clone() },
            ExprKind::Similarity(sim) => {
                let stored = ColumnRef { tbl_id: sim.idx_tbl_id, col_id: sim.val_col_id, name: String::new(), tbl_version: None };
                match (slot_of(&stored), sim.cache.get()) {
                    (Some(slot), Some(query)) => Fast::Similarity {
                        node: e.clone(),
                        slot,
                        col: sim.val_col_id,
                        query: crate::expr::eval::SimilarityQuery::new(&sim.metric, query.clone()),
                    },
                    _ => Fast::Other(e.clone()),
                }
            }
            ExprKind::Arithmetic { lhs, rhs, .. } => Fast::Arith {
                node: e.clone(),
                lhs: sub(lhs),
                rhs: sub(rhs),
                lhs_json: lhs.col_type.is_json(),
                rhs_json: rhs.col_type.is_json(),
                sql: super::query::is_sql_expr(e, &Default::default()),
            },
            _ => Fast::Other(e.clone()),
        }
    }

    /// True if evaluation never falls back to `expr::eval()`.
    pub fn is_native(&self) -> bool {
        match self {
            Fast::Cell { .. } | Fast::Lit(_) | Fast::Similarity { .. } => true,
            Fast::Cmp { lhs, rhs, .. } | Fast::Arith { lhs, rhs, .. } => lhs.is_native() && rhs.is_native(),
            Fast::And(xs) | Fast::Or(xs) => xs.iter().all(Fast::is_native),
            Fast::Not(x) | Fast::IsNull(x) | Fast::InList { lhs: x, .. } => x.is_native(),
            Fast::Other(_) => false,
        }
    }

    /// The value for `row` (`ctx`: the same row's context, for `Other` nodes).
    pub fn eval<'v>(&'v self, row: &[Option<&'v StoredRow>], ctx: &dyn RowContext) -> Result<Cow<'v, Value>> {
        static NULL: Value = Value::Null;
        Ok(match self {
            Fast::Cell { slot, col } => Cow::Borrowed(row[*slot].map_or(&NULL, |r| r.cell(*col))),
            Fast::Lit(v) => Cow::Borrowed(v),
            Fast::Other(e) => Cow::Owned(eval(e, ctx)?),
            Fast::Similarity { node, slot, col, query } => {
                if ctx.python_semantics() {
                    let err = crate::pxt_err!(UnsupportedOperation, "similarity(): cannot be used in a computed column");
                    return Err(locate(node, err, ctx));
                }
                let stored = row[*slot].map_or(&NULL, |r| r.cell(*col));
                Cow::Owned(query.score(stored))
            }
            Fast::Arith { node, lhs, rhs, lhs_json, rhs_json, sql } => {
                let ExprKind::Arithmetic { op, .. } = &node.kind else { unreachable!("an arithmetic node") };
                let a = lhs.eval(row, ctx)?;
                let b = rhs.eval(row, ctx)?;
                // as `expr::eval()`: SQL semantics differ from Python's only for a zero divisor and a `%` of operands
                // with different signs
                let num = |v: &Value| match v {
                    Value::Int(i) => Some(*i as f64),
                    Value::Float(f) => Some(*f),
                    other => other.json_scalar().as_f64(),
                };
                let sensitive = match (num(&a), num(&b)) {
                    (_, Some(y)) if y == 0.0 => true,
                    (Some(x), Some(y)) => *op == ArithOp::Mod && (x < 0.0) != (y < 0.0),
                    _ => false,
                };
                let sql = sensitive && !ctx.python_semantics() && *sql;
                match arithmetic(*op, &a, &b, *lhs_json, *rhs_json, &node.col_type, sql) {
                    Ok(v) => Cow::Owned(v),
                    Err(err) => return Err(locate(node, err, ctx)),
                }
            }
            _ => Cow::Owned(match self.eval_bool(row, ctx)? {
                Some(b) => Value::Bool(b),
                None => Value::Null,
            }),
        })
    }

    /// The value for `row` as a three-valued boolean: Some for `Value::Bool`, None for anything else (NULL), as
    /// `expr::eval()`'s `&`/`|`/`~` and filters treat values.
    pub fn eval_bool(&self, row: &[Option<&StoredRow>], ctx: &dyn RowContext) -> Result<Option<bool>> {
        Ok(match self {
            Fast::Cmp { op, lhs, rhs } => {
                let a = lhs.eval(row, ctx)?;
                let b = rhs.eval(row, ctx)?;
                let a = if matches!(*a, Value::Json(_)) { Cow::Owned(cmp_operand(a.into_owned())) } else { a };
                let b = if matches!(*b, Value::Json(_)) { Cow::Owned(cmp_operand(b.into_owned())) } else { b };
                compare_tri(*op, &a, &b)
            }
            Fast::And(xs) => {
                let mut saw_null = false;
                for x in xs {
                    match x.eval_bool(row, ctx)? {
                        Some(false) => return Ok(Some(false)),
                        Some(true) => {}
                        None => saw_null = true,
                    }
                }
                if saw_null { None } else { Some(true) }
            }
            Fast::Or(xs) => {
                let mut saw_null = false;
                for x in xs {
                    match x.eval_bool(row, ctx)? {
                        Some(true) => return Ok(Some(true)),
                        Some(false) => {}
                        None => saw_null = true,
                    }
                }
                if saw_null { None } else { Some(false) }
            }
            Fast::Not(x) => x.eval_bool(row, ctx)?.map(|b| !b),
            Fast::IsNull(x) => Some(matches!(*x.eval(row, ctx)?, Value::Null | Value::Json(Json::Null))),
            Fast::InList { lhs, values } => {
                let v = lhs.eval(row, ctx)?;
                if v.is_null() {
                    return Ok(None);
                }
                Some(values.iter().any(|x| values_equal(&v, x)))
            }
            other => match *other.eval(row, ctx)? {
                Value::Bool(b) => Some(b),
                _ => None,
            },
        })
    }
}

/// A filter over the cells of a base table's rows that is evaluated over a run of rows of one storage chunk at once,
/// from the chunk's typed columns (`store::segment::ColChunk`): comparisons of a numeric or timestamp column with a
/// literal, `is None`, `&`, `|` and `~`. None of these can fail, so evaluating every operand for every row gives the
/// results of `Fast::eval_bool()`, which short-circuits.
pub enum VPred {
    Cmp { col: ColId, op: CmpOp, lit: Value },
    IsNull(ColId),
    And(Vec<VPred>),
    Or(Vec<VPred>),
    Not(Box<VPred>),
}

/// Three-valued results of `VPred::eval()`, one per row.
pub const V_FALSE: u8 = 0;
pub const V_TRUE: u8 = 1;
pub const V_NULL: u8 = 2;

impl VPred {
    /// The vectorized form of `f`, if it has one (columns read from slot 0 only).
    pub fn compile(f: &Fast) -> Option<VPred> {
        Some(match f {
            Fast::Cmp { op, lhs, rhs } => match (lhs.as_ref(), rhs.as_ref()) {
                (Fast::Cell { slot: 0, col }, Fast::Lit(v)) if is_vector_lit(v) => VPred::Cmp { col: *col, op: *op, lit: v.clone() },
                (Fast::Lit(v), Fast::Cell { slot: 0, col }) if is_vector_lit(v) => {
                    VPred::Cmp { col: *col, op: flipped(*op), lit: v.clone() }
                }
                _ => return None,
            },
            Fast::IsNull(x) => match x.as_ref() {
                Fast::Cell { slot: 0, col } => VPred::IsNull(*col),
                _ => return None,
            },
            Fast::And(xs) => VPred::And(xs.iter().map(VPred::compile).collect::<Option<_>>()?),
            Fast::Or(xs) => VPred::Or(xs.iter().map(VPred::compile).collect::<Option<_>>()?),
            Fast::Not(x) => VPred::Not(Box::new(VPred::compile(x)?)),
            _ => return None,
        })
    }

    /// The columns the filter reads.
    pub fn columns(&self, out: &mut Vec<ColId>) {
        match self {
            VPred::Cmp { col, .. } | VPred::IsNull(col) => {
                if !out.contains(col) {
                    out.push(*col);
                }
            }
            VPred::And(xs) | VPred::Or(xs) => xs.iter().for_each(|x| x.columns(out)),
            VPred::Not(x) => x.columns(out),
        }
    }

    /// The results for rows `range` of a chunk whose columns are `cols` (as from `columns()`), into `out` (one per row);
    /// false if a column is not in a typed form this filter can compare (the caller evaluates the rows one by one).
    pub fn eval(&self, cols: &[(ColId, std::sync::Arc<ColChunk>)], range: std::ops::Range<usize>, out: &mut [u8]) -> bool {
        let col = |id: &ColId| cols.iter().find(|(c, _)| c == id).map(|(_, c)| c.as_ref());
        match self {
            VPred::Cmp { col: id, op, lit } => match col(id) {
                Some(c) => cmp_column(c, *op, lit, range, out),
                None => false,
            },
            VPred::IsNull(id) => {
                let nulls = match col(id) {
                    Some(ColChunk::Int(c)) | Some(ColChunk::Timestamp(c)) => &c.nulls,
                    Some(ColChunk::Float(c)) => &c.nulls,
                    Some(ColChunk::Str(c)) => &c.nulls,
                    _ => return false,
                };
                match nulls {
                    Some(n) => out.iter_mut().zip(&n[range]).for_each(|(o, n)| *o = if *n { V_TRUE } else { V_FALSE }),
                    None => out.fill(V_FALSE),
                }
                true
            }
            VPred::Not(x) => {
                if !x.eval(cols, range, out) {
                    return false;
                }
                out.iter_mut().for_each(|o| {
                    *o = match *o {
                        V_FALSE => V_TRUE,
                        V_TRUE => V_FALSE,
                        n => n,
                    }
                });
                true
            }
            VPred::And(xs) | VPred::Or(xs) => {
                let and = matches!(self, VPred::And(_));
                // an operand that decides the result (false for `&`, true for `|`) wins over NULL
                let (decides, neutral) = if and { (V_FALSE, V_TRUE) } else { (V_TRUE, V_FALSE) };
                out.fill(neutral);
                let mut tmp = vec![0u8; out.len()];
                for x in xs {
                    if !x.eval(cols, range.clone(), &mut tmp) {
                        return false;
                    }
                    for (o, t) in out.iter_mut().zip(&tmp) {
                        if *o == decides || *t == decides {
                            *o = decides;
                        } else if *t == V_NULL {
                            *o = V_NULL;
                        }
                    }
                }
                true
            }
        }
    }
}

/// Splits the rows at `addrs[range]` into runs of consecutive rows of one storage chunk and evaluates `vpred` (which
/// reads `cols`) over each: (start, end, results) per run, positions relative to `addrs`; the results are empty for a
/// run whose columns `vpred` cannot read in typed form.
pub fn filter_runs(
    data: &TableData,
    addrs: &[RowAddr],
    vpred: &VPred,
    cols: &[ColId],
    range: std::ops::Range<usize>,
) -> Vec<(usize, usize, Vec<u8>)> {
    chunk_runs(addrs, range)
        .into_iter()
        .map(|(i, j)| {
            let typed: Vec<(ColId, std::sync::Arc<ColChunk>)> = cols.iter().map(|c| (*c, data.column_at(addrs[i], *c).0)).collect();
            let off = addrs[i].1 as usize % crate::store::segment::CHUNK_ROWS;
            let mut res = vec![0u8; j - i];
            if !vpred.eval(&typed, off..off + (j - i), &mut res) {
                res.clear();
            }
            (i, j, res)
        })
        .collect()
}

/// The runs of consecutive rows of one storage chunk among the rows at `addrs[range]`: (start, end) positions in
/// `addrs`.
pub fn chunk_runs(addrs: &[RowAddr], range: std::ops::Range<usize>) -> Vec<(usize, usize)> {
    use crate::store::segment::CHUNK_ROWS;
    let mut out = Vec::new();
    let mut i = range.start;
    while i < range.end {
        let (p, idx) = addrs[i];
        let chunk_end = (idx as usize / CHUNK_ROWS + 1) * CHUNK_ROWS;
        let mut j = i + 1;
        while j < range.end && addrs[j] == (p, idx + (j - i) as u32) && (idx as usize + (j - i)) < chunk_end {
            j += 1;
        }
        out.push((i, j));
        i = j;
    }
    out
}

fn is_vector_lit(v: &Value) -> bool {
    matches!(v, Value::Int(_) | Value::Float(_) | Value::Timestamp(_) | Value::Date(_) | Value::String(_))
}

/// `lit op col` as `col op' lit`.
fn flipped(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
        op => op,
    }
}

/// `compare_tri(op, cell, lit)` for the cells of rows `range` of a typed column: an ordering of the cell and the literal
/// as `cmp_values()` orders them (equality is that ordering's, as `values_equal()` has it for these kinds), NULL for a
/// NULL cell. False if the column's kind and the literal's are not handled here.
fn cmp_column(c: &ColChunk, op: CmpOp, lit: &Value, range: std::ops::Range<usize>, out: &mut [u8]) -> bool {
    use std::cmp::Ordering;
    let decide = |ord: Ordering| -> u8 {
        let r = match op {
            CmpOp::Eq => ord == Ordering::Equal,
            CmpOp::Ne => ord != Ordering::Equal,
            CmpOp::Lt => ord == Ordering::Less,
            CmpOp::Le => ord != Ordering::Greater,
            CmpOp::Gt => ord == Ordering::Greater,
            CmpOp::Ge => ord != Ordering::Less,
        };
        r as u8
    };
    fn apply<T: Copy>(col: &TypedCol<T>, range: std::ops::Range<usize>, out: &mut [u8], f: impl Fn(T) -> u8) {
        let vals = &col.vals[range.clone()];
        match &col.nulls {
            None => out.iter_mut().zip(vals).for_each(|(o, v)| *o = f(*v)),
            Some(n) => out.iter_mut().zip(vals).zip(&n[range]).for_each(|((o, v), n)| *o = if *n { V_NULL } else { f(*v) }),
        }
    }
    use crate::value::cmp_f64;
    match (c, lit) {
        (ColChunk::Int(col), Value::Int(y)) => apply(col, range, out, |x: i64| decide(x.cmp(y))),
        (ColChunk::Int(col), Value::Float(y)) => apply(col, range, out, |x: i64| decide(cmp_f64(x as f64, *y))),
        (ColChunk::Float(col), Value::Float(y)) => apply(col, range, out, |x: f64| decide(cmp_f64(x, *y))),
        (ColChunk::Float(col), Value::Int(y)) => apply(col, range, out, |x: f64| decide(cmp_f64(x, *y as f64))),
        (ColChunk::Timestamp(col), Value::Timestamp(y)) => apply(col, range, out, |x: i64| decide(x.cmp(y))),
        (ColChunk::Timestamp(col), Value::Date(d)) => {
            let y = *d as i64 * 86_400_000_000;
            apply(col, range, out, |x: i64| decide(x.cmp(&y)))
        }
        // strings order by their bytes (`String`'s order, as `cmp_values()` has it)
        (ColChunk::Str(col), Value::String(y)) => {
            for (o, i) in out.iter_mut().zip(range) {
                *o = if col.is_null(i) { V_NULL } else { decide(col.get(i).as_bytes().cmp(y.as_bytes())) };
            }
        }
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    use crate::types::ColumnType;

    struct Cells<'a>(&'a StoredRow);

    impl RowContext for Cells<'_> {
        fn column_value(&self, col: &ColumnRef) -> Result<Value> {
            Ok(self.0.cell(col.col_id).clone())
        }
    }

    fn col(id: ColId, t: ColumnType) -> ExprRef {
        Expr::new(ExprKind::ColumnRef(ColumnRef { tbl_id: uuid::Uuid::nil(), col_id: id, name: format!("c{id}"), tbl_version: None }), t)
    }

    fn node(kind: ExprKind, t: ColumnType) -> ExprRef {
        Expr::new(kind, t)
    }

    fn lit(v: Value, t: ColumnType) -> ExprRef {
        node(ExprKind::Literal(v), t)
    }

    #[test]
    fn same_as_eval() {
        let int = ColumnType::int(true);
        let float = ColumnType::float(true);
        let boolean = ColumnType::bool(true);
        let rows: Vec<StoredRow> = [
            [Value::Int(3), Value::Float(0.5), Value::Null],
            [Value::Null, Value::Float(f64::NAN), Value::Int(0)],
            [Value::Int(-7), Value::Float(2.0), Value::Int(2)],
            [Value::Int(i64::MAX), Value::Float(-1.0), Value::Int(-2)],
        ]
        .into_iter()
        .map(|cells| StoredRow { rowid: 0, cells: cells.to_vec(), errors: Vec::new(), pos: Vec::new() })
        .collect();
        let (a, b, c) = (col(0, int.clone()), col(1, float.clone()), col(2, int.clone()));
        let cmp = |op, l: &ExprRef, r: &ExprRef| node(ExprKind::Comparison { op, lhs: l.clone(), rhs: r.clone() }, boolean.clone());
        let arith = |op, l: &ExprRef, r: &ExprRef, t: &ColumnType| node(ExprKind::Arithmetic { op, lhs: l.clone(), rhs: r.clone() }, t.clone());
        let and = |xs: Vec<ExprRef>| node(ExprKind::Compound { op: LogicalOp::And, operands: xs }, boolean.clone());
        let or = |xs: Vec<ExprRef>| node(ExprKind::Compound { op: LogicalOp::Or, operands: xs }, boolean.clone());
        let not = |x: ExprRef| node(ExprKind::Compound { op: LogicalOp::Not, operands: vec![x] }, boolean.clone());
        let exprs = vec![
            cmp(CmpOp::Gt, &a, &lit(Value::Int(0), int.clone())),
            cmp(CmpOp::Le, &b, &lit(Value::Float(0.5), float.clone())),
            cmp(CmpOp::Eq, &a, &c),
            cmp(CmpOp::Ne, &b, &b),
            and(vec![cmp(CmpOp::Gt, &a, &lit(Value::Int(0), int.clone())), cmp(CmpOp::Lt, &b, &lit(Value::Float(1.0), float.clone()))]),
            or(vec![cmp(CmpOp::Gt, &a, &lit(Value::Int(0), int.clone())), cmp(CmpOp::Lt, &b, &lit(Value::Float(1.0), float.clone()))]),
            not(cmp(CmpOp::Ge, &c, &lit(Value::Int(1), int.clone()))),
            node(ExprKind::IsNull(a.clone()), boolean.clone()),
            node(ExprKind::InList { lhs: c.clone(), values: vec![Value::Int(2), Value::Int(0)], display: String::new() }, boolean.clone()),
            arith(ArithOp::Add, &a, &c, &int),
            arith(ArithOp::Div, &a, &c, &float),
            arith(ArithOp::Mod, &a, &c, &int),
            arith(ArithOp::Mul, &a, &lit(Value::Int(2), int.clone()), &int),
            arith(ArithOp::Sub, &b, &a, &float),
        ];
        // the vectorized filters give the same results as row-by-row evaluation
        let ts = ColumnType::new(crate::types::TypeKind::Timestamp, true);
        let vrows: Vec<StoredRow> = (0..40)
            .map(|i| {
                let a = if i % 7 == 0 { Value::Null } else { Value::Int(i - 20) };
                let b = match i % 9 {
                    0 => Value::Null,
                    1 => Value::Float(f64::NAN),
                    _ => Value::Float((i as f64 - 20.0) / 3.0),
                };
                let t = if i % 5 == 0 { Value::Null } else { Value::Timestamp(i * 86_400_000_000 / 2) };
                let s = if i % 6 == 0 { Value::Null } else { Value::String(format!("k{}", i % 13)) };
                StoredRow { rowid: i, cells: vec![a, b, t, s], errors: Vec::new(), pos: Vec::new() }
            })
            .collect();
        let tc = col(2, ts.clone());
        let string = ColumnType::string(true);
        let sc = col(3, string.clone());
        let vexprs = vec![
            cmp(CmpOp::Eq, &a, &lit(Value::Int(3), int.clone())),
            cmp(CmpOp::Lt, &a, &lit(Value::Float(2.5), float.clone())),
            cmp(CmpOp::Ge, &b, &lit(Value::Float(0.0), float.clone())),
            cmp(CmpOp::Ne, &b, &lit(Value::Int(1), int.clone())),
            cmp(CmpOp::Gt, &b, &lit(Value::Float(f64::NAN), float.clone())),
            cmp(CmpOp::Eq, &b, &lit(Value::Float(f64::NAN), float.clone())),
            cmp(CmpOp::Le, &tc, &lit(Value::Timestamp(10 * 86_400_000_000), ts.clone())),
            cmp(CmpOp::Lt, &tc, &lit(Value::Date(6), ColumnType::new(crate::types::TypeKind::Date, true))),
            cmp(CmpOp::Gt, &lit(Value::Int(0), int.clone()), &a),
            cmp(CmpOp::Eq, &sc, &lit(Value::String("k3".into()), string.clone())),
            cmp(CmpOp::Lt, &sc, &lit(Value::String("k12".into()), string.clone())),
            or(vec![cmp(CmpOp::Ge, &sc, &lit(Value::String("k7".into()), string.clone())), node(ExprKind::IsNull(sc.clone()), boolean.clone())]),
            and(vec![cmp(CmpOp::Gt, &a, &lit(Value::Int(-5), int.clone())), cmp(CmpOp::Lt, &b, &lit(Value::Float(4.0), float.clone()))]),
            or(vec![cmp(CmpOp::Gt, &a, &lit(Value::Int(10), int.clone())), node(ExprKind::IsNull(b.clone()), boolean.clone())]),
            not(and(vec![cmp(CmpOp::Ne, &a, &lit(Value::Int(1), int.clone())), not(node(ExprKind::IsNull(tc.clone()), boolean.clone()))])),
        ];
        for e in &vexprs {
            let f = Fast::compile(e, &|_| Some(0));
            let v = VPred::compile(&f).unwrap_or_else(|| panic!("not vectorized: {}", crate::expr::display::display(e)));
            let mut cols = Vec::new();
            v.columns(&mut cols);
            let typed: Vec<(ColId, std::sync::Arc<ColChunk>)> =
                cols.iter().map(|c| (*c, std::sync::Arc::new(ColChunk::of(&vrows, *c)))).collect();
            let mut out = vec![0u8; vrows.len()];
            assert!(v.eval(&typed, 0..vrows.len(), &mut out));
            for (r, got) in vrows.iter().zip(&out) {
                let expected = match f.eval_bool(&[Some(r)], &Cells(r)).unwrap() {
                    Some(true) => V_TRUE,
                    Some(false) => V_FALSE,
                    None => V_NULL,
                };
                assert_eq!(*got, expected, "{} row {}", crate::expr::display::display(e), r.rowid);
            }
        }
        for e in &exprs {
            let f = Fast::compile(e, &|_| Some(0));
            for r in &rows {
                let slots = [Some(r)];
                let ctx = Cells(r);
                let expected = eval(e, &ctx);
                let got = f.eval(&slots, &ctx).map(Cow::into_owned);
                assert_eq!(format!("{expected:?}"), format!("{got:?}"), "{}", crate::expr::display::display(e));
            }
        }
    }
}
