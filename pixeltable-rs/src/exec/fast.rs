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
    Similarity { node: ExprRef, slot: usize, col: ColId, metric: String, query: Value },
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
                        metric: sim.metric.clone(),
                        query: query.clone(),
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
    pub fn eval<'v>(&'v self, row: &'v [Option<&'v StoredRow>], ctx: &dyn RowContext) -> Result<Cow<'v, Value>> {
        static NULL: Value = Value::Null;
        Ok(match self {
            Fast::Cell { slot, col } => Cow::Borrowed(row[*slot].map_or(&NULL, |r| r.cell(*col))),
            Fast::Lit(v) => Cow::Borrowed(v),
            Fast::Other(e) => Cow::Owned(eval(e, ctx)?),
            Fast::Similarity { node, slot, col, metric, query } => {
                if ctx.python_semantics() {
                    let err = crate::pxt_err!(UnsupportedOperation, "similarity(): cannot be used in a computed column");
                    return Err(locate(node, err, ctx));
                }
                let stored = row[*slot].map_or(&NULL, |r| r.cell(*col));
                Cow::Owned(crate::expr::eval::similarity_score(metric, stored, query))
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
