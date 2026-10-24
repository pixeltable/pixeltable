//! Row-at-a-time evaluation.
//!
//! Semantics follow what upstream produces on its usual (SQL) evaluation path: NULL propagates through operators,
//! comparisons with NULL are NULL (and a NULL predicate filters the row out), division by zero yields NULL, and
//! `//` is floor division. Integer arithmetic is checked: a result outside the 64-bit range raises upstream's
//! (Postgres') "bigint out of range" error instead of wrapping or panicking.

use std::cmp::Ordering;

use super::*;
use crate::error::{Error, Result};
use crate::value::{Json, cmp_values, values_equal};

/// Supplies column values for the row being evaluated.
pub trait RowContext {
    fn column_value(&self, col: &ColumnRef) -> Result<Value>;

    /// The element a JSON mapper currently binds its relative root `id` to.
    fn relative_root(&self, _id: u64) -> Option<Value> {
        None
    }

    /// A value computed outside of row evaluation (aggregate and window function results), keyed by node.
    fn precomputed(&self, _e: &Expr) -> Option<Value> {
        None
    }

    /// The error recorded for (or, for read-validated media, found in) the column's cell, if any.
    fn column_error(&self, _col: &ColumnRef) -> Result<Option<crate::store::rows::CellError>> {
        Ok(None)
    }

    /// True when evaluating with Python semantics (upstream materializes computed columns in Python, where a
    /// division by zero raises `ZeroDivisionError`; query expressions evaluated in SQL yield NULL instead).
    fn python_semantics(&self) -> bool {
        false
    }
}

/// A context that binds a mapper's relative root to one element and delegates everything else.
struct ScopedCtx<'a> {
    parent: &'a dyn RowContext,
    root_id: u64,
    value: Value,
}

impl RowContext for ScopedCtx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        self.parent.column_value(col)
    }
    fn relative_root(&self, id: u64) -> Option<Value> {
        if id == self.root_id { Some(self.value.clone()) } else { self.parent.relative_root(id) }
    }
    fn precomputed(&self, e: &Expr) -> Option<Value> {
        self.parent.precomputed(e)
    }
    fn column_error(&self, col: &ColumnRef) -> Result<Option<crate::store::rows::CellError>> {
        self.parent.column_error(col)
    }
    fn python_semantics(&self) -> bool {
        self.parent.python_semantics()
    }
}

/// A JSON value as the value of an expression of type `t` (scalar types unwrap the JSON scalar).
pub fn json_as(j: Json, t: &ColumnType) -> Value {
    match j {
        Json::Null => Value::Null,
        j if t.is_json() => Value::Json(j),
        j => Value::Json(j).json_scalar(),
    }
}

/// Upstream `JsonPath._eval_path()`: a wildcard or slice projects the rest of the path over each element (one
/// result per element, null where it does not resolve).
fn eval_json_path(v: &Json, path: &[JsonPathElem]) -> Json {
    let mut cur = v;
    for (i, el) in path.iter().enumerate() {
        match el {
            JsonPathElem::Wildcard | JsonPathElem::Slice(..) => {
                let Json::Array(items) = cur else { return Json::Null };
                let selected: Vec<Json> = match el {
                    JsonPathElem::Slice(a, b, c) => crate::functions::string::py_slice(items, *a, *b, *c).unwrap_or_default(),
                    _ => items.clone(),
                };
                return Json::Array(selected.iter().map(|x| eval_json_path(x, &path[i + 1..])).collect());
            }
            JsonPathElem::Index(idx) => match cur {
                Json::Array(items) => {
                    let n = items.len() as i64;
                    if -n <= *idx && *idx < n {
                        cur = &items[(if *idx < 0 { idx + n } else { *idx }) as usize];
                    } else {
                        return Json::Null;
                    }
                }
                _ => return Json::Null,
            },
            JsonPathElem::Key(k) => match cur {
                Json::Object(m) => match m.get(k) {
                    Some(x) => cur = x,
                    None => return Json::Null,
                },
                _ => return Json::Null,
            },
        }
    }
    cur.clone()
}

/// Python truthiness of a value.
pub fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::String(s) | Value::Media(s) => !s.is_empty(),
        Value::Json(j) => match j {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
            Json::String(s) => !s.is_empty(),
            Json::Array(a) => !a.is_empty(),
            Json::Object(m) => !m.is_empty(),
        },
        Value::Binary(b) => !b.is_empty(),
        _ => true,
    }
}

/// Python's ordering of two values (sort keys); None where Python raises TypeError.
pub fn py_cmp_values(a: &Value, b: &Value) -> Option<Ordering> {
    use crate::value::{py_cmp_json, value_to_json};
    match (a, b) {
        (Value::Null, _) | (_, Value::Null) => None,
        (Value::Timestamp(x), Value::Timestamp(y)) => Some(x.cmp(y)),
        (Value::Date(x), Value::Date(y)) => Some(x.cmp(y)),
        (Value::Uuid(x), Value::Uuid(y)) => Some(x.cmp(y)),
        (Value::Binary(x), Value::Binary(y)) => Some(x.cmp(y)),
        _ => py_cmp_json(&value_to_json(a).ok()?, &value_to_json(b).ok()?),
    }
}

fn eval_mapper(op: MapperOp, asc: bool, src: &Expr, target: &Expr, root_id: u64, ctx: &dyn RowContext) -> Result<Value> {
    let items = match eval(src, ctx)? {
        Value::Json(Json::Array(items)) => items,
        _ => return Ok(Value::Null),
    };
    let root_type = root_type_of(target, root_id).unwrap_or_else(|| ColumnType::json(true));
    let mut targets: Vec<Value> = Vec::with_capacity(items.len());
    for it in &items {
        let scoped = ScopedCtx { parent: ctx, root_id, value: json_as(it.clone(), &root_type) };
        targets.push(eval(target, &scoped)?);
    }
    Ok(Value::Json(match op {
        MapperOp::Map => Json::Array(targets.iter().map(crate::value::value_to_json).collect::<Result<_>>()?),
        MapperOp::Filter => Json::Array(items.into_iter().zip(targets.iter()).filter(|(_, t)| py_truthy(t)).map(|(i, _)| i).collect()),
        MapperOp::Sort => {
            let mut keyed: Vec<(Value, Json)> = targets.into_iter().zip(items).collect();
            let mut ok = true;
            keyed.sort_by(|(a, _), (b, _)| {
                let ord = py_cmp_values(a, b).unwrap_or_else(|| {
                    ok = false;
                    Ordering::Equal
                });
                if asc { ord } else { ord.reverse() }
            });
            if !ok && keyed.len() > 1 {
                crate::bail!(
                    InvalidArgument,
                    "sort(): the sort keys are not orderable (e.g. a mix of incompatible types or a null value)"
                );
            }
            Json::Array(keyed.into_iter().map(|(_, j)| j).collect())
        }
    }))
}

/// The type of the relative root `id` within `e` (the root node's own type).
fn root_type_of(e: &Expr, id: u64) -> Option<ColumnType> {
    let mut found = None;
    e.visit(&mut |x| {
        if let ExprKind::RelativeRoot { id: rid } = &x.kind {
            if *rid == id && found.is_none() {
                found = Some(x.col_type.clone());
            }
        }
    });
    found
}

/// numpy basic indexing (ints and slices); an int drops its dimension.
pub fn eval_array_index(a: &crate::value::NdArray, index: &[crate::types::ArrayIndex]) -> Result<Value> {
    use crate::types::ArrayIndex;
    let shape = &a.shape;
    if index.len() > shape.len() {
        return Err(Error::py_exc(
            "IndexError",
            format!("too many indices for array: array is {}-dimensional, but {} were indexed", shape.len(), index.len()),
        ));
    }
    // per dimension: the selected positions, and whether the dimension is kept
    let mut dims: Vec<(Vec<usize>, bool)> = Vec::with_capacity(shape.len());
    for (axis, &n) in shape.iter().enumerate() {
        match index.get(axis) {
            Some(ArrayIndex::Int(i)) => {
                let ni = n as i64;
                if !(-ni <= *i && *i < ni) {
                    return Err(Error::py_exc("IndexError", format!("index {i} is out of bounds for axis {axis} with size {n}")));
                }
                dims.push((vec![(if *i < 0 { i + ni } else { *i }) as usize], false));
            }
            Some(ArrayIndex::Slice(st, en, step)) => {
                let all: Vec<usize> = (0..n).collect();
                dims.push((crate::functions::string::py_slice(&all, *st, *en, *step)?, true));
            }
            None => dims.push(((0..n).collect(), true)),
        }
    }
    let mut strides = vec![1usize; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    let mut flat: Vec<usize> = vec![0];
    for (k, (positions, _)) in dims.iter().enumerate() {
        let mut next = Vec::with_capacity(flat.len() * positions.len());
        for base in &flat {
            for p in positions {
                next.push(base + p * strides[k]);
            }
        }
        flat = next;
    }
    let out_shape: Vec<usize> = dims.iter().filter(|(_, keep)| *keep).map(|(p, _)| p.len()).collect();
    if out_shape.is_empty() {
        return Ok(a.element(flat[0]));
    }
    Ok(Value::Array(std::sync::Arc::new(a.gather(&flat, out_shape))))
}

pub fn eval(e: &Expr, ctx: &dyn RowContext) -> Result<Value> {
    if let Some(v) = ctx.precomputed(e) {
        return Ok(v);
    }
    match &e.kind {
        ExprKind::ColumnRef(c) => ctx.column_value(c),
        ExprKind::Literal(v) => Ok(v.clone()),
        ExprKind::Arithmetic { op, lhs, rhs } => {
            let a = eval(lhs, ctx)?;
            let b = eval(rhs, ctx)?;
            // upstream evaluates SQL-expressible arithmetic of a query in SQL, computed columns in Python
            let sql = !ctx.python_semantics() && crate::exec::query::is_sql_expr(e, &Default::default());
            arithmetic(*op, &a, &b, lhs.col_type.is_json(), rhs.col_type.is_json(), &e.col_type, sql)
        }
        ExprKind::StringOp { op, lhs, rhs } => {
            let a = eval(lhs, ctx)?;
            let b = eval(rhs, ctx)?;
            Ok(match (op, &a, &b) {
                (_, Value::Null, _) | (_, _, Value::Null) => Value::Null,
                (StringOp::Concat, Value::String(x), Value::String(y)) => Value::String(format!("{x}{y}")),
                (StringOp::Repeat, Value::String(s), Value::Int(n)) | (StringOp::Repeat, Value::Int(n), Value::String(s)) => {
                    let n = usize::try_from(*n).unwrap_or(0);
                    if s.len().checked_mul(n).is_none_or(|len| len > MAX_STRING_BYTES) {
                        crate::bail!(UnsupportedOperation, "requested length too large");
                    }
                    Value::String(s.repeat(n))
                }
                _ => return Err(Error::internal("string operator applied to non-string values")),
            })
        }
        ExprKind::Comparison { op, lhs, rhs } => {
            let a = eval(lhs, ctx)?.json_scalar();
            let b = eval(rhs, ctx)?.json_scalar();
            Ok(compare(*op, &a, &b))
        }
        ExprKind::Compound { op, operands } => match op {
            LogicalOp::Not => {
                let operand = operands.first().ok_or_else(|| Error::internal("~ without an operand"))?;
                let v = eval(operand, ctx)?;
                Ok(match v {
                    Value::Bool(b) => Value::Bool(!b),
                    _ => Value::Null,
                })
            }
            LogicalOp::And => {
                let mut saw_null = false;
                for o in operands {
                    match eval(o, ctx)? {
                        Value::Bool(false) => return Ok(Value::Bool(false)),
                        Value::Bool(true) => {}
                        _ => saw_null = true,
                    }
                }
                Ok(if saw_null { Value::Null } else { Value::Bool(true) })
            }
            LogicalOp::Or => {
                let mut saw_null = false;
                for o in operands {
                    match eval(o, ctx)? {
                        Value::Bool(true) => return Ok(Value::Bool(true)),
                        Value::Bool(false) => {}
                        _ => saw_null = true,
                    }
                }
                Ok(if saw_null { Value::Null } else { Value::Bool(false) })
            }
        },
        ExprKind::IsNull(x) => Ok(Value::Bool(matches!(eval(x, ctx)?, Value::Null | Value::Json(Json::Null)))),
        ExprKind::InList { lhs, values, .. } => {
            let v = eval(lhs, ctx)?;
            if v.is_null() {
                return Ok(Value::Null);
            }
            Ok(Value::Bool(values.iter().any(|x| values_equal(&v, x))))
        }
        ExprKind::InExpr { lhs, rhs } => {
            let v = eval(lhs, ctx)?;
            let set = eval(rhs, ctx)?;
            // upstream `InPredicate._normalize_value_set()`: the value set must be an iterable
            let items: Vec<Value> = match &set {
                Value::Json(Json::Array(a)) => a.iter().map(|j| Value::Json(j.clone()).json_scalar()).collect(),
                Value::Json(Json::Object(m)) => m.keys().map(|k| Value::String(k.clone())).collect(),
                Value::Json(Json::String(s)) | Value::String(s) => s.chars().map(|c| Value::String(c.to_string())).collect(),
                other => crate::bail!(
                    InvalidArgument,
                    "isin(): argument must be an Iterable (eg, list, dict, ...), not {}",
                    crate::value::py_repr_value(&other.json_scalar())
                ),
            };
            if v.is_null() {
                return Ok(Value::Bool(false));
            }
            Ok(Value::Bool(items.iter().any(|x| values_equal(&v, x))))
        }
        ExprKind::TypeCast(x) => crate::value::cast_value(eval(x, ctx)?, &e.col_type),
        ExprKind::InlineList(items) => {
            let vals = items.iter().map(|i| crate::value::value_to_json(&eval(i, ctx)?)).collect::<Result<Vec<_>>>()?;
            Ok(Value::Json(Json::Array(vals)))
        }
        ExprKind::InlineDict(items) => {
            let mut m = serde_json::Map::new();
            for (k, i) in items {
                m.insert(k.clone(), crate::value::value_to_json(&eval(i, ctx)?)?);
            }
            Ok(Value::Json(Json::Object(m)))
        }
        ExprKind::InlineArray(items) => {
            let vals = items.iter().map(|i| eval(i, ctx)).collect::<Result<Vec<_>>>()?;
            super::build::array_from_values(&vals, &e.col_type)
        }
        ExprKind::JsonPath { base, path } => match eval(base, ctx)? {
            Value::Null => Ok(Value::Null),
            Value::Json(j) => Ok(json_as(eval_json_path(&j, path), &e.col_type)),
            other => Ok(json_as(eval_json_path(&crate::value::value_to_json(&other)?, path), &e.col_type)),
        },
        ExprKind::RelativeRoot { id } => {
            ctx.relative_root(*id).ok_or_else(|| Error::internal("relative JSON path evaluated outside of its mapper"))
        }
        ExprKind::JsonMapper { op, asc, src, target, root_id } => eval_mapper(*op, *asc, src, target, *root_id, ctx),
        ExprKind::ArraySlice { base, index } => match eval(base, ctx)? {
            Value::Null => Ok(Value::Null),
            Value::Array(a) => eval_array_index(&a, index),
            other => Err(Error::internal(format!("array index applied to a {}", other.py_type_name()))),
        },
        ExprKind::Similarity(sim) => {
            if ctx.python_semantics() {
                crate::bail!(UnsupportedOperation, "similarity(): cannot be used in a computed column");
            }
            let stored = ctx.column_value(&ColumnRef {
                tbl_id: sim.idx_tbl_id,
                col_id: sim.val_col_id,
                name: String::new(),
                tbl_version: None,
            })?;
            let query = match sim.cache.get() {
                Some(v) => v.clone(),
                None => {
                    let v = eval(&sim.item_embedding, ctx)?;
                    let _ = sim.cache.set(v.clone());
                    v
                }
            };
            Ok(similarity_score(&sim.metric, &stored, &query))
        }
        ExprKind::ColumnProperty { col, prop } => {
            let c = col.as_column_ref().ok_or_else(|| Error::internal("column property of a non-column"))?;
            if prop.is_cellmd() {
                return Ok(match (prop, ctx.column_error(c)?) {
                    (_, None) => Value::Null,
                    (ColumnProp::ErrorType, Some(err)) => Value::String(err.errortype),
                    (ColumnProp::ErrorMsg, Some(err)) => Value::String(err.errormsg),
                    (_, Some(err)) => Value::Json(serde_json::json!({"errortype": err.errortype, "errormsg": err.errormsg})),
                });
            }
            match ctx.column_value(c)? {
                Value::Media(s) => match prop {
                    ColumnProp::FileUrl => Ok(Value::String(crate::media::file_url(&s))),
                    _ => {
                        let path = pyo3::Python::attach(|py| crate::media::local_path(py, &s))?;
                        Ok(Value::String(path.to_string_lossy().into_owned()))
                    }
                },
                _ => Ok(Value::Null),
            }
        }
        ExprKind::FunctionCall(fc) => crate::functions::eval_call(fc, e, ctx),
        ExprKind::Variable(name) => Err(Error::internal(format!("unbound variable {name} in an evaluated expression"))),
    }
}

/// Evaluates a predicate; NULL counts as false.
pub fn eval_predicate(e: &Expr, ctx: &dyn RowContext) -> Result<bool> {
    Ok(matches!(eval(e, ctx)?, Value::Bool(true)))
}

pub fn compare(op: CmpOp, a: &Value, b: &Value) -> Value {
    if a.is_null() || b.is_null() {
        return Value::Null;
    }
    let ord = cmp_values(a, b);
    let eq = values_equal(a, b);
    Value::Bool(match op {
        CmpOp::Eq => eq,
        CmpOp::Ne => !eq,
        CmpOp::Lt => ord == Ordering::Less,
        CmpOp::Le => ord != Ordering::Greater,
        CmpOp::Gt => ord == Ordering::Greater,
        CmpOp::Ge => ord != Ordering::Less,
    })
}

fn numeric_operand(v: &Value, is_json: bool, side: &str, op: ArithOp) -> Result<Value> {
    let v = if is_json { v.json_scalar() } else { v.clone() };
    match v {
        Value::Null | Value::Int(_) | Value::Float(_) => Ok(v),
        Value::Bool(b) if !is_json => Ok(Value::Int(b as i64)),
        other => crate::bail!(
            TypeMismatch,
            "{} requires numeric types, but {side} operand has type `{}`",
            op.symbol(),
            other.py_type_name()
        ),
    }
}

/// The largest string `*` may produce (Postgres' limit for `repeat()`, 1 GB).
const MAX_STRING_BYTES: usize = (1 << 30) - 1;

/// Python's `//` on integers (rounds toward negative infinity); None on overflow (`i64::MIN // -1`) or `b == 0`.
fn floor_div_i64(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?;
    if a.checked_rem(b)? != 0 && ((a < 0) != (b < 0)) { q.checked_sub(1) } else { Some(q) }
}

/// `a ** exp` for a non-negative exponent; None on overflow.
fn pow_i64(a: i64, exp: i64) -> Option<i64> {
    match (a, u32::try_from(exp)) {
        (_, Ok(e)) => a.checked_pow(e),
        // an exponent beyond u32 overflows unless the base is 0, 1 or -1
        (0 | 1, Err(_)) if exp > 0 => Some(a),
        (-1, Err(_)) if exp > 0 => Some(if exp % 2 == 0 { 1 } else { -1 }),
        _ => None,
    }
}

/// `a op b` on numeric values. `sql`: evaluated as upstream's SQL translation would (a zero divisor yields NULL, an
/// integer `%` truncates as in Postgres); otherwise with Python semantics (`ZeroDivisionError`, floored `%`).
pub fn arithmetic(
    op: ArithOp,
    a: &Value,
    b: &Value,
    a_json: bool,
    b_json: bool,
    result_type: &ColumnType,
    sql: bool,
) -> Result<Value> {
    let a = numeric_operand(a, a_json, "left", op)?;
    let b = numeric_operand(b, b_json, "right", op)?;
    if a.is_null() || b.is_null() {
        return Ok(Value::Null);
    }
    let zero_div = |msg: &'static str| if sql { Ok(Value::Null) } else { Err(Error::py_exc("ZeroDivisionError", msg)) };
    let overflow = || crate::pxt_err!(UnsupportedOperation, "bigint out of range");
    // int operands keep int semantics when the result is an Int, and for JSON operands (evaluated like Python)
    let int_semantics = match (&a, &b) {
        (Value::Int(_), Value::Int(y)) => (result_type.is_int() || a_json || b_json) && !(op == ArithOp::Pow && *y < 0),
        _ => false,
    };
    if let (Value::Int(x), Value::Int(y), true) = (&a, &b, int_semantics) {
        let (x, y) = (*x, *y);
        return Ok(match op {
            ArithOp::Add => Value::Int(x.checked_add(y).ok_or_else(overflow)?),
            ArithOp::Sub => Value::Int(x.checked_sub(y).ok_or_else(overflow)?),
            ArithOp::Mul => Value::Int(x.checked_mul(y).ok_or_else(overflow)?),
            ArithOp::Mod => {
                if y == 0 {
                    return zero_div("integer modulo by zero");
                }
                // truncated remainder (Postgres); i64::MIN % -1 is 0
                let r = x.wrapping_rem(y);
                // Python: the result has the sign of the divisor
                Value::Int(if !sql && r != 0 && ((r < 0) != (y < 0)) { r + y } else { r })
            }
            ArithOp::FloorDiv => {
                if y == 0 {
                    return zero_div("integer division or modulo by zero");
                }
                Value::Int(floor_div_i64(x, y).ok_or_else(overflow)?)
            }
            ArithOp::Pow => Value::Int(pow_i64(x, y).ok_or_else(overflow)?),
            ArithOp::Div => {
                if y == 0 {
                    return zero_div("division by zero");
                }
                Value::Float(x as f64 / y as f64)
            }
        });
    }
    let a_float = matches!(a, Value::Float(_)) || matches!(b, Value::Float(_));
    let (x, y) = (a.as_f64().unwrap_or(f64::NAN), b.as_f64().unwrap_or(f64::NAN));
    Ok(match op {
        ArithOp::Add => Value::Float(x + y),
        ArithOp::Sub => Value::Float(x - y),
        ArithOp::Mul => Value::Float(x * y),
        ArithOp::Div => {
            if y == 0.0 {
                return zero_div(if a_float { "float division by zero" } else { "division by zero" });
            }
            Value::Float(x / y)
        }
        ArithOp::Mod => {
            if y == 0.0 {
                return zero_div(if a_float { "float modulo by zero" } else { "integer modulo by zero" });
            }
            // Python float modulo: the result has the sign of the divisor
            let r = x % y;
            Value::Float(if r != 0.0 && (r < 0.0) != (y < 0.0) { r + y } else { r })
        }
        ArithOp::FloorDiv => {
            if y == 0.0 {
                return zero_div(if a_float { "float floor division by zero" } else { "integer division or modulo by zero" });
            }
            Value::Float((x / y).floor())
        }
        ArithOp::Pow => Value::Float(x.powf(y)),
    })
}

fn vector(v: &Value) -> Option<Vec<f64>> {
    use crate::value::ArrayData;
    let Value::Array(a) = v else { return None };
    Some(match &a.data {
        ArrayData::F32(x) => x.iter().map(|f| *f as f64).collect(),
        ArrayData::F64(x) => x.clone(),
        ArrayData::F16(bits) => bits.iter().map(|b| half_to_f64(*b)).collect(),
        ArrayData::I64(x) => x.iter().map(|i| *i as f64).collect(),
        ArrayData::I32(x) => x.iter().map(|i| *i as f64).collect(),
        _ => return None,
    })
}

fn half_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 => if frac == 0.0 { sign * f64::INFINITY } else { f64::NAN },
        e => sign * (1.0 + frac / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The similarity of two embeddings under an index metric (upstream `EmbeddingIndex.similarity_clause()`): cosine
/// similarity, inner product, or L2 distance.
pub fn similarity_score(metric: &str, a: &Value, b: &Value) -> Value {
    let (Some(x), Some(y)) = (vector(a), vector(b)) else { return Value::Null };
    if x.len() != y.len() {
        return Value::Null;
    }
    let dot: f64 = x.iter().zip(&y).map(|(p, q)| p * q).sum();
    Value::Float(match metric {
        "ip" => dot,
        "l2" => x.iter().zip(&y).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt(),
        _ => {
            let nx = x.iter().map(|p| p * p).sum::<f64>().sqrt();
            let ny = y.iter().map(|q| q * q).sum::<f64>().sqrt();
            if nx == 0.0 || ny == 0.0 { f64::NAN } else { dot / (nx * ny) }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_op(op: ArithOp, a: i64, b: i64) -> Result<Value> {
        arithmetic(op, &Value::Int(a), &Value::Int(b), false, false, &ColumnType::int(false), true)
    }

    fn overflows(r: Result<Value>) -> bool {
        matches!(r, Err(Error::Pxt { message, .. }) if message == "bigint out of range")
    }

    #[test]
    fn integer_ops_are_checked() {
        assert!(overflows(int_op(ArithOp::FloorDiv, i64::MIN, -1)));
        assert!(overflows(int_op(ArithOp::Mul, i64::MIN, -1)));
        assert!(overflows(int_op(ArithOp::Add, i64::MAX, 1)));
        assert!(overflows(int_op(ArithOp::Sub, i64::MIN, 1)));
        assert!(overflows(int_op(ArithOp::Pow, 2, 63)));
        assert!(overflows(int_op(ArithOp::Pow, 2, i64::MAX)));
        assert!(matches!(int_op(ArithOp::Mod, i64::MIN, -1), Ok(Value::Int(0))));
        assert!(matches!(int_op(ArithOp::FloorDiv, i64::MIN, 0), Ok(Value::Null)));
        assert!(matches!(int_op(ArithOp::FloorDiv, -7, 2), Ok(Value::Int(-4))));
        assert!(matches!(int_op(ArithOp::FloorDiv, 7, -2), Ok(Value::Int(-4))));
        assert!(matches!(int_op(ArithOp::FloorDiv, i64::MIN, 1), Ok(Value::Int(i64::MIN))));
        assert!(matches!(int_op(ArithOp::Pow, -1, (1 << 40) + 1), Ok(Value::Int(-1))));
        assert!(matches!(int_op(ArithOp::Pow, 1, i64::MAX), Ok(Value::Int(1))));
        assert!(matches!(int_op(ArithOp::Pow, 3, 4), Ok(Value::Int(81))));
    }
}
