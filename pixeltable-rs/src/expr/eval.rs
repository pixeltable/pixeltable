//! Row-at-a-time evaluation.
//!
//! Semantics follow what upstream produces on its usual (SQL) evaluation path: NULL propagates through operators,
//! comparisons with NULL are NULL (and a NULL predicate filters the row out), division by zero yields NULL, and
//! `//` is floor division.

use std::cmp::Ordering;

use super::*;
use crate::error::{Error, Result};
use crate::value::{Json, cmp_values, values_equal};

/// Supplies column values for the row being evaluated.
pub trait RowContext {
    fn column_value(&self, col: &ColumnRef) -> Result<Value>;

    /// The error recorded for (or, for read-validated media, found in) the column's cell, if any.
    fn column_error(&self, _col: &ColumnRef) -> Result<Option<crate::store::rows::CellError>> {
        Ok(None)
    }
}

pub fn eval(e: &Expr, ctx: &dyn RowContext) -> Result<Value> {
    match &e.kind {
        ExprKind::ColumnRef(c) => ctx.column_value(c),
        ExprKind::Literal(v) => Ok(v.clone()),
        ExprKind::Arithmetic { op, lhs, rhs } => {
            let a = eval(lhs, ctx)?;
            let b = eval(rhs, ctx)?;
            arithmetic(*op, &a, &b, lhs.col_type.is_json(), rhs.col_type.is_json(), &e.col_type)
        }
        ExprKind::StringOp { op, lhs, rhs } => {
            let a = eval(lhs, ctx)?;
            let b = eval(rhs, ctx)?;
            Ok(match (op, &a, &b) {
                (_, Value::Null, _) | (_, _, Value::Null) => Value::Null,
                (StringOp::Concat, Value::String(x), Value::String(y)) => Value::String(format!("{x}{y}")),
                (StringOp::Repeat, Value::String(s), Value::Int(n)) | (StringOp::Repeat, Value::Int(n), Value::String(s)) => {
                    Value::String(s.repeat((*n).max(0) as usize))
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
            if v.is_null() || set.is_null() {
                return Ok(Value::Null);
            }
            let items: Vec<Value> = match &set {
                Value::Json(Json::Array(a)) => a.iter().map(|j| Value::Json(j.clone()).json_scalar()).collect(),
                Value::Json(Json::Object(m)) => m.keys().map(|k| Value::String(k.clone())).collect(),
                other => vec![other.json_scalar()],
            };
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
        ExprKind::JsonPath { .. } | ExprKind::ArraySlice { .. } => {
            Err(Error::not_implemented(format!("evaluation of `{}`", super::display::display(e))))
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

fn floor_div_i64(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) { q - 1 } else { q }
}

pub fn arithmetic(op: ArithOp, a: &Value, b: &Value, a_json: bool, b_json: bool, result_type: &ColumnType) -> Result<Value> {
    let a = numeric_operand(a, a_json, "left", op)?;
    let b = numeric_operand(b, b_json, "right", op)?;
    if a.is_null() || b.is_null() {
        return Ok(Value::Null);
    }
    let overflow = || crate::pxt_err!(UnsupportedOperation, "integer out of range in `{}`", op.symbol());
    if let (Value::Int(x), Value::Int(y), true) = (&a, &b, result_type.is_int()) {
        let (x, y) = (*x, *y);
        return Ok(match op {
            ArithOp::Add => Value::Int(x.checked_add(y).ok_or_else(overflow)?),
            ArithOp::Sub => Value::Int(x.checked_sub(y).ok_or_else(overflow)?),
            ArithOp::Mul => Value::Int(x.checked_mul(y).ok_or_else(overflow)?),
            ArithOp::Mod => {
                if y == 0 {
                    Value::Null
                } else {
                    Value::Int(x.checked_rem(y).ok_or_else(overflow)?)
                }
            }
            ArithOp::FloorDiv => {
                if y == 0 {
                    Value::Null
                } else {
                    Value::Int(floor_div_i64(x, y))
                }
            }
            ArithOp::Pow => {
                let exp = u32::try_from(y).map_err(|_| overflow())?;
                Value::Int(x.checked_pow(exp).ok_or_else(overflow)?)
            }
            ArithOp::Div => {
                if y == 0 {
                    Value::Null
                } else {
                    Value::Float(x as f64 / y as f64)
                }
            }
        });
    }
    let (x, y) = (a.as_f64().unwrap_or(f64::NAN), b.as_f64().unwrap_or(f64::NAN));
    Ok(match op {
        ArithOp::Add => Value::Float(x + y),
        ArithOp::Sub => Value::Float(x - y),
        ArithOp::Mul => Value::Float(x * y),
        ArithOp::Div => {
            if y == 0.0 {
                Value::Null
            } else {
                Value::Float(x / y)
            }
        }
        ArithOp::Mod => {
            if y == 0.0 {
                Value::Null
            } else {
                // Python float modulo: the result has the sign of the divisor
                let r = x % y;
                Value::Float(if r != 0.0 && (r < 0.0) != (y < 0.0) { r + y } else { r })
            }
        }
        ArithOp::FloorDiv => {
            if y == 0.0 {
                Value::Null
            } else {
                Value::Float((x / y).floor())
            }
        }
        ArithOp::Pow => Value::Float(x.powf(y)),
    })
}
