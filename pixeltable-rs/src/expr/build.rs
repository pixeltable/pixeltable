//! Typed constructors: each one computes the node's type and raises the error upstream raises when the node is
//! constructed.

use super::*;
use crate::error::{Error, Result};
use crate::types::{DType, JsonSchema, JsonSpec, TypeKind};
use crate::value::{ArrayData, Json, NdArray};

fn display(kind: &ExprKind, t: &ColumnType) -> String {
    super::display::display(&Expr::new(kind.clone(), t.clone()))
}

pub fn arithmetic(op: ArithOp, lhs: ExprRef, rhs: ExprRef) -> Result<ExprRef> {
    let (t1, t2) = (&lhs.col_type, &rhs.col_type);
    let nullable = t1.nullable || t2.nullable;
    let col_type = if t1.is_json() || t2.is_json() || op == ArithOp::Div {
        ColumnType::float(nullable)
    } else if op == ArithOp::Pow {
        let nonneg_int_exp = matches!(rhs.as_literal(), Some(Value::Int(i)) if *i >= 0);
        if t1.is_int() && t2.is_int() && nonneg_int_exp { ColumnType::int(nullable) } else { ColumnType::float(nullable) }
    } else {
        t1.supertype(t2, false).unwrap_or_else(|| ColumnType::invalid(nullable))
    };
    let kind = ExprKind::Arithmetic { op, lhs: lhs.clone(), rhs: rhs.clone() };
    if !t1.is_numeric() && !t1.is_json() {
        crate::bail!(
            TypeMismatch,
            "{}: {} requires numeric types, but left operand has type `{}`",
            display(&kind, &col_type),
            op.symbol(),
            t1.repr()
        );
    }
    if !t2.is_numeric() && !t2.is_json() {
        crate::bail!(
            TypeMismatch,
            "{}: {} requires numeric types, but right operand has type `{}`",
            display(&kind, &col_type),
            op.symbol(),
            t2.repr()
        );
    }
    Ok(Expr::new(kind, col_type))
}

pub fn string_op(op: StringOp, lhs: ExprRef, rhs: ExprRef) -> Result<ExprRef> {
    let (t1, t2) = (&lhs.col_type, &rhs.col_type);
    let col_type = ColumnType::string(t1.nullable || t2.nullable);
    let kind = ExprKind::StringOp { op, lhs: lhs.clone(), rhs: rhs.clone() };
    match op {
        StringOp::Concat if !(t1.is_string() && t2.is_string()) => crate::bail!(
            TypeMismatch,
            "{}: {} on strings requires `String` and `String`, but operands have types `{}` and `{}`",
            display(&kind, &col_type),
            op.symbol(),
            t1.repr(),
            t2.repr()
        ),
        StringOp::Repeat if !((t1.is_string() && t2.is_int()) || (t1.is_int() && t2.is_string())) => crate::bail!(
            TypeMismatch,
            "{}: {} on strings requires `String` and `Int`, but operands have types `{}` and `{}`",
            display(&kind, &col_type),
            op.symbol(),
            t1.repr(),
            t2.repr()
        ),
        _ => {}
    }
    Ok(Expr::new(kind, col_type))
}

/// A comparison; a literal on the left of a column is moved to the right (upstream normalizes search arguments).
pub fn comparison(op: CmpOp, lhs: ExprRef, rhs: ExprRef) -> Result<ExprRef> {
    let (op, lhs, rhs) = if lhs.as_literal().is_some() && rhs.as_column_ref().is_some() {
        (op.reverse(), rhs, lhs)
    } else {
        (op, lhs, rhs)
    };
    Ok(Expr::new(ExprKind::Comparison { op, lhs, rhs }, ColumnType::bool(false)))
}

pub fn compound(op: LogicalOp, operands: Vec<ExprRef>) -> Result<ExprRef> {
    if op == LogicalOp::Not && operands.len() != 1 {
        return Err(Error::internal("~ takes exactly one operand"));
    }
    let nullable = operands.iter().any(|e| e.col_type.nullable);
    Ok(Expr::new(ExprKind::Compound { op, operands }, ColumnType::bool(nullable)))
}

/// `a & b` / `a | b`: the operand must be a boolean expression (Python `TypeError` otherwise).
pub fn logical_binary(op: LogicalOp, lhs: ExprRef, rhs: ExprRef) -> Result<ExprRef> {
    if !rhs.col_type.is_bool() {
        return Err(Error::Type(format!(
            "Other needs to be an expression that returns a boolean: {}",
            rhs.col_type.repr()
        )));
    }
    compound(op, vec![lhs, rhs])
}

pub fn is_null(e: ExprRef) -> ExprRef {
    Expr::new(ExprKind::IsNull(e), ColumnType::bool(false))
}

pub fn not(e: ExprRef) -> Result<ExprRef> {
    compound(LogicalOp::Not, vec![e])
}

pub fn in_list(lhs: ExprRef, values: Vec<Value>, display: String) -> Result<ExprRef> {
    if !lhs.col_type.is_scalar() {
        crate::bail!(UnsupportedOperation, "isin(): only supported for scalar types, not {}", lhs.col_type.repr());
    }
    Ok(Expr::new(ExprKind::InList { lhs, values, display }, ColumnType::bool(false)))
}

pub fn in_expr(lhs: ExprRef, rhs: ExprRef) -> Result<ExprRef> {
    if !lhs.col_type.is_scalar() {
        crate::bail!(UnsupportedOperation, "isin(): only supported for scalar types, not {}", lhs.col_type.repr());
    }
    if !rhs.col_type.is_json() {
        crate::bail!(
            InvalidArgument,
            "isin(): argument must have a JSON type, but {} has type {}",
            super::display::display(&rhs),
            rhs.col_type.repr()
        );
    }
    Ok(Expr::new(ExprKind::InExpr { lhs, rhs }, ColumnType::bool(false)))
}

/// `e.astype(t)`; a non-nullable operand stays non-nullable.
pub fn type_cast(e: ExprRef, target: ColumnType) -> ExprRef {
    let t = if e.col_type.nullable { target } else { target.with_nullable(false) };
    Expr::new(ExprKind::TypeCast(e), t)
}

fn literal_to_json(v: &Value) -> Option<Json> {
    Some(match v {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Int(i) => Json::from(*i),
        Value::Float(f) => serde_json::Number::from_f64(*f).map(Json::Number)?,
        Value::String(s) => Json::String(s.clone()),
        Value::Json(j) => j.clone(),
        _ => return None,
    })
}

/// An inline list `[e1, e2, ...]`; collapses to a JSON literal if every element is a literal.
pub fn inline_list(items: Vec<ExprRef>) -> ExprRef {
    let schema = JsonSchema {
        spec: JsonSpec::List(items.iter().map(|e| e.col_type.clone()).collect()),
        variadic: None,
        optional_keys: Default::default(),
    };
    let t = ColumnType::new(TypeKind::Json(Some(Box::new(schema))), false);
    let literal: Option<Vec<Json>> = items.iter().map(|e| e.as_literal().and_then(literal_to_json)).collect();
    match literal {
        Some(vals) => Expr::literal(Value::Json(Json::Array(vals)), t),
        None => Expr::new(ExprKind::InlineList(items), t),
    }
}

/// An inline dict `{'k': e, ...}`; collapses to a JSON literal if every value is a literal.
pub fn inline_dict(items: Vec<(String, ExprRef)>) -> ExprRef {
    let fields = items.iter().map(|(k, e)| (k.clone(), e.col_type.clone())).collect();
    let schema = JsonSchema { spec: JsonSpec::Dict(fields), variadic: None, optional_keys: Default::default() };
    let t = ColumnType::new(TypeKind::Json(Some(Box::new(schema))), false);
    let mut m = serde_json::Map::new();
    let mut all_literal = true;
    for (k, e) in &items {
        match e.as_literal().and_then(literal_to_json) {
            Some(j) => {
                m.insert(k.clone(), j);
            }
            None => {
                all_literal = false;
                break;
            }
        }
    }
    if all_literal { Expr::literal(Value::Json(Json::Object(m)), t) } else { Expr::new(ExprKind::InlineDict(items), t) }
}

/// An inline array (`pxt.array([...])`); collapses to an array literal if every element is a literal.
pub fn inline_array(items: Vec<ExprRef>) -> Result<ExprRef> {
    let mut elem_type = ColumnType::invalid(false);
    for (i, e) in items.iter().enumerate() {
        match elem_type.supertype(&e.col_type, true) {
            Some(t) => elem_type = t,
            None => crate::bail!(
                InvalidType,
                "Could not infer element type of array: element of type `{}` at index {} is not compatible with type `{}` of preceding elements",
                e.col_type.repr(),
                i,
                elem_type.repr()
            ),
        }
    }
    let col_type = if elem_type.is_scalar() {
        let dtype = DType::for_scalar(&elem_type.kind)
            .ok_or_else(|| crate::pxt_err!(InvalidType, "Element type is not a valid dtype for an array: {}", elem_type.repr()))?;
        ColumnType::new(TypeKind::Array { shape: Some(vec![Some(items.len() as u64)]), dtype: Some(dtype) }, false)
    } else if let TypeKind::Array { shape, dtype } = &elem_type.kind {
        match (shape, dtype) {
            (Some(s), Some(d)) => {
                let mut full = vec![Some(items.len() as u64)];
                full.extend(s.iter().copied());
                ColumnType::new(TypeKind::Array { shape: Some(full), dtype: Some(*d) }, false)
            }
            _ => ColumnType::new(TypeKind::Array { shape: None, dtype: *dtype }, false),
        }
    } else {
        crate::bail!(InvalidType, "Element type is not a valid dtype for an array: {}", elem_type.repr());
    };
    if items.iter().all(|e| e.as_literal().is_some()) {
        let vals: Vec<Value> = items.iter().filter_map(|e| e.as_literal().cloned()).collect();
        check_homogeneous(&vals)?;
        if let Some(arr) = literal_array(&items, &col_type) {
            return Ok(Expr::literal(Value::Array(std::sync::Arc::new(arr)), col_type));
        }
    }
    Ok(Expr::new(ExprKind::InlineArray(items), col_type))
}

fn value_shape(v: &Value) -> Vec<usize> {
    match v {
        Value::Array(a) => a.shape.clone(),
        _ => Vec::new(),
    }
}

/// Numpy's error for ragged nested sequences (`np.array([[1, 2], [3]])`).
fn check_homogeneous(vals: &[Value]) -> Result<()> {
    let shapes: Vec<Vec<usize>> = vals.iter().map(value_shape).collect();
    let Some(first) = shapes.first() else { return Ok(()) };
    if shapes.iter().all(|s| s == first) {
        return Ok(());
    }
    let mut common = 0;
    while shapes.iter().all(|s| s.len() > common && s[common] == first[common]) {
        common += 1;
    }
    let mut detected = vec![vals.len()];
    detected.extend(first[..common].iter().copied());
    let detected: Vec<Option<u64>> = detected.iter().map(|n| Some(*n as u64)).collect();
    Err(Error::Value(format!(
        "setting an array element with a sequence. The requested array has an inhomogeneous shape after {} dimensions. The detected shape was {} + inhomogeneous part.",
        detected.len(),
        crate::types::shape_repr(&detected)
    )))
}

/// Builds the array value of an inline array from its evaluated elements.
pub fn array_from_values(vals: &[Value], t: &ColumnType) -> Result<Value> {
    if vals.iter().any(Value::is_null) {
        return Err(Error::Type("Expected non-None array elements".into()));
    }
    check_homogeneous(vals)?;
    let dtype = match &t.kind {
        TypeKind::Array { dtype: Some(d), .. } => *d,
        _ => return Err(Error::not_implemented("inline arrays without a known dtype")),
    };
    let mut shape = vec![vals.len()];
    shape.extend(vals.first().map(value_shape).unwrap_or_default());
    let mut flat: Vec<Value> = Vec::new();
    for v in vals {
        match v {
            Value::Array(a) => flat.extend(array_values(a)),
            v => flat.push(v.clone()),
        }
    }
    let bad = || Error::internal("inline array element does not match the array dtype");
    let data = match dtype {
        DType::I64 => ArrayData::I64(flat.iter().map(Value::as_i64).collect::<Option<_>>().ok_or_else(bad)?),
        DType::F32 => ArrayData::F32(flat.iter().map(|v| v.as_f64().map(|f| f as f32)).collect::<Option<_>>().ok_or_else(bad)?),
        DType::F64 => ArrayData::F64(flat.iter().map(Value::as_f64).collect::<Option<_>>().ok_or_else(bad)?),
        DType::Bool => ArrayData::Bool(flat.iter().map(Value::as_bool).collect::<Option<_>>().ok_or_else(bad)?),
        DType::Str => ArrayData::Str(
            flat.iter()
                .map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
                .collect::<Option<_>>()
                .ok_or_else(bad)?,
        ),
        _ => return Err(Error::not_implemented(format!("inline arrays of dtype {}", dtype.name()))),
    };
    Ok(Value::Array(std::sync::Arc::new(NdArray { shape, data })))
}

fn literal_array(items: &[ExprRef], t: &ColumnType) -> Option<NdArray> {
    let TypeKind::Array { shape: Some(shape), dtype: Some(dtype) } = &t.kind else { return None };
    let shape: Vec<usize> = shape.iter().map(|d| d.map(|n| n as usize)).collect::<Option<_>>()?;
    let mut flat: Vec<Value> = Vec::new();
    for e in items {
        match e.as_literal()? {
            Value::Array(a) => flat.extend(array_values(a)),
            v => flat.push(v.clone()),
        }
    }
    let data = match dtype {
        DType::I64 => ArrayData::I64(flat.iter().map(Value::as_i64).collect::<Option<_>>()?),
        DType::F32 => ArrayData::F32(flat.iter().map(|v| v.as_f64().map(|f| f as f32)).collect::<Option<_>>()?),
        DType::F64 => ArrayData::F64(flat.iter().map(Value::as_f64).collect::<Option<_>>()?),
        DType::Bool => ArrayData::Bool(flat.iter().map(Value::as_bool).collect::<Option<_>>()?),
        DType::Str => ArrayData::Str(
            flat.iter().map(|v| if let Value::String(s) = v { Some(s.clone()) } else { None }).collect::<Option<_>>()?,
        ),
        _ => return None,
    };
    Some(NdArray { shape, data })
}

fn array_values(a: &NdArray) -> Vec<Value> {
    match &a.data {
        ArrayData::Bool(v) => v.iter().map(|b| Value::Bool(*b)).collect(),
        ArrayData::I64(v) => v.iter().map(|i| Value::Int(*i)).collect(),
        ArrayData::I32(v) => v.iter().map(|i| Value::Int(*i as i64)).collect(),
        ArrayData::F32(v) => v.iter().map(|f| Value::Float(*f as f64)).collect(),
        ArrayData::F64(v) => v.iter().map(|f| Value::Float(*f)).collect(),
        ArrayData::Str(v) => v.iter().map(|s| Value::String(s.clone())).collect(),
        _ => Vec::new(),
    }
}
