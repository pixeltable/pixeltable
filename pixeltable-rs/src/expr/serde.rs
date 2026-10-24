//! JSON (de)serialization of expressions.
//!
//! Format: `{"_classname": <node kind>, ...fields, "components": [children]}`, close to upstream's `as_dict()`.
//! Literal values use a tagged encoding (`value_to_json`) so they round-trip exactly; column types use the
//! upstream `as_dict()` form. Function calls carry a `FuncRef` descriptor (see `functions`).

use base64::Engine;
use serde_json::{Map, json};
use std::sync::Arc;

use super::*;
use crate::error::{Error, Result};
use crate::types::ArrayIndex;
use crate::value::{Json, NdArray};

fn bad(d: &Json) -> Error {
    Error::internal(format!("malformed serialized expression: {d}"))
}

/// Tagged JSON encoding of a value.
pub fn value_to_json(v: &Value) -> Result<Json> {
    Ok(match v {
        Value::Null => Json::Null,
        Value::Bool(b) => json!(b),
        Value::Int(i) => json!(i),
        Value::Float(f) => {
            if f.is_finite() {
                json!({ "$float": f })
            } else {
                json!({ "$float": f.to_string() })
            }
        }
        Value::String(s) => json!(s),
        Value::Timestamp(us) => json!({ "$ts": us }),
        Value::Date(d) => json!({ "$date": d }),
        Value::Uuid(u) => json!({ "$uuid": u.to_string() }),
        Value::Binary(b) => json!({ "$b64": base64::engine::general_purpose::STANDARD.encode(b) }),
        Value::Json(j) => json!({ "$json": j }),
        Value::Media(p) => json!({ "$media": p }),
        Value::Array(a) => json!({ "$array": serde_json::to_value(a.as_ref())? }),
        Value::PyObj(_) => return Err(Error::internal("cannot serialize an in-flight Python object")),
    })
}

pub fn value_from_json(j: &Json) -> Result<Value> {
    Ok(match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Value::Int(i),
            None => Value::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Json::String(s) => Value::String(s.clone()),
        Json::Object(m) => {
            let (tag, val) = m.iter().next().ok_or_else(|| bad(j))?;
            match tag.as_str() {
                "$float" => match val {
                    Json::String(s) => Value::Float(s.parse::<f64>().unwrap_or(f64::NAN)),
                    other => Value::Float(other.as_f64().ok_or_else(|| bad(j))?),
                },
                "$ts" => Value::Timestamp(val.as_i64().ok_or_else(|| bad(j))?),
                "$date" => Value::Date(val.as_i64().and_then(|d| i32::try_from(d).ok()).ok_or_else(|| bad(j))?),
                "$uuid" => Value::Uuid(val.as_str().and_then(|s| uuid::Uuid::parse_str(s).ok()).ok_or_else(|| bad(j))?),
                "$b64" => Value::Binary(
                    val.as_str()
                        .and_then(|s| base64::engine::general_purpose::STANDARD.decode(s).ok())
                        .ok_or_else(|| bad(j))?,
                ),
                "$json" => Value::Json(val.clone()),
                "$media" => Value::Media(val.as_str().ok_or_else(|| bad(j))?.to_string()),
                "$array" => Value::Array(Arc::new(serde_json::from_value::<NdArray>(val.clone())?)),
                _ => return Err(bad(j)),
            }
        }
        Json::Array(_) => return Err(bad(j)),
    })
}

fn components(children: &[&ExprRef]) -> Result<Json> {
    Ok(Json::Array(children.iter().map(|c| to_json(c)).collect::<Result<_>>()?))
}

fn index_to_json(i: &ArrayIndex) -> Json {
    match i {
        ArrayIndex::Int(n) => json!(n),
        ArrayIndex::Slice(a, b, c) => json!([a, b, c]),
    }
}

fn index_from_json(j: &Json) -> Result<ArrayIndex> {
    match j {
        Json::Number(n) => Ok(ArrayIndex::Int(n.as_i64().ok_or_else(|| bad(j))?)),
        Json::Array(a) if a.len() == 3 => Ok(ArrayIndex::Slice(a[0].as_i64(), a[1].as_i64(), a[2].as_i64())),
        _ => Err(bad(j)),
    }
}

fn path_elem_to_json(p: &JsonPathElem) -> Json {
    match p {
        JsonPathElem::Key(k) => json!({ "key": k }),
        JsonPathElem::Index(i) => json!({ "index": i }),
        JsonPathElem::Slice(a, b, c) => json!({ "slice": [a, b, c] }),
        JsonPathElem::Wildcard => json!("*"),
    }
}

fn path_elem_from_json(j: &Json) -> Result<JsonPathElem> {
    if j.as_str() == Some("*") {
        return Ok(JsonPathElem::Wildcard);
    }
    if let Some(k) = j.get("key").and_then(Json::as_str) {
        return Ok(JsonPathElem::Key(k.to_string()));
    }
    if let Some(i) = j.get("index").and_then(Json::as_i64) {
        return Ok(JsonPathElem::Index(i));
    }
    if let Some(Json::Array(a)) = j.get("slice") {
        if a.len() == 3 {
            return Ok(JsonPathElem::Slice(a[0].as_i64(), a[1].as_i64(), a[2].as_i64()));
        }
    }
    Err(bad(j))
}

fn parse_id(j: Option<&Json>) -> Option<u64> {
    match j? {
        Json::String(s) => s.parse().ok(),
        Json::Number(n) => n.as_u64(),
        _ => None,
    }
}

pub fn to_json(e: &Expr) -> Result<Json> {
    let mut m = Map::new();
    let classname = match &e.kind {
        ExprKind::ColumnRef(c) => {
            m.insert("tbl_id".into(), json!(c.tbl_id.to_string()));
            m.insert("col_id".into(), json!(c.col_id));
            m.insert("name".into(), json!(c.name));
            m.insert("tbl_version".into(), json!(c.tbl_version));
            "ColumnRef"
        }
        ExprKind::Literal(v) => {
            m.insert("val".into(), value_to_json(v)?);
            "Literal"
        }
        ExprKind::Arithmetic { op, .. } => {
            m.insert("operator".into(), json!(*op as i64));
            "ArithmeticExpr"
        }
        ExprKind::StringOp { op, .. } => {
            m.insert("operator".into(), json!(*op as i64));
            "StringOp"
        }
        ExprKind::Comparison { op, .. } => {
            m.insert("operator".into(), json!(*op as i64));
            "Comparison"
        }
        ExprKind::Compound { op, .. } => {
            m.insert("operator".into(), json!(*op as i64));
            "CompoundPredicate"
        }
        ExprKind::IsNull(_) => "IsNull",
        ExprKind::InList { values, display, .. } => {
            m.insert("value_list".into(), Json::Array(values.iter().map(value_to_json).collect::<Result<_>>()?));
            m.insert("display".into(), json!(display));
            "InPredicate"
        }
        ExprKind::InExpr { .. } => "InPredicate",
        ExprKind::TypeCast(_) => "TypeCast",
        ExprKind::InlineList(_) => "InlineList",
        ExprKind::InlineArray(_) => "InlineArray",
        ExprKind::InlineDict(items) => {
            m.insert("keys".into(), Json::Array(items.iter().map(|(k, _)| json!(k)).collect()));
            "InlineDict"
        }
        ExprKind::JsonPath { path, .. } => {
            m.insert("path".into(), Json::Array(path.iter().map(path_elem_to_json).collect()));
            "JsonPath"
        }
        ExprKind::ArraySlice { index, .. } => {
            m.insert("index".into(), Json::Array(index.iter().map(index_to_json).collect()));
            "ArraySlice"
        }
        ExprKind::RelativeRoot { id } => {
            m.insert("root_id".into(), json!(id.to_string()));
            "RelativeRoot"
        }
        ExprKind::JsonMapper { op, asc, root_id, .. } => {
            m.insert("op".into(), json!(*op as i64));
            m.insert("asc".into(), json!(asc));
            m.insert("root_id".into(), json!(root_id.to_string()));
            "JsonMapper"
        }
        ExprKind::FunctionCall(fc) => {
            m.insert("fn".into(), fc.func.to_json()?);
            m.insert("kwarg_names".into(), Json::Array(fc.kwargs.iter().map(|(k, _)| json!(k)).collect()));
            m.insert("num_group_by".into(), json!(fc.group_by.len()));
            m.insert("num_order_by".into(), json!(fc.order_by.len()));
            m.insert("is_method_call".into(), json!(fc.is_method_call));
            "FunctionCall"
        }
    };
    m.insert("_classname".into(), json!(classname));
    m.insert("col_type".into(), e.col_type.to_dict());
    m.insert("components".into(), components(&e.kind.children())?);
    Ok(Json::Object(m))
}

/// Deserializes an expression. Column references are taken as stored (callers that need to re-resolve names
/// against the current catalog do so separately).
pub fn from_json(d: &Json) -> Result<ExprRef> {
    let classname = d.get("_classname").and_then(Json::as_str).ok_or_else(|| bad(d))?;
    let col_type = ColumnType::from_dict(d.get("col_type").ok_or_else(|| bad(d))?)?;
    let comps: Vec<ExprRef> = match d.get("components") {
        Some(Json::Array(items)) => items.iter().map(from_json).collect::<Result<_>>()?,
        _ => Vec::new(),
    };
    let op_code = || d.get("operator").and_then(Json::as_i64).ok_or_else(|| bad(d));
    let two = |comps: &[ExprRef]| -> Result<(ExprRef, ExprRef)> {
        match comps {
            [a, b] => Ok((a.clone(), b.clone())),
            _ => Err(bad(d)),
        }
    };
    let one = |comps: &[ExprRef]| -> Result<ExprRef> { comps.first().cloned().ok_or_else(|| bad(d)) };
    let kind = match classname {
        "ColumnRef" => ExprKind::ColumnRef(ColumnRef {
            tbl_id: d
                .get("tbl_id")
                .and_then(Json::as_str)
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
                .ok_or_else(|| bad(d))?,
            col_id: d.get("col_id").and_then(Json::as_u64).and_then(|v| u32::try_from(v).ok()).ok_or_else(|| bad(d))?,
            name: d.get("name").and_then(Json::as_str).unwrap_or_default().to_string(),
            tbl_version: d.get("tbl_version").and_then(Json::as_i64),
        }),
        "Literal" => ExprKind::Literal(value_from_json(d.get("val").ok_or_else(|| bad(d))?)?),
        "ArithmeticExpr" => {
            let (lhs, rhs) = two(&comps)?;
            ExprKind::Arithmetic { op: ArithOp::from_i64(op_code()?).ok_or_else(|| bad(d))?, lhs, rhs }
        }
        "StringOp" => {
            let (lhs, rhs) = two(&comps)?;
            let op = if op_code()? == 0 { StringOp::Concat } else { StringOp::Repeat };
            ExprKind::StringOp { op, lhs, rhs }
        }
        "Comparison" => {
            let (lhs, rhs) = two(&comps)?;
            ExprKind::Comparison { op: CmpOp::from_i64(op_code()?).ok_or_else(|| bad(d))?, lhs, rhs }
        }
        "CompoundPredicate" => {
            ExprKind::Compound { op: LogicalOp::from_i64(op_code()?).ok_or_else(|| bad(d))?, operands: comps }
        }
        "IsNull" => ExprKind::IsNull(one(&comps)?),
        "InPredicate" => match d.get("value_list") {
            Some(Json::Array(vals)) => ExprKind::InList {
                lhs: one(&comps)?,
                values: vals.iter().map(value_from_json).collect::<Result<_>>()?,
                display: d.get("display").and_then(Json::as_str).unwrap_or_default().to_string(),
            },
            _ => {
                let (lhs, rhs) = two(&comps)?;
                ExprKind::InExpr { lhs, rhs }
            }
        },
        "TypeCast" => ExprKind::TypeCast(one(&comps)?),
        "InlineList" => ExprKind::InlineList(comps),
        "InlineArray" => ExprKind::InlineArray(comps),
        "InlineDict" => {
            let keys: Vec<String> = match d.get("keys") {
                Some(Json::Array(k)) => k.iter().filter_map(|k| k.as_str().map(str::to_owned)).collect(),
                _ => return Err(bad(d)),
            };
            if keys.len() != comps.len() {
                return Err(bad(d));
            }
            ExprKind::InlineDict(keys.into_iter().zip(comps).collect())
        }
        "JsonPath" => ExprKind::JsonPath {
            base: one(&comps)?,
            path: match d.get("path") {
                Some(Json::Array(p)) => p.iter().map(path_elem_from_json).collect::<Result<_>>()?,
                _ => return Err(bad(d)),
            },
        },
        "ArraySlice" => ExprKind::ArraySlice {
            base: one(&comps)?,
            index: match d.get("index") {
                Some(Json::Array(p)) => p.iter().map(index_from_json).collect::<Result<_>>()?,
                _ => return Err(bad(d)),
            },
        },
        "RelativeRoot" => ExprKind::RelativeRoot { id: parse_id(d.get("root_id")).ok_or_else(|| bad(d))? },
        "JsonMapper" => {
            let (src, target) = two(&comps)?;
            ExprKind::JsonMapper {
                op: MapperOp::from_i64(d.get("op").and_then(Json::as_i64).unwrap_or(0)).ok_or_else(|| bad(d))?,
                asc: d.get("asc").and_then(Json::as_bool).unwrap_or(true),
                src,
                target,
                root_id: parse_id(d.get("root_id")).ok_or_else(|| bad(d))?,
            }
        }
        "FunctionCall" => {
            let func = crate::functions::FuncRef::from_json(d.get("fn").ok_or_else(|| bad(d))?)?;
            let kwarg_names: Vec<String> = match d.get("kwarg_names") {
                Some(Json::Array(k)) => k.iter().filter_map(|k| k.as_str().map(str::to_owned)).collect(),
                _ => Vec::new(),
            };
            let n_group_by = d.get("num_group_by").and_then(Json::as_u64).unwrap_or(0) as usize;
            let n_order_by = d.get("num_order_by").and_then(Json::as_u64).unwrap_or(0) as usize;
            let n_args = comps.len().checked_sub(kwarg_names.len() + n_group_by + n_order_by).ok_or_else(|| bad(d))?;
            let mut comps = comps;
            let order_by = comps.split_off(comps.len() - n_order_by);
            let group_by = comps.split_off(comps.len() - n_group_by);
            let kw_vals = comps.split_off(n_args);
            ExprKind::FunctionCall(FunctionCall {
                func,
                args: comps,
                kwargs: kwarg_names.into_iter().zip(kw_vals).collect(),
                group_by,
                order_by,
                is_method_call: d.get("is_method_call").and_then(Json::as_bool).unwrap_or(false),
            })
        }
        _ => return Err(bad(d)),
    };
    Ok(Expr::new(kind, col_type))
}
