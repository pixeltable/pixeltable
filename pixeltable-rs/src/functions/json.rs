//! `pixeltable.functions.json`

use super::{Builtin, CallTypes};
use crate::error::{Error, Result};
use crate::types::{ColumnType, JsonSchema, JsonSpec, TypeKind};
use crate::value::{Json, Value, py_cmp_json, py_eq_json};

pub const MODULE: &str = "pixeltable.functions.json";

fn json_val(j: Json) -> Result<Value> {
    Ok(match j {
        Json::Null => Value::Null,
        j => Value::Json(j),
    })
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(crate::error::ErrorCode::InvalidArgument, msg)
}

fn py_type_name(j: &Json) -> &'static str {
    Value::Json(j.clone()).py_type_name()
}

fn number_array(j: Option<Json>, fn_name: &str) -> Result<Vec<f64>> {
    let err = || invalid(format!("{fn_name}() is only defined for a JSON array of numbers"));
    match j {
        Some(Json::Array(items)) => items.iter().map(|x| if x.is_number() { x.as_f64().ok_or_else(err) } else { Err(err()) }).collect(),
        _ => Err(err()),
    }
}

fn opt_float(x: Option<f64>) -> Result<Value> {
    Ok(x.map_or(Value::Null, Value::Float))
}

fn schema_of(t: Option<&ColumnType>) -> Option<&JsonSchema> {
    match t.map(|t| &t.kind) {
        Some(TypeKind::Json(Some(s))) => Some(s),
        _ => None,
    }
}

fn list_type(elem: ColumnType) -> ColumnType {
    ColumnType::new(
        TypeKind::Json(Some(Box::new(JsonSchema { spec: JsonSpec::List(vec![]), variadic: Some(elem), optional_keys: Default::default() }))),
        true,
    )
}

fn get_ret(ct: &CallTypes<'_>) -> Result<ColumnType> {
    if let (Some(schema), Some(Value::String(key))) = (schema_of(ct.ty(0)), ct.literal(1)) {
        if let JsonSpec::Dict(fields) = &schema.spec {
            if let Some(t) = fields.get(key) {
                return Ok(t.with_nullable(true));
            }
        }
    }
    Ok(ColumnType::json(true))
}

fn values_ret(ct: &CallTypes<'_>) -> Result<ColumnType> {
    if let Some(JsonSpec::Dict(fields)) = schema_of(ct.ty(0)).map(|s| &s.spec) {
        if !fields.is_empty() {
            if let Some(t) = ColumnType::common_supertype(fields.values(), false) {
                return Ok(list_type(t));
            }
        }
    }
    Ok(ColumnType::json(true))
}

fn variadic_elem(t: Option<&ColumnType>) -> Option<ColumnType> {
    let s = schema_of(t)?;
    match &s.spec {
        JsonSpec::List(items) if items.is_empty() => s.variadic.clone(),
        _ => None,
    }
}

fn flatten_ret(ct: &CallTypes<'_>) -> Result<ColumnType> {
    let outer = variadic_elem(ct.ty(0));
    if let Some(inner) = variadic_elem(outer.as_ref()) {
        return Ok(list_type(inner));
    }
    Ok(ColumnType::json(true))
}

fn concat_ret(ct: &CallTypes<'_>) -> Result<ColumnType> {
    let (Some(a), Some(b)) = (schema_of(ct.ty(0)), schema_of(ct.ty(1))) else { return Ok(ColumnType::json(true)) };
    let (JsonSpec::List(ai), JsonSpec::List(bi)) = (&a.spec, &b.spec) else { return Ok(ColumnType::json(true)) };
    if a.variadic.is_none() && b.variadic.is_none() {
        let mut items = ai.clone();
        items.extend(bi.iter().cloned());
        let schema = JsonSchema { spec: JsonSpec::List(items), variadic: None, optional_keys: Default::default() };
        return Ok(ColumnType::new(TypeKind::Json(Some(Box::new(schema))), true));
    }
    let mut elems: Vec<ColumnType> = ai.iter().chain(bi.iter()).cloned().collect();
    elems.extend(a.variadic.iter().cloned());
    elems.extend(b.variadic.iter().cloned());
    Ok(match ColumnType::common_supertype(elems.iter(), false) {
        Some(t) => list_type(t),
        None => ColumnType::json(true),
    })
}

fn merge_ret(ct: &CallTypes<'_>) -> Result<ColumnType> {
    let (Some(a), Some(b)) = (schema_of(ct.ty(0)), schema_of(ct.ty(1))) else { return Ok(ColumnType::json(true)) };
    let (JsonSpec::Dict(ad), JsonSpec::Dict(bd)) = (&a.spec, &b.spec) else { return Ok(ColumnType::json(true)) };
    let mut fields = ad.clone();
    for (k, v) in bd {
        fields.insert(k.clone(), v.clone());
    }
    let optional = fields
        .keys()
        .filter(|k| (bd.contains_key(*k) && b.optional_keys.contains(*k)) || (!bd.contains_key(*k) && a.optional_keys.contains(*k)))
        .cloned()
        .collect();
    let schema = JsonSchema { spec: JsonSpec::Dict(fields), variadic: None, optional_keys: optional };
    Ok(ColumnType::new(TypeKind::Json(Some(Box::new(schema))), true))
}

fn sort_ret(ct: &CallTypes<'_>) -> Result<ColumnType> {
    Ok(match ct.ty(0) {
        Some(t) if t.is_json() => t.with_nullable(true),
        _ => ColumnType::json(true),
    })
}

/// Python's `sorted()` on JSON values; None if the elements are not mutually orderable.
pub fn py_sorted(items: &[Json], asc: bool) -> Option<Vec<Json>> {
    let mut ok = true;
    let mut v: Vec<Json> = items.to_vec();
    v.sort_by(|a, b| {
        let ord = py_cmp_json(a, b).unwrap_or_else(|| {
            ok = false;
            std::cmp::Ordering::Equal
        });
        if asc { ord } else { ord.reverse() }
    });
    if ok { Some(v) } else { None }
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin::new("dumps", "(obj: Json) -> String", |a| {
        let j = a.json(0)?.unwrap_or(Json::Null);
        Ok(Value::String(crate::value::py_json_dumps(&j)))
    })
    .sql(),
    Builtin::new("len", "(self: Json) -> Int", |a| {
        let j = a.json(0)?.unwrap_or(Json::Null);
        match &j {
            Json::Array(v) => Ok(Value::Int(v.len() as i64)),
            Json::Object(m) => Ok(Value::Int(m.len() as i64)),
            Json::String(s) => Ok(Value::Int(s.chars().count() as i64)),
            other => Err(invalid(format!("len() is not defined for a JSON {}", py_type_name(other)))),
        }
    })
    .method()
    .sql(),
    Builtin::new("is_empty", "(self: Json | None) -> Bool", |a| {
        Ok(Value::Bool(match a.json(0)? {
            None | Some(Json::Null) => true,
            Some(Json::Array(v)) => v.is_empty(),
            Some(Json::Object(m)) => m.is_empty(),
            Some(Json::String(s)) => s.is_empty(),
            Some(_) => false,
        }))
    })
    .method()
    .sql(),
    Builtin::new("contains", "(self: Json, value: Json) -> Bool", |a| {
        let v = a.json(1)?.unwrap_or(Json::Null);
        Ok(Value::Bool(match a.json(0)? {
            Some(Json::Array(items)) => items.iter().any(|x| py_eq_json(x, &v)),
            Some(Json::Object(m)) => matches!(&v, Json::String(k) if m.contains_key(k)),
            _ => false,
        }))
    })
    .method(),
    Builtin::new("get", "(self: Json | None, key: String, default: Json | None = None) -> Json", |a| {
        let key = a.str(1)?.to_string();
        let default = a.json(2)?.unwrap_or(Json::Null);
        json_val(match a.json(0)? {
            Some(Json::Object(m)) => m.get(&key).cloned().unwrap_or(default),
            _ => default,
        })
    })
    .method()
    .ret(get_ret),
    Builtin::new("sum", "(self: Json) -> Float", |a| Ok(Value::Float(number_array(a.json(0)?, "sum")?.iter().sum()))).method().sql(),
    Builtin::new("min", "(self: Json) -> Float | None", |a| {
        opt_float(number_array(a.json(0)?, "min")?.into_iter().reduce(f64::min))
    })
    .method()
    .sql(),
    Builtin::new("max", "(self: Json) -> Float | None", |a| {
        opt_float(number_array(a.json(0)?, "max")?.into_iter().reduce(f64::max))
    })
    .method()
    .sql(),
    Builtin::new("mean", "(self: Json) -> Float | None", |a| {
        let nums = number_array(a.json(0)?, "mean")?;
        opt_float(if nums.is_empty() { None } else { Some(nums.iter().sum::<f64>() / nums.len() as f64) })
    })
    .method()
    .sql(),
    Builtin::new("count", "(self: Json, value: Json) -> Int", |a| {
        let v = a.json(1)?.unwrap_or(Json::Null);
        match a.json(0)? {
            Some(Json::Array(items)) => Ok(Value::Int(items.iter().filter(|x| py_eq_json(x, &v)).count() as i64)),
            _ => Err(invalid("count() is only defined for a JSON array")),
        }
    })
    .method(),
    Builtin::new("keys", "(self: Json) -> Json[(String, ...)]", |a| match a.json(0)? {
        Some(Json::Object(m)) => json_val(Json::Array(m.keys().map(|k| Json::String(k.clone())).collect())),
        _ => Err(invalid("keys() is only defined for a JSON object")),
    })
    .method(),
    Builtin::new("values", "(self: Json) -> Json", |a| match a.json(0)? {
        Some(Json::Object(m)) => json_val(Json::Array(m.values().cloned().collect())),
        _ => Err(invalid("values() is only defined for a JSON object")),
    })
    .method()
    .ret(values_ret),
    Builtin::new("items", "(self: Json) -> Json", |a| match a.json(0)? {
        Some(Json::Object(m)) => json_val(Json::Array(m.into_iter().map(|(k, v)| Json::Array(vec![Json::String(k), v])).collect())),
        _ => Err(invalid("items() is only defined for a JSON object")),
    })
    .method(),
    Builtin::new("flatten", "(self: Json) -> Json", |a| match a.json(0)? {
        Some(Json::Array(items)) => {
            let mut out = Vec::new();
            for x in items {
                match x {
                    Json::Array(inner) => out.extend(inner),
                    x => out.push(x),
                }
            }
            json_val(Json::Array(out))
        }
        _ => Err(invalid("flatten() is only defined for a JSON array")),
    })
    .method()
    .ret(flatten_ret),
    Builtin::new("concat", "(self: Json | None, other: Json | None) -> Json", |a| match (a.json(0)?, a.json(1)?) {
        (None, _) | (_, None) => Ok(Value::Null),
        (Some(Json::Array(mut x)), Some(Json::Array(y))) => {
            x.extend(y);
            json_val(Json::Array(x))
        }
        _ => Err(invalid("concat() is only defined for two JSON arrays")),
    })
    .method()
    .ret(concat_ret),
    Builtin::new("merge", "(self: Json | None, other: Json | None) -> Json", |a| match (a.json(0)?, a.json(1)?) {
        (None, _) | (_, None) => Ok(Value::Null),
        (Some(Json::Object(mut x)), Some(Json::Object(y))) => {
            for (k, v) in y {
                x.insert(k, v);
            }
            json_val(Json::Object(x))
        }
        _ => Err(invalid("merge() is only defined for two JSON objects")),
    })
    .method()
    .ret(merge_ret),
    Builtin::new("_sort", "(list_: Json | None, asc: Bool = True) -> Json", |a| match a.json(0)? {
        Some(Json::Array(items)) => match py_sorted(&items, a.bool(1)?) {
            Some(v) => json_val(Json::Array(v)),
            None => Err(invalid("sort(): the array elements are not orderable; pass a key, e.g. .sort(lambda R: R.field)")),
        },
        _ => Ok(Value::Null),
    })
    .display("sort")
    .ret(sort_ret),
];
