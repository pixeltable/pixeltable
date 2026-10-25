//! `str(expr)`, as upstream renders it (used in messages and `repr()`s).

use super::*;
use pyo3::prelude::*;
use crate::error::py_repr;
use crate::value::{Json, py_float_repr};

pub fn display(e: &Expr) -> String {
    match &e.kind {
        ExprKind::ColumnRef(c) => c.name.clone(),
        ExprKind::Literal(v) => literal_str(v, &e.col_type),
        ExprKind::Arithmetic { op, lhs, rhs } => {
            let wrap = |x: &ExprRef| {
                if matches!(x.kind, ExprKind::Arithmetic { .. }) { format!("({})", display(x)) } else { display(x) }
            };
            format!("{} {} {}", wrap(lhs), op.symbol(), wrap(rhs))
        }
        ExprKind::StringOp { op, lhs, rhs } => {
            let wrap = |x: &ExprRef| {
                if matches!(x.kind, ExprKind::StringOp { .. }) { format!("({})", display(x)) } else { display(x) }
            };
            format!("{} {} {}", wrap(lhs), op.symbol(), wrap(rhs))
        }
        ExprKind::Comparison { op, lhs, rhs } => format!("{} {} {}", display(lhs), op.symbol(), display(rhs)),
        ExprKind::Compound { op, operands } => {
            if *op == LogicalOp::Not {
                format!("~({})", operands.first().map(|o| display(o)).unwrap_or_default())
            } else {
                operands.iter().map(|o| format!("({})", display(o))).collect::<Vec<_>>().join(&format!(" {} ", op.symbol()))
            }
        }
        ExprKind::IsNull(x) => format!("{} == None", display(x)),
        ExprKind::InList { lhs, display: values, .. } => format!("{}.isin({})", display(lhs), values),
        ExprKind::InExpr { lhs, rhs } => format!("{}.isin({})", display(lhs), display(rhs)),
        ExprKind::TypeCast(x) => format!("{}.astype({})", display(x), e.col_type.repr()),
        ExprKind::InlineList(items) | ExprKind::InlineArray(items) => {
            format!("[{}]", items.iter().map(|i| display(i)).collect::<Vec<_>>().join(", "))
        }
        ExprKind::InlineDict(items) => format!(
            "{{{}}}",
            items.iter().map(|(k, v)| format!("'{}': {}", k, display(v))).collect::<Vec<_>>().join(", ")
        ),
        ExprKind::JsonPath { base, path } => {
            let anchor = display(base);
            if path.is_empty() {
                return anchor;
            }
            let sep = if matches!(path.first(), Some(JsonPathElem::Key(_))) { "." } else { "" };
            format!("{anchor}{sep}{}", json_path_str(path))
        }
        ExprKind::RelativeRoot { .. } => "R".to_string(),
        ExprKind::JsonMapper { op, asc, src, target, .. } => match op {
            MapperOp::Map => format!("{}.map(lambda R: {})", display(src), display(target)),
            MapperOp::Filter => format!("{}.filter(lambda R: {})", display(src), display(target)),
            MapperOp::Sort => {
                let asc_repr = if *asc { "" } else { ", asc=False" };
                format!("{}.sort(lambda R: {}{asc_repr})", display(src), display(target))
            }
        },
        ExprKind::ArraySlice { base, index } => {
            let parts: Vec<String> = index
                .iter()
                .map(|i| match i {
                    crate::types::ArrayIndex::Int(n) => n.to_string(),
                    crate::types::ArrayIndex::Slice(a, b, c) => slice_str(*a, *b, *c),
                })
                .collect();
            format!("{}[{}]", display(base), parts.join(", "))
        }
        ExprKind::ColumnProperty { col, prop } => format!("{}.{}", display(col), prop.name()),
        ExprKind::Similarity(sim) => format!("{}.similarity({}, {})", display(&sim.col), display(&sim.item), py_repr(&sim.idx_name)),
        ExprKind::Variable(name) => name.clone(),
        ExprKind::RowidRef { .. } => String::new(),
        ExprKind::FunctionCall(fc) => crate::functions::display_call(fc),
    }
}

/// How an error message shows the value of an expression (upstream `ColumnType.print_value()`): `str()` of the value,
/// quoted for strings and UUIDs (also inside a JSON value).
pub fn print_value(v: &Value, t: &ColumnType) -> String {
    if t.is_string() || t.is_uuid() {
        return format!("'{}'", crate::value::py_str(v));
    }
    match v {
        Value::Json(Json::String(s)) => format!("'{s}'"),
        v => crate::value::py_str(v),
    }
}

/// Upstream `Expr.print_list()`.
pub fn print_list(items: &[ExprRef]) -> String {
    if items.len() == 1 {
        return display(&items[0]);
    }
    format!("({})", items.iter().map(|e| display(e)).collect::<Vec<_>>().join(", "))
}

/// The path part of a JSON path's display (upstream `JsonPath._json_path()`).
pub fn json_path_str(path: &[JsonPathElem]) -> String {
    let mut out = String::new();
    for (i, el) in path.iter().enumerate() {
        match el {
            JsonPathElem::Wildcard => out.push_str("[*]"),
            JsonPathElem::Key(k) => {
                if i > 0 {
                    out.push('.');
                }
                out.push_str(k);
            }
            JsonPathElem::Index(n) => out.push_str(&format!("[{n}]")),
            JsonPathElem::Slice(a, b, c) => out.push_str(&format!("[{}]", slice_str(*a, *b, *c))),
        }
    }
    out
}

fn slice_str(a: Option<i64>, b: Option<i64>, c: Option<i64>) -> String {
    let f = |x: Option<i64>| x.map(|n| n.to_string()).unwrap_or_default();
    match c {
        Some(_) => format!("{}:{}:{}", f(a), f(b), f(c)),
        None => format!("{}:{}", f(a), f(b)),
    }
}

/// Python's `str()` of a JSON value converted to Python objects (dicts and lists use reprs of their items).
pub fn json_py_str(j: &Json, top_level: bool) -> String {
    match j {
        Json::Null => "None".into(),
        Json::Bool(b) => if *b { "True".into() } else { "False".into() },
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                py_float_repr(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Json::String(s) => if top_level { s.clone() } else { py_repr(s) },
        Json::Array(items) => format!("[{}]", items.iter().map(|i| json_py_str(i, false)).collect::<Vec<_>>().join(", ")),
        Json::Object(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, v)| format!("{}: {}", py_repr(k), json_py_str(v, false))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// `str(Literal(v))`.
pub fn literal_str(v: &Value, _t: &ColumnType) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Int(i) => i.to_string(),
        Value::Float(f) => py_float_repr(*f),
        Value::String(s) => format!("'{s}'"),
        Value::Media(s) => s.clone(),
        Value::Uuid(u) => format!("'{u}'"),
        Value::Date(d) => crate::value::date_from_days(*d).map(|d| format!("'{}'", d.format("%Y-%m-%d"))).unwrap_or_default(),
        Value::Timestamp(us) => format!("'{}'", crate::value::time::isoformat(&crate::value::time::to_local(*us), 'T')),
        Value::Binary(b) => format!("{b:?}"),
        Value::Json(j) => json_py_str(j, true),
        Value::Array(a) => pyo3::Python::attach(|py| {
            crate::value::py::ndarray_to_py(py, a)
                .and_then(|arr| arr.call_method0("tolist"))
                .and_then(|l| l.str().map(|s| s.to_string()))
                .unwrap_or_else(|_| "<array>".to_string())
        }),
        Value::PyObj(o) => pyo3::Python::attach(|py| o.bind(py).str().map(|s| s.to_string()).unwrap_or_default()),
    }
}
