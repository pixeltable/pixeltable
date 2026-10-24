//! `pixeltable.functions.string`

use super::{Builtin, expect_arg};
use crate::error::{Error, Result};
use crate::types::ColumnType;
use crate::value::Value;

const MODULE: &str = "pixeltable.functions.string";

fn string_to_string(args: &[ColumnType]) -> Result<ColumnType> {
    let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
    expect_arg("string function", "self", t, t.is_string(), "String")?;
    Ok(ColumnType::string(false))
}

fn string_to_int(args: &[ColumnType]) -> Result<ColumnType> {
    let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
    expect_arg("len", "self", t, t.is_string(), "String")?;
    Ok(ColumnType::int(false))
}

fn str_arg(args: &[Value]) -> Result<&str> {
    match args.first() {
        Some(Value::String(s)) => Ok(s),
        _ => Err(Error::internal("expected a string argument")),
    }
}

fn contains_type(args: &[ColumnType]) -> Result<ColumnType> {
    let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
    expect_arg("contains", "self", t, t.is_string(), "String")?;
    if let Some(s) = args.get(1) {
        expect_arg("contains", "substr", s, s.is_string(), "String")?;
    }
    if let Some(c) = args.get(2) {
        expect_arg("contains", "case", c, c.is_bool(), "Bool")?;
    }
    Ok(ColumnType::bool(false))
}

fn contains(args: &[Value]) -> Result<Value> {
    let s = str_arg(args)?;
    let sub = match args.get(1) {
        Some(Value::String(sub)) => sub,
        _ => return Err(Error::internal("contains(): expected a string `substr`")),
    };
    let case = !matches!(args.get(2), Some(Value::Bool(false)));
    Ok(Value::Bool(if case { s.contains(sub.as_str()) } else { s.to_lowercase().contains(&sub.to_lowercase()) }))
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin {
        name: "contains",
        module: MODULE,
        params: &["self", "substr", "case"],
        required: 2,
        strict: true,
        return_type: contains_type,
        eval: contains,
    },
    Builtin {
        name: "upper",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: string_to_string,
        eval: |args| Ok(Value::String(str_arg(args)?.to_uppercase())),
    },
    Builtin {
        name: "lower",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: string_to_string,
        eval: |args| Ok(Value::String(str_arg(args)?.to_lowercase())),
    },
    Builtin {
        name: "len",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: string_to_int,
        eval: |args| Ok(Value::Int(str_arg(args)?.chars().count() as i64)),
    },
];
